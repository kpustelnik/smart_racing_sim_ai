local HttpService = game:GetService("HttpService")
local Workspace = game:GetService('Workspace')
local PhysicsService = game:GetService('PhysicsService')
local RunService = game:GetService('RunService')

-- Initialize Collision Groups
PhysicsService:RegisterCollisionGroup('CAR')
PhysicsService:CollisionGroupSetCollidable('CAR', 'CAR', false)

local TrackWalls = Workspace:WaitForChild('Track'):WaitForChild('Walls')
local CheckpointsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CheckpointsController'))
local CarsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CarsController'))

-- === CONFIGURATION ===
local modelId: string = 'global_racer_v1' 
local websocketUrl: string = `ws://localhost:8000/ws/{modelId}`
local websocketClient = HttpService:CreateWebStreamClient(Enum.WebStreamClientType.WebSocket, {
	Url = websocketUrl
})

local FPS: number = 15
local NUM_CARS = 5

-- === STATE MANAGEMENT ===
local AllCars: { [number]: Model } = {}
local CarCurrentConnections: { [number]: { RBXScriptConnection }? } = {}
local CarsSpawned: number = 0
local CarData = {}

-- Store latest actions here to access them without blocking the game thread
local latestActions = {} 
for i = 1, NUM_CARS do 
	latestActions[i] = {0, 0, 0} -- Defaults: [Throttle, Steering, Nitro]
end

-- === ASYNC NETWORK LISTENER ===
-- This runs in the background. When Python replies, we update 'latestActions'.
websocketClient.MessageReceived:Connect(function(msg)
	local success, result = pcall(HttpService.JSONDecode, HttpService, msg)
	if success and result and result.actions then
		-- Python sends a batch of actions for all cars
		for i, action in ipairs(result.actions) do
			latestActions[i] = action 
		end
	end
end)

-- === CAR SPAWNING LOGIC ===
local SpawnCar: (number) -> ()
SpawnCar = function (CarNumber: number): ()
	local CurrentCar: Model? = AllCars[CarNumber]
	local CurrentConnections: { RBXScriptConnection }? = CarCurrentConnections[CarNumber]

	-- Cleanup previous car
	if CurrentCar then
		CurrentCar:Destroy()
		CurrentCar = nil
	end
	if CurrentConnections ~= nil then 
		for _, conn: RBXScriptConnection in ipairs(CurrentConnections) do conn:Disconnect() end 
	end

	CurrentConnections = {}
	CarCurrentConnections[CarNumber] = CurrentConnections
	CarData[CarNumber] = nil

	-- Spawn new car
	local Car: Model = CarsController.SpawnCars({ 'Car' })[1]
	for _, v: Instance in pairs(Car:GetDescendants()) do
		if v:IsA('BasePart') then
			v.CollisionGroup = 'CAR'
		end
	end

	CurrentCar = Car
	AllCars[CarNumber] = Car

	-- Initial State
	local lastRc: {number}? = CarsController.GetRaycasts(Car)
	if lastRc then
		table.insert(lastRc, 0) -- Nitro status
		table.insert(lastRc, 0) -- Velocity status
	end

	local carInputs = Car:WaitForChild('Inputs')
	local carEngine = Car:WaitForChild('Engine')
	local carController = require(Car:WaitForChild("Scripts"):WaitForChild("Controller"))

	local cc = {
		carInputs = carInputs,
		carEngine = carEngine,
		lastRc = lastRc,
		carController = carController,
		terminationPending = false,
		lap = 0,
		lastCheckpoint = 1,
		startTime = os.time(),

		-- Logic to detect being stuck
		lastMoveTime = os.time(), 

		lastReportedProgress = 0,
		lastSteering = 0 -- Used for smoothness calculation
	}
	CarData[CarNumber] = cc

	-- Wall Collision Detection
	local WallTouched = function (hit: BasePart)
		if hit:IsDescendantOf(Car) then 
			cc.terminationPending = true 
		end
	end

	for _, Wall: Instance in ipairs(TrackWalls:GetChildren()) do
		if Wall:IsA('BasePart') then
			table.insert(CurrentConnections, Wall.Touched:Connect(WallTouched))
		end
	end
end

local AddCar = function()
	CarsSpawned += 1
	local CarNumber: number = CarsSpawned
	SpawnCar(CarNumber)
end

-- Initialize Cars
for _: number = 1, NUM_CARS do AddCar() end

-- === MAIN LOOP ===
local runningLast: boolean = false
local totalDt: number = 0

RunService.Heartbeat:Connect(function (dt: number): ()
	totalDt += dt
	if totalDt < 1/FPS then return end

	if runningLast then
		warn('Skipping frame - previous frame still processing')
		return
	end

	dt = totalDt
	totalDt = 0
	runningLast = true

	local dataToSend = {}

	-- 1. GATHER DATA & CALCULATE REWARDS
	for CarNumber: number, Car: Model in ipairs(AllCars) do
		local cc = CarData[CarNumber]
		if cc == nil then continue end

		-- A. Track Progress & Checkpoints
		local trackProgress: number, checkpoint: number = CheckpointsController.GetTrackProgressState(Car:GetPivot().Position)

		-- Handle Checkpoint Logic
		if checkpoint == cc.lastCheckpoint + 1 or (cc.lastCheckpoint == #CheckpointsController.Checkpoints and checkpoint == 1) then
			cc.lastCheckpoint = checkpoint
			if checkpoint == 1 then
				cc.lap += 1
				cc.lastCheckpoint = 1
			end
		end

		local totalProgress: number = cc.lap + trackProgress

		-- [Observations & Physics State]
		local velocityStatus: number = cc.carEngine:GetAttribute('_speed') or 0
		local maxVelocity: number = cc.carEngine:GetAttribute('forwardMaxSpeed') or 100
		local nitroStatus: number = cc.carEngine:GetAttribute('_nitro') or 0
		local speedRatio = math.clamp(velocityStatus / maxVelocity, -1, 1)

		-- [Stuck/Crash Detection Logic]
		if math.abs(velocityStatus) > 5 then
			cc.lastMoveTime = os.time()
		end
		if os.time() - cc.lastMoveTime > 3 then
			cc.terminationPending = true 
		end
		if os.time() - cc.startTime > 60 and cc.lap == 0 then
			cc.terminationPending = true 
		end

		local rcResults: {number}? = CarsController.GetRaycasts(Car) -- Raycasts are normalized 0-1

		-- Get Inputs (Actions)
		local curThrottle: number = cc.carInputs:GetAttribute('throttleInput') or 0
		local curSteering: number = cc.carInputs:GetAttribute('steeringInput') or 0
		local curNitro: number = cc.carInputs:GetAttribute('nitroInput') and 1 or 0 

		-- ==========================================================
		-- REWARD CALCULATION (Optimized for SAC)
		-- ==========================================================
		local currentReward = 0

		cc.lastReportedProgress = totalProgress

		-- 1. VELOCITY REWARD (Primary Driver)
		-- Reward high speed in the forward direction.
		-- We multiply by direction to punish high-speed reversing.
		
		currentReward += velocityStatus/maxVelocity * 5 * math.exp(velocityStatus/maxVelocity)


		-- 3. STEERING SMOOTHNESS PENALTY (Critical for SAC)
		-- Penalize rapid changes in steering (jitter)
		local lastSteering = cc.lastSteering or 0
		local steeringDelta = math.abs(curSteering - lastSteering)
		currentReward -= steeringDelta * 0.5 
		cc.lastSteering = curSteering -- Update cache

		-- 4. WALL "CUSHION" PENALTY (Exponential Repulsion)
		-- Instead of a cliff, create a "force field" near walls.
		-- Rays are 0 to 1. If closer than 0.3 (30%), penalty increases.
		if rcResults then
			local minRayDist = 1
			for i = 1, #rcResults - 2 do -- Ignore last 2 (velocity/nitro slots)
				local dist = rcResults[i]
				if dist < minRayDist then minRayDist = dist end
			end

			if minRayDist < 0.3 then
				currentReward -= (0.3 - minRayDist) * 1.0 
			end
		end

		-- 5. TERMINATION PENALTY
		-- High enough to discourage crashing, but low enough to allow risk-taking
		local finalReward = if not cc.terminationPending then currentReward else -20.0

		-- Construct Payload
		if rcResults then
			-- Clean observation copy
			local obs = table.clone(rcResults) 
			table.insert(obs, nitroStatus)
			table.insert(obs, speedRatio)

			table.insert(dataToSend, {
				last_observation = cc.lastRc,
				current_observation = obs,
				last_action = {curThrottle, curSteering, curNitro},
				reward = finalReward,
				terminated = cc.terminationPending
			})

			cc.lastRc = obs 
		end
	end

	-- 2. SEND & APPLY (NON-BLOCKING)

	-- Send current state to Python (Fire & Forget)
	local success, jsonPayload = pcall(HttpService.JSONEncode, HttpService, dataToSend)
	if success then
		websocketClient:Send(jsonPayload)
	end

	-- Apply the latest available actions immediately
	for CarNumber: number, _: Model in ipairs(AllCars) do
		local cc = CarData[CarNumber]
		if cc == nil then continue end

		if cc.terminationPending then
			task.spawn(SpawnCar, CarNumber)
			continue
		end

		local aiSteering: boolean = true 
		if aiSteering then
			-- Read from the cached 'latestActions' table
			local action = latestActions[CarNumber]
			if action then
				cc.carInputs:SetAttribute('throttleInput', action[1])
				cc.carInputs:SetAttribute('steeringInput', action[2])
				cc.carInputs:SetAttribute('nitroInput', action[3] > 0.5)
			end
		end

		cc.lastRc = dataToSend[CarNumber].current_observation
		cc.carController:update(dt)
	end

	runningLast = false
end)

return {}