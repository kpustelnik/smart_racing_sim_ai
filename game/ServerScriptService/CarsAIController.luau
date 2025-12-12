local HttpService = game:GetService("HttpService")
local Workspace = game:GetService('Workspace')
local PhysicsService = game:GetService('PhysicsService')
PhysicsService:RegisterCollisionGroup('CAR')
PhysicsService:CollisionGroupSetCollidable('CAR', 'CAR', true) -- Enable cars collisisons

local TrackWalls = Workspace:WaitForChild('Track'):WaitForChild('Walls')

local CheckpointsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CheckpointsController'))
local CarsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CarsController'))

local modelId: string = 'multi_car_orient'
local carModel: string = 'Car'
local websocketUrl: string = `ws://localhost:8000/ws/{modelId}`
local websocketClient: WebStreamClient = HttpService:CreateWebStreamClient(Enum.WebStreamClientType.WebSocket, {
  Url = websocketUrl
})

local AllCars: { [number]: Model } = {}
local CarCurrentConnections = {} :: { [number]: { RBXScriptConnection }? }
local CarsSpawned: number = 0
local CarData = {}
local SpawnCar: (number) -> ()
SpawnCar = function (CarNumber: number): ()
  local CurrentCar: Model? = AllCars[CarNumber]
  local CurrentConnections: { RBXScriptConnection }? = CarCurrentConnections[CarNumber]
  if CurrentCar then
    CurrentCar:Destroy()
    CurrentCar = nil
  end
  if CurrentConnections ~= nil then for _, conn: RBXScriptConnection in ipairs(CurrentConnections) do conn:Disconnect() end end
  CurrentConnections = {}
  CarCurrentConnections[CarNumber] = CurrentConnections
  CarData[CarNumber] = nil

  local CarSpawn = {CarsController.GetSpawns()[CarNumber] }
  local Car: Model = CarsController.SpawnCars({ carModel }, CarSpawn)[1]
  for _, v: Instance in pairs(Car:GetDescendants()) do
    if v:IsA('BasePart') then
      v.CollisionGroup = 'CAR'
    end
  end
  CurrentCar = Car
  AllCars[CarNumber] = Car

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
    lastBestProgress = 0,
    lastBestProgressTime = os.time(),
    lastReportedProgress = 0
  }
  CarData[CarNumber] = cc

  -- Walls collision detection
  local WallTouched = function (hit: BasePart)
    if hit:IsDescendantOf(Car) then cc.terminationPending = true end
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
for _: number = 1, 5 do AddCar() end

local runningLast: boolean = false
local FPS: number = 15
local totalDt: number = 0
game:GetService('RunService').Heartbeat:Connect(function (dt: number): ()
  totalDt += dt
  if totalDt < 1/FPS then return end
  if runningLast then
    warn('Executed while running last')
    return
  end
  dt = totalDt
  totalDt = 0

  runningLast = true
  local dataToSend = {}
  for CarNumber: number, Car: Model in ipairs(AllCars) do
    local cc = CarData[CarNumber]
    if cc == nil then continue end
    
    local carPosition = Car:GetPivot().Position
    local carOrientation: Vector3 = Car:GetPivot().LookVector
    local trackProgress: number, checkpoint: number, trackOrientation: number? = CheckpointsController.GetTrackProgressState(carPosition, carOrientation)
    if checkpoint == cc.lastCheckpoint + 1 or (cc.lastCheckpoint == #CheckpointsController.Checkpoints and checkpoint == 1) then
      cc.lastCheckpoint = checkpoint
      if checkpoint == 1 then
        cc.lap += 1
        cc.lastCheckpoint = 1
      end
    end

    local maxLegalProgress: number = CheckpointsController.GetSizeTo(cc.lastCheckpoint + 1) / CheckpointsController.GetTrackSize()
    local currentLegalProgress: number = CheckpointsController.GetSizeTo(cc.lastCheckpoint) / CheckpointsController.GetTrackSize()
    if trackProgress > maxLegalProgress then trackProgress = currentLegalProgress end -- Return to minimal just so the robot does not learn to cheat by reversing
    
    local totalProgress: number = cc.lap + trackProgress
    if totalProgress > cc.lastBestProgress then
      cc.lastBestProgress = totalProgress
      cc.lastBestProgressTime = os.time()
    elseif os.time() - cc.lastBestProgressTime > 4 then
      cc.terminationPending = true -- Stuck for too long
    end
    
    local rcResults: {number}? = CarsController.GetRaycasts(Car)
    local nitroStatus: number = cc.carEngine:GetAttribute('_nitro') :: any
    local velocityStatus: number = cc.carEngine:GetAttribute('_speed') :: any
    local maxVelocity: number = cc.carEngine:GetAttribute('forwardMaxSpeed') :: any
    if rcResults then
      table.insert(rcResults, nitroStatus)
      table.insert(rcResults, math.clamp(velocityStatus / maxVelocity, -1, 1))
      
      local realOrientation: number = if trackOrientation ~= nil then trackOrientation / math.pi else 0
      local isReverse: number = if realOrientation > 0.5 or realOrientation < -0.5 then 1 else 0
      table.insert(rcResults, isReverse)
    end

    local curThrottle: number = cc.carInputs:GetAttribute('throttleInput') :: any
    local curSteering: number = cc.carInputs:GetAttribute('steeringInput') :: any
    local curNitro: number = cc.carInputs:GetAttribute('nitroInput') :: any

    local progressToReport: number = totalProgress - cc.lastReportedProgress
    cc.lastReportedProgress = totalProgress

    progressToReport *= 1000
    if progressToReport < 0 then
      progressToReport *= 20
    end

    if cc.terminationPending then
      print(`Car {CarNumber} terminated. Total Progress: {totalProgress}`)
    end

    table.insert(dataToSend, {
      last_observation = cc.lastRc,
      current_observation = rcResults,
      last_action = {curThrottle, curSteering, curNitro},
      reward = if not cc.terminationPending then progressToReport else -10,
      terminated = cc.terminationPending
    })
  end

  -- local aiTraining: boolean = carInputs:GetAttribute('aiTraining') == true
  local aiTraining: boolean = true -- TODO: Implement the value itself back as itn eeds to be handled globally
  websocketClient:Send(HttpService:JSONEncode(dataToSend))
  local result = HttpService:JSONDecode(websocketClient.MessageReceived:Wait())
  for CarNumber: number, _: Model in ipairs(AllCars) do
    local cc = CarData[CarNumber]
    if cc == nil then continue end
    
    if cc.terminationPending then
      task.spawn(SpawnCar, CarNumber)
      continue
    end

    local aiSteering: boolean = cc.carInputs:GetAttribute('aiSteering') == true
    if aiSteering then
      cc.carInputs:SetAttribute('throttleInput', result.actions[CarNumber][1])
      cc.carInputs:SetAttribute('steeringInput', result.actions[CarNumber][2])
      cc.carInputs:SetAttribute('nitroInput', result.actions[CarNumber][3] > 0.5)
    end

    cc.lastRc = dataToSend[CarNumber].current_observation
    cc.carController:update(dt)
  end

  runningLast = false
end)

return {}