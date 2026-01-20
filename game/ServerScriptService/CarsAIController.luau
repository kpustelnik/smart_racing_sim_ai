local HttpService = game:GetService("HttpService")
local Workspace = game:GetService('Workspace')
local PhysicsService = game:GetService('PhysicsService')
PhysicsService:RegisterCollisionGroup('CAR')
PhysicsService:CollisionGroupSetCollidable('CAR', 'CAR', false) -- Enable cars collisisons

local TrackWalls = Workspace:WaitForChild('Track'):WaitForChild('Walls')

local CheckpointsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CheckpointsController'))
local CarsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CarsController'))

local modelId: string = 'car'
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
    truncationPending = false,
    lap = 0,
    lastCheckpoint = 1,
    startTime = os.time(),
    lastBestProgress = 0,
    lastBestProgressTime = os.time(),
    lastReportedProgress = 0,
    Acted = false
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

local CarsNames: { [string]: number } = {}
local getCarName = function (CarNumber: number): string?
  for name: string, number in pairs(CarsNames) do
    if number == CarNumber then return name end
  end
  return nil
end

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
  local AnyActed = false
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
      cc.truncationPending = true -- Stuck for too long
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

    -- non-linear speed bonus (exponential scaling)
    local normalizedSpeed: number = math.clamp(velocityStatus / maxVelocity, 0, 1)
    local speedBonus: number = (math.exp(normalizedSpeed * 2) - 1) * 0.5 -- max ~3.2 at full speed
    progressToReport += speedBonus

    if cc.terminationPending then
      --print(`Car {CarNumber} terminated. Total Progress: {totalProgress}`)
    end

    if not cc.Acted then
      AnyActed = true
    end
    cc.Acted = false

    local carName = getCarName(CarNumber) :: any
    if carName ~= nil then
      dataToSend[carName] = {
        -- last_observation = cc.lastRc,
        obs = rcResults,
        -- last_action = {curThrottle, curSteering, curNitro},
        reward = if not cc.terminationPending then progressToReport else -10,
        terminated = cc.terminationPending,
        truncated = cc.truncationPending
      }
      cc.lastRc = rcResults
    end
  end

  if AnyActed then
    warn('Sent observations before any action was taken')
  end

  -- local aiTraining: boolean = carInputs:GetAttribute('aiTraining') == true
  --local aiTraining: boolean = true -- TODO: Implement the value itself back as it needs to be handled globally
  websocketClient:Send(HttpService:JSONEncode(dataToSend))

  runningLast = false
end)

local handles = {}

type SpawnAgentsData = {
  agents: { string }
}
handles['SPAWN_AGENTS'] = function (data: SpawnAgentsData): () 
  local agents: { string } = data.agents
  
  for _, AgentId: string in pairs(agents) do
    CarsSpawned += 1
    local CarNumber: number = CarsSpawned
    SpawnCar(CarNumber)
    CarsNames[AgentId] = CarNumber
  end
end

type ResetData = {
  agents: { string }
}
handles['RESET'] = function (data: ResetData): ()
  local agents: { string } = data.agents
  
  for _, AgentId: string in pairs(agents) do
    local CarNumber: number? = CarsNames[AgentId]
    if CarNumber then
      local cCarData = CarData[CarNumber]
      if cCarData == nil then continue end

      --cCarData.terminationPending = true
      task.spawn(SpawnCar, CarNumber)
    end
  end
end

local CloseEnv = function (): ()
  print("Closing envv")
  for agentId, carNumber in pairs(CarsNames) do
    local CurrentCar: Model? = AllCars[carNumber]
    local CurrentConnections: { RBXScriptConnection }? = CarCurrentConnections[carNumber]
    if CurrentCar then
      CurrentCar:Destroy()
      CurrentCar = nil
    end
    if CurrentConnections ~= nil then for _, conn: RBXScriptConnection in ipairs(CurrentConnections) do conn:Disconnect() end end
    CarData[carNumber] = nil
  end
end
handles['CLOSE'] = CloseEnv

type ActionData = {

}
handles['ACTION'] = function (data: ActionData): ()
  for agentId: string, agentData in pairs(data) do
    local CarNumber: number? = CarsNames[agentId]
    if CarNumber == nil then continue end
    local cc = CarData[CarNumber]
    if cc == nil then continue end

    --[[
    local aiSteering: boolean = cc.carInputs:GetAttribute('aiSteering') == true
    if aiSteering then
    ]]
    cc.carInputs:SetAttribute('throttleInput', agentData[1])
    cc.carInputs:SetAttribute('steeringInput', agentData[2])
    cc.carInputs:SetAttribute('nitroInput', agentData[3] > 0.5)
    cc.Acted = true

    if cc.lastUpdateTime ~= nil and typeof(cc.lastUpdateTime) == "number" then
      local dt: number = tick() - cc.lastUpdateTime
      cc.carController:update(dt)
    end
    cc.lastUpdateTime = tick()
  end
end

type BackendMessage = {
  command: string,
  data: any
}

websocketClient.Closed:Connect(function(): ()
  CloseEnv()
end)

websocketClient.MessageReceived:Connect(function(msg): ()
  local decodedMsg: BackendMessage = HttpService:JSONDecode(msg)
  local handler = handles[decodedMsg.command]
  if handler then
    local success: boolean, result = pcall(handler, decodedMsg.data)
    if not success then
      print("Error handling message:", result)
    end
  end
end)

return {}