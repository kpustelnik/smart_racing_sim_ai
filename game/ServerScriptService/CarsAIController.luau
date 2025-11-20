local HttpService = game:GetService("HttpService")

local CheckpointsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CheckpointsController'))
local CarsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CarsController'))

local Car: Model = CarsController.SpawnCars({ 'Car' })[1]

local websocketUrl: string = 'ws://localhost:8000/ws'
local websocketClient: WebStreamClient = HttpService:CreateWebStreamClient(Enum.WebStreamClientType.WebSocket, {
  Url = websocketUrl
})

local lastRc: {number}? = CarsController.GetRaycasts(Car)
table.insert(lastRc, 0) -- Nitro status
local carInputs = Car:WaitForChild('Inputs')
local carEngine = Car:WaitForChild('Engine')
local runningLast: boolean = false

local CarController = require(Car:WaitForChild("Scripts"):WaitForChild("Controller"))
game:GetService('RunService').Heartbeat:Connect(function (dt: number): ()
  if runningLast then
    warn('Executed while running last')
    return
  end
  runningLast = true
  local trackProgress: number = CheckpointsController.GetTrackProgressState(Car:GetPivot().Position)
  local rcResults: {number}? = CarsController.GetRaycasts(Car)

  local nitroStatus: number = carEngine:GetAttribute('_nitro') :: any
  local curThrottle: number = carInputs:GetAttribute('throttleInput') :: any
  local curSteering: number = carInputs:GetAttribute('steeringInput') :: any
  local curNitro: number = carInputs:GetAttribute('nitroInput') :: any

  table.insert(rcResults, nitroStatus)

  local aiTraining: boolean = carInputs:GetAttribute('aiTraining') == true
  local aiSteering: boolean = carInputs:GetAttribute('aiSteering') == true
  if aiTraining or aiSteering then
    websocketClient:Send(HttpService:JSONEncode({
      last_observation = lastRc,
      current_observation = rcResults,
      last_action = {curThrottle, curSteering, curNitro},
      reward = trackProgress, -- TODO: Add laps
      terminated = 0
    }))
  end

  if aiSteering then
    local result = HttpService:JSONDecode(websocketClient.MessageReceived:Wait())
    carInputs:SetAttribute('throttleInput', result.action[1])
    carInputs:SetAttribute('steeringInput', result.action[2])
    carInputs:SetAttribute('nitroInput', result.action[3] > 0.5)
  end
  CarController:update(dt)
  lastRc = rcResults
  runningLast = false
end)

return {}