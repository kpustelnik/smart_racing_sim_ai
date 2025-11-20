local HttpService = game:GetService("HttpService")

local CheckpointsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CheckpointsController'))
local CarsController = require(script.Parent:WaitForChild('Track'):WaitForChild('CarsController'))
-- local CarController = require(script.Parent:WaitForChild('Car'):WaitForChild("Controller"))

local Car: Model = CarsController.SpawnCars({ 'Car' })[1]

local websocketUrl: string = 'ws://localhost:8000/ws'
local websocketClient: WebStreamClient = HttpService:CreateWebStreamClient(Enum.WebStreamClientType.WebSocket, {
  Url = websocketUrl
})

local curThrottle: number = 1
local curSteering: number = 0
local curNitro: number = 0
local lastRc = nil
local runningLast: boolean = false

local skipSending: boolean = true
game:GetService('RunService').Heartbeat:Connect(function (dt: number): ()
  if runningLast then
    warn('Executed while running last')
    return
  end
  runningLast = true
  local trackProgress: number = CheckpointsController.GetTrackProgressState(Car:GetPivot().Position)
  local rcResults: {number}? = CarsController.GetRaycasts(Car)

  if not skipSending then
    websocketClient:Send(HttpService:JSONEncode({
      last_observation = lastRc,
      current_observation = rcResults,
      last_action = {curThrottle, curSteering, curNitro},
      reward = trackProgress, -- TODO: Add laps
      terminated = 0
    }))

    local result = HttpService:JSONDecode(websocketClient.MessageReceived:Wait())
    print(result)
    curThrottle = result.action[1]
    curSteering = result.action[2]
  end
  skipSending = false
  --[[
  CarController:update(dt, {
    hand_brake = false,
    throttle = curThrottle,
    steering = curSteering
  })
  ]]
  lastRc = rcResults
  runningLast = false
end)

return {}