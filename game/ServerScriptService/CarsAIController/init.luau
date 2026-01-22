--!strict
local HttpService = game:GetService("HttpService")
local CmdHandler = require(script:WaitForChild('CmdHandler'))

type BackendMessage = CmdHandler.BackendMessage

local TimeoutSeconds: number = 20

return function (wsUrl: string, modelId: string, carModel: string): () -- Setup the AI controller
  -- WebSocket setup
  local websocketClient: WebStreamClient = HttpService:CreateWebStreamClient(Enum.WebStreamClientType.WebSocket, { Url = wsUrl })
  local CurrentCmdHandler: CmdHandler.CmdHandler = CmdHandler.New(websocketClient)
  
  local startTime: number = tick()
  local lastMsgTime: number = startTime
  websocketClient.Opened:Connect(function (): () print(`WebSocket connected for model {modelId}`) end)
  websocketClient.Error:Connect(function(...): () warn("Websocket error encountered:", ...) end)
  websocketClient.MessageReceived:Connect(function (msg: string): ()
    lastMsgTime = tick()
    local jsonSuccess: boolean, jsonResult: string | BackendMessage = pcall(function (): BackendMessage return HttpService:JSONDecode(msg) end)
    if not jsonSuccess or typeof(jsonResult) == 'string' then
      warn("Failed to decode message:", jsonResult)
      return
    end

    local decodedMsg: BackendMessage = jsonResult
    CurrentCmdHandler:Handle(decodedMsg)
  end)

  while websocketClient.ConnectionState ~= Enum.WebStreamClientState.Closed and (lastMsgTime - startTime) <= TimeoutSeconds do task.wait(1) end
  pcall(function(): () websocketClient:Close() end) -- Try to close gracefully
  warn(`WebSocket for model {modelId} closed.`)
  CurrentCmdHandler:Cleanup()
end

--[[


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
local UnresponsiveTicks: number = 0
local CloseEnv
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
  local NoneActed = true
  local AnyCar = false
  for CarNumber: number, Car: Model in ipairs(AllCars) do
    local cc = CarData[CarNumber]
    if cc == nil then continue end
    AnyCar = true
    
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
    --[[local normalizedSpeed: number = math.clamp(velocityStatus / maxVelocity, 0, 1)
    local speedBonus: number = (math.exp(normalizedSpeed * 2) - 1) * 0.5 -- max ~3.2 at full speed
    progressToReport += speedBonus]]

    if cc.terminationPending then
      print(`Car {CarNumber} terminated. Total Progress: {totalProgress}`)
      task.spawn(SpawnCar, CarNumber, Car:GetAttribute('ENV') :: any) -- Respawn od razu
    end

    --[[
    # WAŻNE: W SB3 jeśli terminated=True, to 'observations' traktowane jest jako 
                # PIERWSZA KLATKA NOWEGO EPIZODU.
                # Informację o stanie końcowym (uderzeniu w ścianę) SB3 wyciąga z 'infos'.
                
                if terminations[agent]:
                    # Opcjonalnie: Jeśli masz dostęp do stanu w momencie uderzenia, wrzuć go do info
                    # SB3 używa tego do poprawnego obliczenia lossa (TD error)
                    infos[agent]["terminal_observation"] = observations[agent]
                    ]]

    if cc.Acted then
      NoneActed = false
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

  if NoneActed then
    warn('Sent observations before any action was taken')
    UnresponsiveTicks += 1
  else
    UnresponsiveTicks = 0
  end
  if UnresponsiveTicks > 250 and AnyCar then CloseEnv() end

  -- local aiTraining: boolean = carInputs:GetAttribute('aiTraining') == true
  --local aiTraining: boolean = true -- TODO: Implement the value itself back as it needs to be handled globally
  websocketClient:Send(HttpService:JSONEncode(dataToSend))

  runningLast = false
end)
]]