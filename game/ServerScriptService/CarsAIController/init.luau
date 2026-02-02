--!strict
local HttpService = game:GetService("HttpService")
local CmdHandler = require(script:WaitForChild('CmdHandler'))

type BackendMessage = CmdHandler.BackendMessage

local TimeoutSeconds: number = 20

return function (wsUrl: string, modelId: string, carModel: string, enableEnvCollisions: boolean?): () -- Setup the AI controller
  -- WebSocket setup
  local websocketClient: WebStreamClient = HttpService:CreateWebStreamClient(Enum.WebStreamClientType.WebSocket, { Url = wsUrl })
  local CurrentCmdHandler: CmdHandler.CmdHandler = CmdHandler.New(websocketClient, carModel)

  if enableEnvCollisions ~= nil then
    CurrentCmdHandler.EnableEnvCollisions = enableEnvCollisions
  end
  
  local startTime: number = tick()
  local lastMsgTime: number = startTime
  websocketClient.Opened:Connect(function (): () print(`WebSocket connected for model {modelId}`) end)
  websocketClient.Error:Connect(function(...): () warn("Websocket error encountered:", ...) end)
  local msgReceivedConn: RBXScriptConnection?
  local handleFunc = function (msg: string): ()
    lastMsgTime = tick()
    local jsonSuccess: boolean, jsonResult: string | BackendMessage = pcall(function (): BackendMessage return HttpService:JSONDecode(msg) end)
    if not jsonSuccess or typeof(jsonResult) == 'string' then
      warn("Failed to decode message:", jsonResult)
      return
    end

    local decodedMsg: BackendMessage = jsonResult
    CurrentCmdHandler:Handle(decodedMsg)
  end
  msgReceivedConn = websocketClient.MessageReceived:Connect(handleFunc)

  CurrentCmdHandler:Setup() -- Start simulating
  while websocketClient.ConnectionState ~= Enum.WebStreamClientState.Closed and (tick() - lastMsgTime) <= TimeoutSeconds do
    task.wait(1)
  end
  pcall(function(): () websocketClient:Close() end) -- Try to close gracefully
  warn(`WebSocket for model {modelId} closed.`)
  CurrentCmdHandler:Cleanup()
end