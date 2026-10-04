--!strict
local HttpService = game:GetService("HttpService")
local CmdHandler = require(script:WaitForChild('CmdHandler'))

type BackendMessage = CmdHandler.BackendMessage

local TimeoutSeconds: number = 20

return function (wsUrl: string, modelId: string, carModel: string, enableEnvCollisions: boolean?, terminateAll: boolean?, wallKill: boolean?, noProgressTimeout: number?): () -- Setup the AI controller
  -- WebSocket setup
  local websocketClient: WebStreamClient = HttpService:CreateWebStreamClient(Enum.WebStreamClientType.WebSocket, { Url = wsUrl })
  local CurrentCmdHandler: CmdHandler.CmdHandler = CmdHandler.New(websocketClient, carModel)

  if enableEnvCollisions ~= nil then CurrentCmdHandler.EnableEnvCollisions = enableEnvCollisions end
  if terminateAll ~= nil then CurrentCmdHandler.TerminateAll = terminateAll end
  if wallKill ~= nil then CurrentCmdHandler.WallKill = wallKill end
  if noProgressTimeout ~= nil then CurrentCmdHandler.NoProgressTimeout = noProgressTimeout end

  local lastMsgTime: number = tick()

  local connections: { RBXScriptConnection } = {}

  local function teardown(): ()
    pcall(function(): () websocketClient:Close() end) -- Try to close gracefully
    pcall(function(): () CurrentCmdHandler:Cleanup() end)
    for _, connection: RBXScriptConnection in ipairs(connections) do connection:Disconnect() end
    table.clear(connections)
  end

  local function handleMessage(msg: string): ()
    lastMsgTime = tick()
    local jsonSuccess: boolean, jsonResult: string | BackendMessage = pcall(function (): BackendMessage return HttpService:JSONDecode(msg) end)
    if not jsonSuccess or typeof(jsonResult) == 'string' then
      warn("Failed to decode message:", jsonResult)
      return
    end

    local decodedMsg: BackendMessage = jsonResult
    CurrentCmdHandler:Handle(decodedMsg)
  end

  local ok: boolean, err: string? = xpcall(function(): string?
    table.insert(connections, websocketClient.Opened:Connect(function (): () print(`WebSocket connected for model {modelId}`) end))
    table.insert(connections, websocketClient.Error:Connect(function(...): () warn("Websocket error encountered:", ...) end))
    table.insert(connections, websocketClient.MessageReceived:Connect(handleMessage))

    CurrentCmdHandler:Setup() -- Start simulating
    while websocketClient.ConnectionState ~= Enum.WebStreamClientState.Closed and (tick() - lastMsgTime) <= TimeoutSeconds do
      task.wait(1)
    end
    return nil
  end, debug.traceback)

  teardown()
  warn(`WebSocket for model {modelId} closed.`)

  if not ok then error(err or 'CarsAIController session failed', 0) end
end
