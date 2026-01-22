--!strict
local modelId: string = 'telsa_nospecialf'
local carModel: string = 'Telsa'
local wsUrl: string = `ws://localhost:8000/ws/{modelId}`

local TrackGeneration = require(script.Parent:WaitForChild('TrackGeneration'))

local CarsAIController = require(script.Parent:WaitForChild("CarsAIController"))

local createWS = function (...): (boolean, any) return pcall(CarsAIController, ...) end
-- Spawn the base model
task.spawn(function(): ()
  while true do
    local success: boolean = createWS(wsUrl, modelId, carModel)
    if success then
      print("Successfully ran WS.")
    end
    task.wait(5)
  end
end)
-- Another model may be spawned concurrently here...