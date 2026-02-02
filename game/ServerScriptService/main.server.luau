--!strict
local modelId: string = 'car_new'
local carModel: string = 'Car'
local wsUrl: string = `ws://localhost:8000/ws/{modelId}`
local enableCollisions: boolean = true

local TrackGeneration = require(script.Parent:WaitForChild('TrackGeneration'))

local CarsAIController = require(script.Parent:WaitForChild("CarsAIController"))

local createWS = function (...): (boolean, any) return pcall(CarsAIController, ...) end
-- Spawn the base model
task.spawn(function(): ()
  while true do
    local success: boolean = createWS(wsUrl, modelId, carModel, enableCollisions)
    if success then
      print("Successfully ran websocket.")
    end
    task.wait(5)
  end
end)
-- Another model may be spawned concurrently here...