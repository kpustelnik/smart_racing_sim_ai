--!strict
local modelId: string = 'car_colls_new'
local carModel: string = 'Car'
local wsUrl: string = `ws://localhost:8000/ws/{modelId}`
local wallKill: boolean = true
local enableCollisions: boolean = true
local terminateAll: boolean = false

local seed: number = 1778399617 -- os.time()
print("Current seed:", seed)
game:GetService("Workspace"):SetAttribute("SEED", seed)

task.wait(5) -- Wait for other scripts to load
local TrackGeneration = require(script.Parent:WaitForChild('TrackGeneration'))
local trackSuccess: boolean = TrackGeneration('Main', 'Main')
if not trackSuccess then error("Failed to generate the track") end

local CarsAIController = require(script.Parent:WaitForChild("CarsAIController"))

local createWS = function (...): (boolean, any) return pcall(CarsAIController, ...) end
-- Spawn the base model
task.spawn(function(): ()
  while true do
    local success: boolean = createWS(wsUrl, modelId, carModel, enableCollisions, terminateAll, wallKill)
    if success then
      print("Successfully ran websocket.")
    end
    task.wait(5)
  end
end)
-- Another model may be spawned concurrently here...