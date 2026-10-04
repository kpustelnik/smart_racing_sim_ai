--!strict
local modelId: string = 'car_colls_new'
local carModel: string = 'Car'
local wsUrl: string = `ws://localhost:8000/ws/{modelId}`
local wallKill: boolean = true
local enableCollisions: boolean = true
local terminateAll: boolean = false
local noProgressTimeout: number = 5 -- Seconds without new best progress before an agent is truncated

local fixedSeed: number? = nil
local seedModule: Instance? = script.Parent:FindFirstChild('seed')
if seedModule ~= nil and seedModule:IsA('ModuleScript') then
    fixedSeed = require(seedModule) :: number?
end

local GenerationMethod: string = 'Main'
local Finalizer: string = 'MainTile'

local seed: number = if fixedSeed == nil then os.time() else fixedSeed
print("Current seed:", seed)
game:GetService("Workspace"):SetAttribute("SEED", seed)

task.wait(5) -- Wait for other scripts to load
local TrackGeneration: (string, string) -> boolean = require(script.Parent:WaitForChild('TrackGeneration'))
local trackSuccess: boolean = TrackGeneration(GenerationMethod, Finalizer)
if not trackSuccess then error("Failed to generate the track") end

local CarsAIController = require(script.Parent:WaitForChild("CarsAIController"))

local createWS = function (...): (boolean, any) return pcall(CarsAIController, ...) end
-- Spawn the base model
task.spawn(function(): ()
  while true do
    local success: boolean = createWS(wsUrl, modelId, carModel, enableCollisions, terminateAll, wallKill, noProgressTimeout)
    if success then
      print("Successfully ran websocket.")
    end
    task.wait(5)
  end
end)
-- Another model may be spawned concurrently here...