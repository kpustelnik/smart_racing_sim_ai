--!strict

local Types = require(script:WaitForChild("types"))
type Point = Types.Point

local Finalizers = script:WaitForChild("Finalizers")
local Methods = script:WaitForChild("Methods")

local Workspace = game:GetService("Workspace")
local pathFolder = Instance.new("Folder")
pathFolder.Name = 'Path'
pathFolder.Parent = Workspace

return function (Generator: string, Finalizer: string): boolean
  local genSuccess: boolean, finalPath: { Point } = pcall(function(): { Point }
    local GeneratorModule = require(Methods:WaitForChild(Generator))
    return GeneratorModule(pathFolder)
  end)
  if not genSuccess then
    warn("Track generation failed: "..tostring(finalPath))
    return false
  end

  local finSuccess: boolean = pcall(function()
    local FinalizerModule = require(Finalizers:WaitForChild(Finalizer))
    FinalizerModule(finalPath, pathFolder)
  end)
  return finSuccess
end