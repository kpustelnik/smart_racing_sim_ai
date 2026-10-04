--!strict

local Types = require(script:WaitForChild("types"))
type Point = Types.Point

local Finalizers = script:WaitForChild("Finalizers")
local Methods = script:WaitForChild("Methods")

local Workspace = game:GetService("Workspace")
local pathFolder = Instance.new("Folder")
pathFolder.Name = 'Path'
pathFolder.Parent = Workspace

local function resolve(container: Instance, name: string, kind: string): ModuleScript?
  local module: Instance? = container:FindFirstChild(name)
  if module == nil then
    warn(`Track generation failed: no {kind} named '{name}' in {container:GetFullName()}`)
    return nil
  end
  if not module:IsA("ModuleScript") then
    warn(`Track generation failed: {kind} '{name}' is a {module.ClassName}, not a ModuleScript`)
    return nil
  end
  return module
end

return function (Generator: string, Finalizer: string): boolean
  local GeneratorModule: ModuleScript? = resolve(Methods, Generator, "generator")
  local FinalizerModule: ModuleScript? = resolve(Finalizers, Finalizer, "finalizer")
  if GeneratorModule == nil or FinalizerModule == nil then return false end

  local genSuccess: boolean, finalPath: any = xpcall(function(): { Point }
    return (require(GeneratorModule) :: any)(pathFolder)
  end, debug.traceback)
  if not genSuccess then
    warn(`Track generation failed in '{Generator}': {finalPath}`)
    return false
  end

  if type(finalPath) ~= "table" or #finalPath < 2 then
    warn(`Track generation failed: '{Generator}' found no path for seed {Workspace:GetAttribute("SEED")}`)
    return false
  end

  local finSuccess: boolean, finError: any = xpcall(function(): ()
    (require(FinalizerModule) :: any)(finalPath, pathFolder)
  end, debug.traceback)
  if not finSuccess then
    warn(`Track finalization failed in '{Finalizer}': {finError}`)
    return false
  end

  return true
end
