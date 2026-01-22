--!strict

local ServerScriptService = game:GetService("ServerScriptService")
local Utilities = ServerScriptService:WaitForChild("Utilities")
local aligner = require(Utilities:WaitForChild('aligner'))

local Workspace = game:GetService("Workspace")
local Terrain = Workspace:WaitForChild("Terrain")

local Types = require(script.Parent.Parent:WaitForChild("types"))

local shouldAlign = true

local function catmullRom(p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: number)
    local t2 = t * t
    local t3 = t2 * t
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
        (-p0 + 3*p1 - 3*p2 + p3) * t3
    )
end

return function (path: { Types.Point }, pathFolder: Folder): ()
  local prev = nil
  for i, curPoint: Types.Point in ipairs(path) do
    local nextPoint: Types.Point? = if i ~= #path then path[i + 1] else nil
    if curPoint.Next == nextPoint then
      prev = nil
      continue
    end
    if nextPoint ~= nil then
      local Mag: number = (curPoint.Position - nextPoint.Position).Magnitude
      for s = 0, Mag, 1 do
        local t: number = s / Mag
        local t1: number = math.min((s + 1) / Mag, 1)
        local Part: BasePart = Instance.new("Part")
        Part.Anchored = true
        Part.Color = Color3.new(0.2, 0.2, 0.2)
        local pos: Vector3 = catmullRom(
          if curPoint.Direction ~= nil then curPoint.Position - 30 * curPoint.Direction elseif i - 1 >= 1 then path[i-1].Position else curPoint.Position, 
          curPoint.Position, 
          nextPoint.Position, 
          if nextPoint.Direction ~= nil then nextPoint.Position + 30 * nextPoint.Direction elseif i + 2 <= #path then path[i+2].Position else nextPoint.Position, 
          t
        )
        local nextPos: Vector3 = catmullRom(
          if curPoint.Direction ~= nil then curPoint.Position - 30 * curPoint.Direction elseif i - 1 >= 1 then path[i-1].Position else curPoint.Position, 
          curPoint.Position, 
          nextPoint.Position, 
          if nextPoint.Direction ~= nil then nextPoint.Position + 30 * nextPoint.Direction elseif i + 2 <= #path then path[i+2].Position else nextPoint.Position, 
          t1
        )
        Part.CFrame = if s == 0 and curPoint.Direction ~= nil then CFrame.lookAt(curPoint.Position, curPoint.Position + curPoint.Direction, Vector3.new(0, 1, 0)) elseif s > Mag - 1 and nextPoint.Direction ~= nil then CFrame.lookAt(nextPoint.Position, nextPoint.Position - nextPoint.Direction, Vector3.new(0, 1, 0)) else CFrame.lookAt(pos, nextPos, Vector3.new(0, 1, 0))
        Part.Size = Vector3.new(60, 1, 1)
        
        if prev ~= nil and shouldAlign then
          aligner({
            Object = prev,
            Normal = Enum.NormalId.Front,
            IsWedge = false
          }, {
            Object = Part,
            Normal = Enum.NormalId.Back,
            IsWedge = false 
          }, { Mode = "Outer Touch" })
        end

        -- Clear the terrain underneath the road
        Terrain:FillBlock(Part.CFrame * CFrame.new(0, -15, 0), Part.Size + Vector3.new(7, 27, 3), Enum.Material.Asphalt)
        Terrain:FillBlock(Part.CFrame * CFrame.new(0, 15, 0), Part.Size + Vector3.new(2, 33, 3), Enum.Material.Air)
        Part.Parent = pathFolder
        prev = Part
      end
    end
  end
end