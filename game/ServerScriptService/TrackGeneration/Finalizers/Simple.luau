--!strict

local ServerScriptService = game:GetService("ServerScriptService")
local Utilities: Folder = ServerScriptService:WaitForChild("Utilities")
local aligner = require(Utilities:WaitForChild('aligner'))
local Spline = require(Utilities:WaitForChild('Spline'))

local Workspace = game:GetService("Workspace")
local Terrain: Terrain = Workspace.Terrain

local Types = require(script.Parent.Parent:WaitForChild("types"))
local FinalizerCommon = require(script.Parent.Parent:WaitForChild("FinalizerCommon"))

local shouldAlign: boolean = true

local trackWidth: number = 49
local partSpacing: number = 1
local partSize: Vector3 = Vector3.new(trackWidth, 1, 1)

local UP: Vector3 = Vector3.new(0, 1, 0)

return function (path: { Types.Point }, pathFolder: Folder): ()
  local createCheckpoint = FinalizerCommon.checkpointWriter()

  local prev: BasePart? = nil
  for i: number, curPoint: Types.Point in ipairs(path) do
    local nextPoint: Types.Point? = if i ~= #path then path[i + 1] else nil
    if curPoint.Next == nextPoint then
      prev = nil
      continue
    end
    if nextPoint ~= nil then
      local span: FinalizerCommon.Span = FinalizerCommon.span(path, i, FinalizerCommon.levelLook)
      local startTangent: Vector3? = span.StartTangent
      local endTangent: Vector3? = span.EndTangent

      createCheckpoint(span.StartCFrame)
      if endTangent ~= nil then createCheckpoint(span.EndCFrame) end

      local points: { Vector3 } = Spline.sampleSegment(span.Segment, partSpacing, 0)
      local count: number = #points

      for k = 1, count - 1 do
        local here: Vector3 = points[k]
        local ahead: Vector3 = points[k + 1]

        local Part: BasePart = Instance.new("Part")
        Part.Anchored = true
        Part.Color = Color3.new(0.2, 0.2, 0.2)
        Part.Size = partSize

        if k == 1 and startTangent ~= nil then
          Part.CFrame = CFrame.lookAt(here, here + startTangent, UP)
        elseif k == count - 1 and endTangent ~= nil then
          Part.CFrame = CFrame.lookAt(here, here + endTangent, UP)
        else
          Part.CFrame = CFrame.lookAt(here, ahead, UP)
        end

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
        Terrain:FillBlock(Part.CFrame * CFrame.new(0, 15, 0), Part.Size + Vector3.new(9, 33, 5), Enum.Material.Air)
        Part.Parent = pathFolder
        prev = Part
      end
    end
  end
end
