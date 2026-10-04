--!strict

local ServerScriptService = game:GetService("ServerScriptService")
local Workspace = game:GetService("Workspace")

local Spline = require(ServerScriptService:WaitForChild("Utilities"):WaitForChild("Spline"))
local Types = require(script.Parent:WaitForChild("types"))

local EPSILON: number = 1e-4
local UP: Vector3 = Vector3.new(0, 1, 0)
local CHECKPOINTS_FOLDER: string = "Checkpoints"

local FinalizerCommon = {}

export type Look = (from: Vector3, towards: Vector3) -> CFrame

export type Span = {
  Segment: Spline.Segment,
  StartCFrame: CFrame,
  EndCFrame: CFrame,
  StartTangent: Vector3?,
  EndTangent: Vector3?,
}

function FinalizerCommon.levelLook(from: Vector3, towards: Vector3): CFrame
  local direction: Vector3 = Vector3.new(towards.X - from.X, 0, towards.Z - from.Z)
  if direction.Magnitude < EPSILON then direction = Vector3.new(0, 0, 1) end
  return CFrame.lookAt(from, from + direction.Unit, UP)
end

function FinalizerCommon.nodeTangent(direction: Vector3?, sign: number): Vector3?
  if direction == nil then return nil end
  local flat: Vector3 = Vector3.new(direction.X, 0, direction.Z) * sign
  return if flat.Magnitude > EPSILON then flat.Unit else nil
end

function FinalizerCommon.span(path: { Types.Point }, i: number, look: Look): Span
  local curPoint: Types.Point = path[i]
  local nextPoint: Types.Point = path[i + 1]

  local startTangent: Vector3? = FinalizerCommon.nodeTangent(curPoint.Direction, 1)
  local endTangent: Vector3? = FinalizerCommon.nodeTangent(nextPoint.Direction, -1)
  local segment: Spline.Segment = Spline.segment(
    if i - 1 >= 1 then path[i - 1].Position else nil,
    curPoint.Position,
    nextPoint.Position,
    if i + 2 <= #path then path[i + 2].Position else nil,
    startTangent,
    endTangent
  )

  local startCFrame: CFrame = if startTangent ~= nil
    then look(curPoint.Position, curPoint.Position + startTangent)
    else look(segment.P1, segment.P1 + Spline.velocity(segment, 0))
  local endCFrame: CFrame = if endTangent ~= nil
    then look(nextPoint.Position, nextPoint.Position + endTangent)
    else look(segment.P2, segment.P2 + Spline.velocity(segment, 1))

  return {
    Segment = segment,
    StartCFrame = startCFrame,
    EndCFrame = endCFrame,
    StartTangent = startTangent,
    EndTangent = endTangent,
  }
end

function FinalizerCommon.checkpointWriter(): (cframe: CFrame) -> Part
  local existing: Instance? = Workspace:FindFirstChild(CHECKPOINTS_FOLDER)
  local folder: Instance = if existing ~= nil then existing else Instance.new("Folder")
  folder.Name = CHECKPOINTS_FOLDER
  folder.Parent = Workspace

  local checkpointIndex: number = 0
  return function (cframe: CFrame): Part
    local checkpointPart: Part = Instance.new("Part")
    checkpointPart.Name = tostring(checkpointIndex)
    checkpointIndex += 1
    checkpointPart.Color = Color3.new(1, 1, 0)
    checkpointPart.Material = Enum.Material.Neon
    checkpointPart.Size = Vector3.new(20, 30, 2)
    checkpointPart.CFrame = cframe
    checkpointPart.Anchored = true
    checkpointPart.Transparency = 0.5
    checkpointPart.CanCollide = false
    checkpointPart.CanQuery = false
    checkpointPart.CanTouch = false
    checkpointPart.Parent = folder
    return checkpointPart
  end
end

return FinalizerCommon
