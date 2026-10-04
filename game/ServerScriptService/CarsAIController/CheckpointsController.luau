--!strict
local Utilities = script.Parent.Parent:WaitForChild('Utilities')
local RayClosestPoint = require(Utilities:WaitForChild('RayClosestPoint'))

local ARC_LENGTH_ATTRIBUTE: string = 'ARC_LENGTH'

local CheckpointsController = {}
CheckpointsController.Checkpoints = {} :: { BasePart }

local cumulative: { number } = {}
local totalSize: number = 0

local function spanBetween(from: BasePart, to: BasePart): number
  local arcLength: unknown? = from:GetAttribute(ARC_LENGTH_ATTRIBUTE)
  if typeof(arcLength) == 'number' and arcLength > 0 then return arcLength end
  return (to.Position - from.Position).Magnitude -- Fallback
end

CheckpointsController.LoadCheckpoints = function (CheckpointsParent: Instance): ()
  local newCheckpoints: { BasePart } = {}
  for _, instance: Instance in ipairs(CheckpointsParent:GetDescendants()) do
    if instance:IsA('BasePart') then
      table.insert(newCheckpoints, instance)
    end
  end
  table.sort(newCheckpoints, function (a: BasePart, b: BasePart): boolean
    return (tonumber(a.Name) or 0) < (tonumber(b.Name) or 0)
  end)
  CheckpointsController.Checkpoints = newCheckpoints

  cumulative = table.create(#newCheckpoints + 1)
  cumulative[1] = 0
  local cSize: number = 0
  for i = 1, #newCheckpoints do
    local from: BasePart = newCheckpoints[i]
    local to: BasePart = newCheckpoints[(i % #newCheckpoints) + 1]
    cSize += spanBetween(from, to)
    cumulative[i + 1] = cSize
  end
  totalSize = cSize
end

CheckpointsController.GetSizeTo = function (CheckpointId: number): number
  local count: number = #CheckpointsController.Checkpoints
  if count == 0 then return 0 end

  CheckpointId = math.clamp(CheckpointId, 0, count + 1)

  local index: number = math.floor(CheckpointId) + 1
  return cumulative[math.clamp(index, 1, #cumulative)] or 0
end

CheckpointsController.GetTrackSize = function (): number
  return totalSize
end

CheckpointsController.GetTrackDistanceState = function (Position: Vector3, Orientation: Vector3?): (number, number, number?)
  local count: number = #CheckpointsController.Checkpoints
  if count == 0 then return 0, 0, nil end

  local minDistanceToRay: number = math.huge
  local minDistanceCheckpoint: number = 0
  local minDistanceProgressDistance: number = math.huge
  local Prev: BasePart = CheckpointsController.Checkpoints[1]
  local TrackOrientation: number? = nil
  for i = 2, count + 1 do
    local actualIndex: number = ((i - 1) % count) + 1 -- We should also count from the last checkpoint to the first checkpoint (start)
    local Next: BasePart = CheckpointsController.Checkpoints[actualIndex]
    local CP: Vector3 = RayClosestPoint(Prev.Position, Next.Position, Position)
    local Distance: number = (CP - Position).Magnitude
    if Distance < minDistanceToRay then
      minDistanceToRay = Distance
      minDistanceCheckpoint = i - 1
      minDistanceProgressDistance = (Prev.Position - CP).Magnitude

      if Orientation ~= nil then
        local TrackDirection: Vector3 = (Next.Position - Prev.Position).Unit * Vector3.new(1, 0, 1)
        Orientation = Orientation * Vector3.new(1, 0, 1)
        TrackOrientation = math.atan2(Orientation:Cross(TrackDirection).Y, Orientation:Dot(TrackDirection))
      end
    end
    Prev = Next
  end

  return CheckpointsController.GetSizeTo(minDistanceCheckpoint) + minDistanceProgressDistance, minDistanceCheckpoint, TrackOrientation
end

CheckpointsController.GetTrackProgressState = function (Position: Vector3, Orientation: Vector3?): (number, number, number?)
  local DistanceState: number, LastCheckpoint: number, TrackOrientation: number? = CheckpointsController.GetTrackDistanceState(Position, Orientation)
  local size: number = CheckpointsController.GetTrackSize()
  if size <= 0 then return 0, LastCheckpoint, TrackOrientation end
  return math.clamp(DistanceState / size, 0, 1), LastCheckpoint, TrackOrientation
end

-- Script
local Workspace = game:GetService('Workspace')
local Checkpoints = Workspace:WaitForChild('Checkpoints')
CheckpointsController.LoadCheckpoints(Checkpoints)

return CheckpointsController
