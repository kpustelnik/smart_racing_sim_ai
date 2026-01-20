local Utilities = script.Parent.Parent:WaitForChild('Utilities')
local RayClosestPoint = require(Utilities:WaitForChild('RayClosestPoint'))

local CheckpointsController = {}
CheckpointsController.Checkpoints = {} :: { BasePart }

CheckpointsController.LoadCheckpoints = function (CheckpointsParent: Instance): ()
  local newCheckpoints = {}
  for _, Instance in ipairs(CheckpointsParent:GetDescendants()) do
    if Instance:IsA('BasePart') then
      table.insert(newCheckpoints, Instance)
    end
  end
  table.sort(newCheckpoints, function (a, b) return tonumber(a.Name) < tonumber(b.Name) end)
  CheckpointsController.Checkpoints = newCheckpoints
end

CheckpointsController.GetSizeTo = function (CheckpointId: number): number
  local Size: number = 0
  if #CheckpointsController.Checkpoints == 0 then return Size end

  CheckpointId = math.min(CheckpointId, #CheckpointsController.Checkpoints + 1)
  
  local Prev: BasePart = CheckpointsController.Checkpoints[1]
  for i = 1, CheckpointId do
    local actualIndex = ((i - 1) % #CheckpointsController.Checkpoints) + 1 -- We should also count from the last checkpoint to the first checkpoint (start)
    local Next: BasePart = CheckpointsController.Checkpoints[actualIndex]
    Size += (Next.Position - Prev.Position).Magnitude
    Prev = Next
  end
  return Size
end

CheckpointsController.GetTrackSize = function (): number
  return CheckpointsController.GetSizeTo(math.huge)
end

local Part = Instance.new("Part")
Part.BrickColor = BrickColor.new("Dark green")
Part.Anchored = true
Part.Size = Vector3.new(2,2,2)
Part.CanCollide = false
Part.CanTouch = false
Part.CanQuery = false
Part.Parent = game.Workspace

CheckpointsController.GetTrackDistanceState = function (Position: Vector3, Orientation: Vector3?): (number, number, number?)
  if #CheckpointsController.Checkpoints == 0 then return 0, 0 end

  local minDistanceToRay: number = math.huge
  local minDistanceCheckpoint: number = 0
  local minDistanceProgressDistance: number = math.huge
  local Prev: BasePart = CheckpointsController.Checkpoints[1]
  local TrackOrientation: number? = nil
  for i = 1, #CheckpointsController.Checkpoints + 1 do
    local actualIndex: number = ((i - 1) % #CheckpointsController.Checkpoints) + 1 -- We should also count from the last checkpoint to the first checkpoint (start)
    local Next: BasePart = CheckpointsController.Checkpoints[actualIndex]
    local CP = RayClosestPoint(Prev.Position, Next.Position, Position)
    local Distance = (CP - Position).Magnitude
    if Distance < minDistanceToRay then
      minDistanceToRay = Distance
      minDistanceCheckpoint = i - 1
      minDistanceProgressDistance = (Prev.Position - CP).Magnitude
      Part.CFrame = CFrame.new(CP)

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
  return math.clamp(DistanceState / CheckpointsController.GetTrackSize(), 0, 1), LastCheckpoint, TrackOrientation
end

-- Script (TODO: Move?)
local Workspace = game:GetService('Workspace')
local Checkpoints = Workspace:WaitForChild('Checkpoints')
CheckpointsController.LoadCheckpoints(Checkpoints)

return CheckpointsController