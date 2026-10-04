--!strict

--[[
  Bounding box of a set of parts, expressed in the space of a given basis.
  Adapted from stravant's redupe.
]]
local BoundingBox = {}

-- Returns the centre offset and the size, both in the basis's space.
function BoundingBox.compute(basis: CFrame, parts: { BasePart }): (Vector3, Vector3)
  local inverseBasis: CFrame = basis:Inverse()
  local xmin: number, xmax: number = math.huge, -math.huge
  local ymin: number, ymax: number = math.huge, -math.huge
  local zmin: number, zmax: number = math.huge, -math.huge

  for _, part: BasePart in ipairs(parts) do
    local size: Vector3 = part.Size
    local sx: number, sy: number, sz: number = size.X, size.Y, size.Z

    local localCFrame: CFrame = inverseBasis * part.CFrame
    local _, _, _,
      t00: number, t01: number, t02: number,
      t10: number, t11: number, t12: number,
      t20: number, t21: number, t22: number = localCFrame:GetComponents()

    local halfWidth: number = 0.5 * (math.abs(sx * t00) + math.abs(sy * t01) + math.abs(sz * t02))
    local halfHeight: number = 0.5 * (math.abs(sx * t10) + math.abs(sy * t11) + math.abs(sz * t12))
    local halfDepth: number = 0.5 * (math.abs(sx * t20) + math.abs(sy * t21) + math.abs(sz * t22))
    local x: number, y: number, z: number = localCFrame.X, localCFrame.Y, localCFrame.Z

    xmin = math.min(xmin, x - halfWidth)
    xmax = math.max(xmax, x + halfWidth)
    ymin = math.min(ymin, y - halfHeight)
    ymax = math.max(ymax, y + halfHeight)
    zmin = math.min(zmin, z - halfDepth)
    zmax = math.max(zmax, z + halfDepth)
  end

  local offset: Vector3 = Vector3.new(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
  local extent: Vector3 = Vector3.new(xmax - xmin, ymax - ymin, zmax - zmin)
  return offset, extent
end

return BoundingBox
