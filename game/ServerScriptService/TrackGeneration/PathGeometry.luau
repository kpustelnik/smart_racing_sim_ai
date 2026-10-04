--!strict

local PathGeometry = {}

-- Signed angle between two vectors
function PathGeometry.angleBetween(a: Vector3, b: Vector3): number
  return math.atan2(a.X * b.Z - a.Z * b.X, a.X * b.X + a.Z * b.Z)
end

-- Orientation of the ordered triplet (p, q, r)
local function orientation2d(p: Vector3, q: Vector3, r: Vector3): number
  local value: number = (q.Z - p.Z) * (r.X - q.X) - (q.X - p.X) * (r.Z - q.Z)
  if math.abs(value) < 0.0001 then
    return 0
  else
    return if value > 0 then 1 else 2
  end
end

-- Checks if the segments p1 -> q1 and p2 -> q2 intersect
function PathGeometry.intersects(p1: Vector3, q1: Vector3, p2: Vector3, q2: Vector3): boolean
  local o1: number = orientation2d(p1, q1, p2)
  local o2: number = orientation2d(p1, q1, q2)
  local o3: number = orientation2d(p2, q2, p1)
  local o4: number = orientation2d(p2, q2, q1)

  return o1 ~= o2 and o3 ~= o4
end

-- Returns minimal distance from point to the segment s -> e
function PathGeometry.distanceToSegment(s: Vector3, e: Vector3, point: Vector3): number
  local ray: Ray = Ray.new(s, e - s).Unit
  local cp: Vector3 = ray:ClosestPoint(point)
  local mag: number = (s - e).Magnitude
  if (cp - s).Magnitude > mag then
    return (e - point).Magnitude
  elseif (cp - e).Magnitude > mag then
    return (s - point).Magnitude
  else
    return (cp - point).Magnitude
  end
end

return PathGeometry
