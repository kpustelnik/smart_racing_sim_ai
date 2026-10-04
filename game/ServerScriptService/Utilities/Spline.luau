--!strict

local EPSILON: number = 1e-4
local ALPHA: number = 0.5
local SUBSAMPLES: number = 64
local RESAMPLE_RESOLUTION: number = 0.25

export type Segment = {
  P1: Vector3,
  P2: Vector3,
  T1: Vector3,
  T2: Vector3,
}

local Spline = {}

local function knotSpan(a: Vector3, b: Vector3): number
  return math.max((b - a).Magnitude, EPSILON) ^ ALPHA
end

local function flatten(v: Vector3): Vector3
  return Vector3.new(v.X, 0, v.Z)
end

--[[
  Builds the segment p1 -> p2.
  p0 and p3 are the neighbouring knots and may be nil at the ends of a chain.
  d1 and d2 optionally pin the tangent direction at either knot
]]
function Spline.segment(p0: Vector3?, p1: Vector3, p2: Vector3, p3: Vector3?, d1: Vector3?, d2: Vector3?): Segment
  local chordVector: Vector3 = p2 - p1
  local chord: number = chordVector.Magnitude
  local dt1: number = math.max(chord, EPSILON) ^ ALPHA

  local t1: Vector3
  if d1 ~= nil and d1.Magnitude > EPSILON then
    t1 = d1.Unit * chord
  elseif p0 ~= nil then
    local dt0: number = knotSpan(p0, p1)
    t1 = ((p1 - p0) / dt0 - (p2 - p0) / (dt0 + dt1) + chordVector / dt1) * dt1
  else
    t1 = chordVector
  end

  local t2: Vector3
  if d2 ~= nil and d2.Magnitude > EPSILON then
    t2 = d2.Unit * chord
  elseif p3 ~= nil then
    local dt2: number = knotSpan(p2, p3)
    t2 = (chordVector / dt1 - (p3 - p1) / (dt1 + dt2) + (p3 - p2) / dt2) * dt1
  else
    t2 = chordVector
  end

  if t1.Magnitude < EPSILON then t1 = chordVector end
  if t2.Magnitude < EPSILON then t2 = chordVector end

  return { P1 = p1, P2 = p2, T1 = t1, T2 = t2 }
end

function Spline.evaluate(segment: Segment, u: number): Vector3
  -- Catmull-Rom cubic Hermite form
  local u2: number = u * u
  local u3: number = u2 * u
  return segment.P1 * (2 * u3 - 3 * u2 + 1)
    + segment.T1 * (u3 - 2 * u2 + u)
    + segment.P2 * (-2 * u3 + 3 * u2)
    + segment.T2 * (u3 - u2)

end

function Spline.velocity(segment: Segment, u: number): Vector3
  local u2: number = u * u
  return segment.P1 * (6 * u2 - 6 * u)
    + segment.T1 * (3 * u2 - 4 * u + 1)
    + segment.P2 * (-6 * u2 + 6 * u)
    + segment.T2 * (3 * u2 - 2 * u)
end

-- Cumulative lengths over SUBSAMPLES + 1 evenly spaced values of u
local function measure(segment: Segment): ({ number }, number)
  local lengths: { number } = table.create(SUBSAMPLES + 1)
  lengths[1] = 0

  local previous: Vector3 = segment.P1
  local total: number = 0

  for k = 1, SUBSAMPLES do
    local point: Vector3 = Spline.evaluate(segment, k / SUBSAMPLES)
    total += (point - previous).Magnitude
    lengths[k + 1] = total
    previous = point
  end

  return lengths, total
end

local function measureCost(segment: Segment, spacing: number, maxTurn: number, subsamples: number): { number }
  local costs: { number } = table.create(subsamples + 1)
  costs[1] = 0

  local previous: Vector3 = segment.P1
  local previousDirection: Vector3? = nil
  local total: number = 0

  for k = 1, subsamples do
    local point: Vector3 = Spline.evaluate(segment, k / subsamples)
    local step: Vector3 = point - previous
    total += step.Magnitude / spacing

    if maxTurn > 0 then
      local direction: Vector3 = flatten(step)
      if direction.Magnitude > EPSILON then
        direction = direction.Unit
        if previousDirection ~= nil then
          total += math.acos(math.clamp(previousDirection:Dot(direction), -1, 1)) / maxTurn
        end
        previousDirection = direction
      end
    end

    costs[k + 1] = total
    previous = point
  end

  return costs
end

local function parameterAtLength(cumulative: { number }, target: number, cursor: number): (number, number)
  local count: number = #cumulative
  while cursor < count - 1 and cumulative[cursor + 1] < target do
    cursor += 1
  end

  local before: number = cumulative[cursor]
  local after: number = cumulative[cursor + 1]
  local span: number = after - before
  local fraction: number = if span > EPSILON then (target - before) / span else 0

  return (cursor - 1 + math.clamp(fraction, 0, 1)) / (count - 1), cursor
end

function Spline.arcLengths(points: { Vector3 }): { number }
  local travelled: { number } = table.create(#points)
  travelled[1] = 0
  for i = 2, #points do
    travelled[i] = travelled[i - 1] + (points[i] - points[i - 1]).Magnitude
  end
  return travelled
end

function Spline.sampleSegment(segment: Segment, spacing: number, overshoot: number): ({ Vector3 }, number)
  local lengths: { number }, total: number = measure(segment)
  local count: number = math.clamp(math.ceil(total / spacing), 1, 8192)

  local points: { Vector3 } = table.create(count + 1)
  points[1] = segment.P1

  local cursor: number = 1
  for k = 1, count - 1 do
    local u: number
    u, cursor = parameterAtLength(lengths, total * k / count, cursor)
    points[k + 1] = Spline.evaluate(segment, u)
  end
  points[count + 1] = segment.P2

  local knotIndex: number = count + 1

  if overshoot > 0 then
    local speed: number = Spline.velocity(segment, 1).Magnitude
    if speed > EPSILON then
      local step: number = spacing / speed
      for k = 1, math.ceil(overshoot / spacing) do
        table.insert(points, Spline.evaluate(segment, 1 + k * step))
      end
    end
  end

  return points, knotIndex
end

function Spline.resample(segments: { Segment }, spacing: number, maxTurn: number): ({ Vector3 }, { number })
  local points: { Vector3 } = { segments[1].P1 }
  local knotIndices: { number } = { 1 }

  for _, segment: Segment in ipairs(segments) do
    local _, length: number = measure(segment)
    local subsamples: number = math.clamp(math.ceil(length / (spacing * RESAMPLE_RESOLUTION)), SUBSAMPLES, 8192)
    local costs: { number } = measureCost(segment, spacing, maxTurn, subsamples)
    local totalCost: number = costs[#costs]
    local count: number = math.clamp(math.ceil(totalCost), 2, 4096)

    local cursor: number = 1
    for k = 1, count - 1 do
      local u: number
      u, cursor = parameterAtLength(costs, totalCost * k / count, cursor)
      table.insert(points, Spline.evaluate(segment, u))
    end
    table.insert(points, segment.P2)
    table.insert(knotIndices, #points)
  end

  return points, knotIndices
end

return Spline
