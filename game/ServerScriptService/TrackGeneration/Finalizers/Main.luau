--!strict

local Types = require(script.Parent.Parent:WaitForChild("types"))
local FinalizerCommon = require(script.Parent.Parent:WaitForChild("FinalizerCommon"))

local AssetService = game:GetService('AssetService')
local ServerScriptService = game:GetService('ServerScriptService')
local Workspace = game:GetService('Workspace')
local Terrain: Terrain = Workspace.Terrain

local Spline = require(ServerScriptService:WaitForChild('Utilities'):WaitForChild('Spline'))

-- Options
local trackWidth: number = 49 -- 69
local wallHeight: number = 20
local wallThickness: number = 1
local invisiWallHeight: number = 200
local invisiWallThickness: number = 2
local roadThickness: number = 4
local stationSpacing: number = 3
local maxStationTurn: number = math.rad(2.5)
local foldSafety: number = 0.85
local minWidthFraction: number = 0.5

local arcLengthAttribute: string = 'ARC_LENGTH'

local collisionFidelity: Enum.CollisionFidelity = Enum.CollisionFidelity.PreciseConvexDecomposition
local terrainYieldEvery: number = 400

local EPSILON: number = 1e-4
local UP: Vector3 = Vector3.new(0, 1, 0)

local function flatten(v: Vector3): Vector3
  return Vector3.new(v.X, 0, v.Z)
end

local function unitOrZero(v: Vector3): Vector3
  return if v.Magnitude > EPSILON then v.Unit else Vector3.new()
end

local function travelDirection(point: Types.Point, chord: Vector3): Vector3?
  local direction: Vector3? = point.Direction
  if direction == nil then return nil end

  local flat: Vector3 = flatten(direction)
  if flat.Magnitude < EPSILON then return nil end
  flat = flat.Unit

  if chord.Magnitude > EPSILON and flat:Dot(chord) < 0 then flat = -flat end
  return flat
end

-- Average of the directions into and out of knot j
local function chordAt(knots: { Vector3 }, j: number): Vector3
  local incoming: Vector3 = if j > 1 then unitOrZero(flatten(knots[j] - knots[j - 1])) else Vector3.new()
  local outgoing: Vector3 = if j < #knots then unitOrZero(flatten(knots[j + 1] - knots[j])) else Vector3.new()
  return incoming + outgoing
end

local function buildFrames(stations: { Vector3 }, startDirection: Vector3?, endDirection: Vector3?): { CFrame }
  local count: number = #stations
  local frames: { CFrame } = table.create(count)
  local lastDirection: Vector3 = unitOrZero(flatten(stations[math.min(2, count)] - stations[1]))
  if lastDirection.Magnitude < EPSILON then lastDirection = Vector3.new(0, 0, 1) end

  for i = 1, count do
    local behind: Vector3 = stations[math.max(i - 1, 1)]
    local ahead: Vector3 = stations[math.min(i + 1, count)]
    local direction: Vector3 = unitOrZero(flatten(ahead - behind))
    if direction.Magnitude < EPSILON then direction = lastDirection end
    lastDirection = direction
    frames[i] = CFrame.lookAt(stations[i], stations[i] + direction, UP)
  end

  if startDirection ~= nil then
    frames[1] = CFrame.lookAt(stations[1], stations[1] + startDirection, UP)
  end
  if endDirection ~= nil then
    frames[count] = CFrame.lookAt(stations[count], stations[count] + endDirection, UP)
  end

  return frames
end

local function turnAt(stations: { Vector3 }, i: number): (number, number)
  if i <= 1 or i >= #stations then return 0, 0 end

  local p0: Vector3, p1: Vector3, p2: Vector3 = stations[i - 1], stations[i], stations[i + 1]
  local ax: number, az: number = p1.X - p0.X, p1.Z - p0.Z
  local bx: number, bz: number = p2.X - p1.X, p2.Z - p1.Z
  local cx: number, cz: number = p2.X - p0.X, p2.Z - p0.Z

  local a: number = math.sqrt(ax * ax + az * az)
  local b: number = math.sqrt(bx * bx + bz * bz)
  local c: number = math.sqrt(cx * cx + cz * cz)
  if a < EPSILON or b < EPSILON or c < EPSILON then return 0, 0 end

  local cross: number = ax * bz - az * bx
  local curvature: number = 2 * math.abs(cross) / (a * b * c)

  return curvature, if cross > 0 then 1 elseif cross < 0 then -1 else 0
end

local function feather(values: { number }, radius: number): { number }
  local count: number = #values

  local eroded: { number } = table.create(count)
  for i = 1, count do
    local lowest: number = values[i]
    for k = math.max(1, i - radius), math.min(count, i + radius) do
      lowest = math.min(lowest, values[k])
    end
    eroded[i] = lowest
  end

  local blurred: { number } = table.create(count)
  for i = 1, count do
    local sum: number, taken: number = 0, 0
    for k = math.max(1, i - radius), math.min(count, i + radius) do
      sum += eroded[k]
      taken += 1
    end
    blurred[i] = sum / taken
  end

  return blurred
end

local function edgeHalfWidths(stations: { Vector3 }): ({ number }, { number })
  local count: number = #stations
  local half: number = trackWidth / 2
  local floor: number = half * minWidthFraction

  local left: { number } = table.create(count)
  local right: { number } = table.create(count)
  local narrowed: boolean = false

  for i = 1, count do
    local curvature: number, side: number = turnAt(stations, i)
    local inner: number = half
    if curvature > EPSILON then
      inner = math.clamp(foldSafety / curvature, floor, half)
    end
    if inner < half then narrowed = true end

    left[i] = if side < 0 then inner else half
    right[i] = if side > 0 then inner else half
  end

  if narrowed then
    local radius: number = math.max(1, math.ceil(half / stationSpacing))
    left = feather(left, radius)
    right = feather(right, radius)
  end

  return left, right
end

local function sweepSolid(mesh: EditableMesh, profiles: { { Vector3 } }): ()
  local count: number = #profiles
  local ids: { { number } } = table.create(count)
  for i = 1, count do
    local corners: { Vector3 } = profiles[i]
    ids[i] = {
      mesh:AddVertex(corners[1]),
      mesh:AddVertex(corners[2]),
      mesh:AddVertex(corners[3]),
      mesh:AddVertex(corners[4]),
    }
  end

  for i = 2, count do
    local back: { number }, front: { number } = ids[i - 1], ids[i]
    for k = 1, 4 do
      local n: number = (k % 4) + 1
      mesh:AddTriangle(back[k], front[k], front[n])
      mesh:AddTriangle(back[k], front[n], back[n])
    end
  end

  local first: { number }, last: { number } = ids[1], ids[count]
  mesh:AddTriangle(first[1], first[2], first[3])
  mesh:AddTriangle(first[1], first[3], first[4])
  mesh:AddTriangle(last[1], last[3], last[2])
  mesh:AddTriangle(last[1], last[4], last[3])
end

local function createSolid(profiles: { { Vector3 } }, origin: CFrame, name: string, parent: Instance): MeshPart
  local mesh: EditableMesh = AssetService:CreateEditableMesh()
  sweepSolid(mesh, profiles)

  local part: MeshPart = AssetService:CreateMeshPartAsync(Content.fromObject(mesh), {
    CollisionFidelity = collisionFidelity
  })
  mesh:Destroy()

  part.CFrame = origin
  part.Anchored = true
  part.Name = name
  part.Parent = parent
  return part
end

local function carveTerrain(frames: { CFrame }, stations: { Vector3 }): ()
  local count: number = #stations
  local spans: { number } = table.create(count)
  for i = 1, count do
    local ahead: number = if i < count then (stations[i + 1] - stations[i]).Magnitude else 0
    local behind: number = if i > 1 then (stations[i] - stations[i - 1]).Magnitude else 0
    spans[i] = math.max(ahead, behind, 1)
  end

  for i = 1, count do
    Terrain:FillBlock(
      frames[i] * CFrame.new(0, -13, 0),
      Vector3.new(trackWidth + 4, 27, spans[i] + 2),
      Enum.Material.Asphalt
    )
    if i % terrainYieldEvery == 0 then task.wait() end
  end

  for i = 1, count do
    Terrain:FillBlock(
      frames[i] * CFrame.new(0, 13 + wallHeight / 2, 0),
      Vector3.new(trackWidth + 6, wallHeight + 36, spans[i] + 4),
      Enum.Material.Air
    )
    if i % terrainYieldEvery == 0 then task.wait() end
  end
end

local function buildSection(
  pathFolder: Folder,
  frames: { CFrame },
  leftWidths: { number },
  rightWidths: { number },
  from: number,
  to: number
): ()
  local origin: CFrame = frames[from]
  local up: Vector3 = origin:VectorToObjectSpace(UP)

  local count: number = to - from + 1
  local road: { { Vector3 } } = table.create(count)
  local leftWall: { { Vector3 } } = table.create(count)
  local rightWall: { { Vector3 } } = table.create(count)
  local leftInvisiWall: { { Vector3 } } = table.create(count)
  local rightInvisiWall: { { Vector3 } } = table.create(count)

  for i = from, to do
    local frame: CFrame = frames[i]
    local right: Vector3 = origin:VectorToObjectSpace(frame.RightVector)
    local leftEdge: Vector3 = origin:PointToObjectSpace(frame * Vector3.new(-leftWidths[i], 0, 0))
    local rightEdge: Vector3 = origin:PointToObjectSpace(frame * Vector3.new(rightWidths[i], 0, 0))

    local slot: number = i - from + 1
    road[slot] = {
      leftEdge,
      leftEdge - roadThickness * up,
      rightEdge - roadThickness * up,
      rightEdge,
    }

    local leftOuter: Vector3 = leftEdge - wallThickness * right
    leftWall[slot] = {
      leftOuter + wallHeight * up,
      leftOuter,
      leftEdge,
      leftEdge + wallHeight * up,
    }

    local rightOuter: Vector3 = rightEdge + wallThickness * right
    rightWall[slot] = {
      rightEdge + wallHeight * up,
      rightEdge,
      rightOuter,
      rightOuter + wallHeight * up,
    }

    local leftInvisiInner: Vector3 = leftOuter
    local leftInvisiOuter: Vector3 = leftInvisiInner - invisiWallThickness * right
    leftInvisiWall[slot] = {
      leftInvisiOuter + (invisiWallHeight / 2) * up,
      leftInvisiOuter - (invisiWallHeight / 2) * up,
      leftInvisiInner - (invisiWallHeight / 2) * up,
      leftInvisiInner + (invisiWallHeight / 2) * up,
    }

    local rightInvisiInner: Vector3 = rightOuter
    local rightInvisiOuter: Vector3 = rightInvisiInner + invisiWallThickness * right
    rightInvisiWall[slot] = {
      rightInvisiInner + (invisiWallHeight / 2) * up,
      rightInvisiInner - (invisiWallHeight / 2) * up,
      rightInvisiOuter - (invisiWallHeight / 2) * up,
      rightInvisiOuter + (invisiWallHeight / 2) * up,
    }
  end

  local roadPart: MeshPart = createSolid(road, origin, "Road", pathFolder)
  roadPart.Color = Color3.new(0.1, 0.1, 0.1)
  roadPart:AddTag('ROAD')

  local leftWallPart: MeshPart = createSolid(leftWall, origin, "LeftWall", pathFolder)
  leftWallPart.Color = Color3.new(1, 1, 1)
  leftWallPart:AddTag('WALL')

  local rightWallPart: MeshPart = createSolid(rightWall, origin, "RightWall", pathFolder)
  rightWallPart.Color = Color3.new(1, 1, 1)
  rightWallPart:AddTag('WALL')

  local leftInvisiWallPart: MeshPart = createSolid(leftInvisiWall, origin, "LeftInvisiWall", pathFolder)
  leftInvisiWallPart.Transparency = 0.7
  leftInvisiWallPart:AddTag('WALL')

  local rightInvisiWallPart: MeshPart = createSolid(rightInvisiWall, origin, "RightInvisiWall", pathFolder)
  rightInvisiWallPart.Transparency = 0.7
  rightInvisiWallPart:AddTag('WALL')
end

local function splitIntoRuns(path: { Types.Point }): { { Types.Point } }
  local runs: { { Types.Point } } = {}
  local current: { Types.Point } = {}

  for i: number, point: Types.Point in ipairs(path) do
    table.insert(current, point)

    local nextPoint: Types.Point? = path[i + 1]
    if nextPoint == nil or point.Next == nextPoint then
      if #current >= 2 then table.insert(runs, current) end
      current = {}
    end
  end

  return runs
end

return function (path: { Types.Point }, pathFolder: Folder): ()
  local writeCheckpoint = FinalizerCommon.checkpointWriter()
  local previousCheckpoint: Part? = nil
  local function createCheckpoint(cframe: CFrame, roadDistance: number): ()
    local checkpointPart: Part = writeCheckpoint(cframe)
    if previousCheckpoint ~= nil then
      previousCheckpoint:SetAttribute(arcLengthAttribute, roadDistance)
    end
    previousCheckpoint = checkpointPart
  end

  for _, run: { Types.Point } in ipairs(splitIntoRuns(path)) do
    local knotCount: number = #run

    local knots: { Vector3 } = table.create(knotCount)
    for j: number, point: Types.Point in ipairs(run) do knots[j] = point.Position end

    local directions: { Vector3? } = table.create(knotCount)
    for j: number, point: Types.Point in ipairs(run) do
      directions[j] = travelDirection(point, chordAt(knots, j))
    end

    local segments: { Spline.Segment } = table.create(knotCount - 1)
    for j = 1, knotCount - 1 do
      segments[j] = Spline.segment(
        knots[j - 1], knots[j], knots[j + 1], knots[j + 2],
        directions[j], directions[j + 1]
      )
    end

    local stations: { Vector3 }, knotStations: { number } = Spline.resample(segments, stationSpacing, maxStationTurn)
    local frames: { CFrame } = buildFrames(stations, directions[1], directions[knotCount])
    local leftWidths: { number }, rightWidths: { number } = edgeHalfWidths(stations)

    carveTerrain(frames, stations)

    -- Arc length from the first station up to each station
    local travelled: { number } = Spline.arcLengths(stations)

    local lastCheckpointStation: number = knotStations[1]
    local function checkpointAt(station: number): ()
      createCheckpoint(frames[station], travelled[station] - travelled[lastCheckpointStation])
      lastCheckpointStation = station
    end

    for j = 1, knotCount - 1 do
      checkpointAt(knotStations[j])
      if run[j + 1].Direction ~= nil then checkpointAt(knotStations[j + 1]) end
      buildSection(pathFolder, frames, leftWidths, rightWidths, knotStations[j], knotStations[j + 1])
    end
  end
end
