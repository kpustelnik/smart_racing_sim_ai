local Workspace = game:GetService("Workspace")
local Map = Workspace:WaitForChild("Map")
local SpawnerPart = Map:WaitForChild("Spawner")
local PrebuiltObjects = Map:WaitForChild('Prebuilt')

local Types = require(script:WaitForChild("types"))
type Point = Types.Point

-- TODO

local PointsNumber: number = 120
local minimumPoints: number = 30
local targetPoints: number = 40
local minimumDistance: number = 80
local maxDistance: number = 500
local trimPercentage: number = 0.6
local pathDistance: number = 100
-- local minPrebuilt: number = 4

local angleBetween = function (a: Vector3, b: Vector3): number
  return math.atan2(a.X * b.Z - a.Z * b.X, a.X * b.X + a.Z * b.Z)
end

local orientation2d = function (p: Vector3, q: Vector3, r: Vector3): number
  local value = (q.Z - p.Z) * (r.X - q.X) - (q.X - p.X) * (r.Z - q.Z)
  if math.abs(value) < 0.0001 then
    return 0
  else
    return if value > 0 then 1 else 2
  end
end
local intersects = function (p1: Vector3, q1: Vector3, p2: Vector3, q2: Vector3): boolean
  local o1: number = orientation2d(p1, q1, p2)
  local o2: number = orientation2d(p1, q1, q2)
  local o3: number = orientation2d(p2, q2, p1)
  local o4: number = orientation2d(p2, q2, q1)

  if o1 ~= o2 and o3 ~= o4 then return true end
  return false
end
local getPathDistance = function (s: Vector3, e: Vector3, point: Vector3): number
  local ray = Ray.new(s, e-s).Unit
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

assert(SpawnerPart:IsA('BasePart'), 'Spawner part should be a part')

local PointsFolder: Folder = Instance.new("Folder")
PointsFolder.Name = "Points"
PointsFolder.Parent = workspace

local random = Random.new()
local Points: { Point } = {}

-- Add prebuilt objects
for _, object in ipairs(PrebuiltObjects:GetChildren()) do
  local objectPoints = {}
  local objectNodes: Instance? = object:FindFirstChild('Nodes')
  if objectNodes ~= nil then
    for _, node in ipairs(objectNodes:GetChildren()) do
      if node:IsA('BasePart') then
        local SpecialAttribute = node:GetAttribute('SPECIAL')
        local Point: Point = {
          Position = node.Position,
          Characteristics = 0,
          Special = if SpecialAttribute ~= nil and typeof(SpecialAttribute) == 'string' then SpecialAttribute else nil,
          Direction = node.CFrame.LookVector,
          Part = node,
          Bonus = -100000
        }
        table.insert(Points, Point)
        objectPoints[node.Name] = Point
      end
    end
  end
  local ConfigModule = object:FindFirstChild('CONFIG')
  if ConfigModule ~= nil and ConfigModule:IsA('ModuleScript') then
    local Config = require(ConfigModule) :: any
    if Config.NodeMap ~= nil then
      for from: string, to: string in pairs(Config.NodeMap) do
        if objectPoints[from] ~= nil and objectPoints[to] ~= nil then
          objectPoints[from].Next = objectPoints[to]
        end
      end
    end
  end
end

-- Generate points
for _: number = 1, PointsNumber do
  local pos: Vector3? = nil
  while pos == nil do
    local offsetX: number = random:NextNumber(-SpawnerPart.Size.X / 2, SpawnerPart.Size.X / 2)
    local offsetZ: number = random:NextNumber(-SpawnerPart.Size.Z / 2, SpawnerPart.Size.Z / 2)
    local pointCFrame: CFrame = SpawnerPart.CFrame * CFrame.new(offsetX, 0, offsetZ)
    local result: RaycastResult<BasePart>? = Workspace:Raycast(pointCFrame.Position, Vector3.new(0, -500, 0))
    if result ~= nil then
      if result.Instance:IsDescendantOf(PrebuiltObjects) then continue end
      pos = result.Position + Vector3.new(0, 0.5, 0)
    end
  end
  if pos == nil then continue end

  local reverseCharacteristics: boolean = random:NextInteger(1, 10) <= 2

  local PointVisualisation: Part = Instance.new("Part")
  PointVisualisation.Size = Vector3.new(5, 5, 5)
  PointVisualisation.Position = pos
  PointVisualisation.Color = if reverseCharacteristics then Color3.new(1, 0, 0) else Color3.new(0, 1, 0)
  PointVisualisation.Anchored = true
  PointVisualisation.Parent = PointsFolder
  PointVisualisation.CanCollide = false
  local x: BasePart = PointVisualisation
  table.insert(Points, {
    Position = pos,
    Characteristics = if not reverseCharacteristics then 0 else 1,
    Part = x
  })
end

local Utilities = game:GetService("ServerScriptService"):WaitForChild("Utilities")
local PriorityQueue = require(Utilities:WaitForChild('PrioritiyQueue'))
local queue = PriorityQueue()

-- Perform connecting phase
local Start: Point, End: Point = Points[1], Points[#Points]
for _, cPoint: Point in ipairs(Points) do
  if cPoint.Special == 'START' then
    Start = cPoint
  elseif cPoint.Special == 'END' then
    End = cPoint
  end
end

local pathFolder = Instance.new("Folder")
pathFolder.Name = 'Path'
pathFolder.Parent = Workspace
local genPath = function (path)
  pathFolder:ClearAllChildren()
  for i, curPoint: Point in ipairs(path) do
    local nextPoint: Point? = if i ~= #path then path[i + 1] else nil
    if curPoint.Next == nextPoint then continue end
    if nextPoint ~= nil then
      local PathPart: Part = Instance.new("Part")
      PathPart.Color = Color3.new(0.5, 0.5, 0.5)
      PathPart.CFrame = CFrame.lookAt((nextPoint.Position + curPoint.Position) / 2, nextPoint.Position, Vector3.new(0, 1, 0))
      PathPart.Size = Vector3.new(2, 1, (nextPoint.Position - curPoint.Position).Magnitude)
      PathPart.Anchored = true
      PathPart.Parent = pathFolder
      PathPart.Color = Color3.new(1,1,0)
      PathPart.CanCollide = false
    end
  end
end

queue:put({ Start }, 0) -- Add the starting point to the queue
local finalPath = nil

local j: number = 0
while not queue:empty() do
  local curPath: {Point}, curMeasure: number = queue:pop()
  local lastPoint = curPath[#curPath]
  -- Got to the END node - finished the path
  if lastPoint == End then
    if minimumPoints ~= nil and #curPath < minimumPoints then continue end
    finalPath = curPath
    break
  end
  local lastDirection: Vector3? = if lastPoint.Direction ~= nil then lastPoint.Direction elseif #curPath > 1 then (lastPoint.Position - curPath[#curPath - 1].Position) else nil
  if lastDirection == nil then continue end -- Shouldn't generally happen

  for _, newPoint: Point in ipairs(Points) do
    if table.find(curPath, newPoint) ~= nil then continue end -- Point already in path - skip
    if newPoint == End and lastPoint == Start then continue end

    local distance: number = (newPoint.Position - lastPoint.Position).Magnitude
    if distance < minimumDistance then continue end
    if distance > maxDistance then continue end

    -- Check intersections
    if lastPoint ~= Start and newPoint ~= End and intersects(lastPoint.Position, newPoint.Position, End.Position, Start.Position) then
      continue
    end
    local doesIntersect: boolean = false
    for i = 1, #curPath - 2 do
      if intersects(lastPoint.Position, newPoint.Position, curPath[i].Position, curPath[i + 1].Position) then
        doesIntersect = true
        break
      end
    end
    if doesIntersect then continue end

    -- Ensure every path is far enough
    local tooClose = false
    for i = 1, #curPath - 2 do
      if getPathDistance(curPath[i].Position, curPath[i + 1].Position, newPoint.Position) < pathDistance then tooClose = true; break end
      if getPathDistance(lastPoint.Position, newPoint.Position, curPath[i].Position) < pathDistance then tooClose = true; break end
    end
    if tooClose then continue end

    local curDirection: Vector3 = newPoint.Position - lastPoint.Position
    local angle = angleBetween(curDirection, lastDirection)
    local normalizedAngle = angle / math.pi
    if math.abs(normalizedAngle) >= 2/3 then continue end
    if newPoint.Direction ~= nil and math.abs(angleBetween(curDirection, -newPoint.Direction) / math.pi) > 1/4 then continue end
    if lastPoint.Direction ~= nil and math.abs(angleBetween(curDirection, lastPoint.Direction) / math.pi) > 1/4 then continue end

    -- Check if the track does not directly through an object (should improve!!!)
    local raycast = Workspace:Raycast(lastPoint.Position, newPoint.Position - lastPoint.Position)
    if raycast ~= nil then
      if raycast.Instance:IsDescendantOf(PrebuiltObjects) then
        continue
      end
    end
    
    local measure: number = 0
    if lastPoint.Characteristics == 0 then
      measure = distance + math.abs(normalizedAngle) * 200
    elseif lastPoint.Characteristics == 1 then
      measure = distance - math.abs(normalizedAngle) * 100
    end
    measure = measure + 300 * (math.abs(#curPath - targetPoints) - targetPoints) -- Prefer longer tracks
    if #curPath > targetPoints then
      measure = measure + (End.Position - newPoint.Position).Magnitude
    end
    for _, point in ipairs(curPath) do
      if point.Bonus == nil then continue end
      measure = measure + point.Bonus
    end

    local newPath: { Point } = table.clone(curPath)
    table.insert(newPath, newPoint)
    if newPoint.Next ~= nil then table.insert(newPath, newPoint.Next) end
    queue:put(newPath, measure)
  end

  j = (j + 1)
  if j % 200 == 0 then
    genPath(curPath)
    print(curMeasure, #curPath)
    -- Trim the queue
    --[[ Trimming by queue size
    local newSize = queue.current_size * (1 - trimPercentage)
    while queue.current_size > newSize do queue:pop() end
    ]]
    -- Trimming by path size
    local newPathSize = math.floor(#curPath * (1 - trimPercentage))
    -- TODO: Adaptive trimPercentage
    for i = queue.current_size, 1, -1 do
      local val = queue.heap_val[i]
      if #val > newPathSize then -- Remove
        queue.heap_val[i], queue.heap_pri[i] = queue.heap_val[queue.current_size], queue.heap_pri[queue.current_size]
        queue.heap_val[queue.current_size], queue.heap_pri[queue.current_size] = nil, nil
        queue.current_size -= 1
        queue:sink(i)
      end
    end
    task.wait()
  end
end

print("Finished")
genPath(finalPath)

local Finalizer = require(script:WaitForChild("Finalizers"):WaitForChild("Simple"))
Finalizer(finalPath, pathFolder)

return {}