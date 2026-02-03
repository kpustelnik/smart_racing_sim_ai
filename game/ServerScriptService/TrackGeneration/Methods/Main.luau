--!strict

local Workspace = game:GetService("Workspace")
local CollectionService = game:GetService("CollectionService")
local Map = Workspace:WaitForChild("Map")
local SpawnerPart = Map:WaitForChild("Spawner")
local PrebuiltObjects = Map:WaitForChild('Prebuilt')

local Utilities = game:GetService("ServerScriptService"):WaitForChild("Utilities")
local PriorityQueue = require(Utilities:WaitForChild('PriorityQueue'))
local queue = PriorityQueue()

local Types = require(script.Parent.Parent:WaitForChild("types"))
type Point = Types.Point

-- Options
local PointsNumber: number = 120
local minimumPoints: number = 30
local targetPoints: number = 40
local minimumDistance: number = 130
local maxDistance: number = 500
local trimPercentage: number = 0.6
local pathDistance: number = 100
local PrebuiltStructureMargin = 30
-- local minPrebuilt: number = 4

-- Functions

-- Angle between two vectors
local angleBetween = function (a: Vector3, b: Vector3): number return math.atan2(a.X * b.Z - a.Z * b.X, a.X * b.X + a.Z * b.Z) end

-- Orientation of the ordered triplet (p, q, r)
local orientation2d = function (p: Vector3, q: Vector3, r: Vector3): number
  local value = (q.Z - p.Z) * (r.X - q.X) - (q.X - p.X) * (r.Z - q.Z)
  if math.abs(value) < 0.0001 then
    return 0
  else
    return if value > 0 then 1 else 2
  end
end

-- Checks if two lines intersect
local intersects = function (p1: Vector3, q1: Vector3, p2: Vector3, q2: Vector3): boolean
  local o1: number = orientation2d(p1, q1, p2)
  local o2: number = orientation2d(p1, q1, q2)
  local o3: number = orientation2d(p2, q2, p1)
  local o4: number = orientation2d(p2, q2, q1)

  if o1 ~= o2 and o3 ~= o4 then return true end
  return false
end

-- Returns minimal distance from point to line s -> e
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

--[[
	Micro-optimized code for computing a local space and the global space
	bounding box for a set of parts and attachments as fast as possible.
]]
local function computeTwoBoundingBoxes(basisCFrame1: CFrame, allParts: { BasePart })
	local inverseBasis1: CFrame = basisCFrame1:Inverse()
	local xmin1: number, xmax1: number = math.huge, -math.huge
	local ymin1: number, ymax1: number = math.huge, -math.huge
	local zmin1: number, zmax1: number = math.huge, -math.huge
	local xmin2: number, xmax2: number = math.huge, -math.huge
	local ymin2: number, ymax2: number = math.huge, -math.huge
	local zmin2: number, zmax2: number = math.huge, -math.huge

	for _, part: BasePart in ipairs(allParts) do
    local cframe = part.CFrame
    local csize: Vector3 = part.Size
    local sx: number, sy: number, sz: number = csize.X, csize.Y, csize.Z

    -- Calculation for bounding box in the space of basisCFrame1
    local localCFrame1: CFrame = inverseBasis1 * cframe -- put cframe in our local basis
    local _, _, _,
      t00: number, t01: number, t02: number,
      t10: number, t11: number, t12: number,
      t20: number, t21: number, t22: number = localCFrame1:components()
    local hw1: number = 0.5 * (math.abs(sx * t00) + math.abs(sy * t01) + math.abs(sz * t02))
    local hh1: number = 0.5 * (math.abs(sx * t10) + math.abs(sy * t11) + math.abs(sz * t12))
    local hd1: number = 0.5 * (math.abs(sx * t20) + math.abs(sy * t21) + math.abs(sz * t22))
    local x1: number, y1: number, z1: number = localCFrame1.X, localCFrame1.Y, localCFrame1.Z
    xmin1 = math.min(xmin1, x1 - hw1)
    xmax1 = math.max(xmax1, x1 + hw1)
    ymin1 = math.min(ymin1, y1 - hh1)
    ymax1 = math.max(ymax1, y1 + hh1)
    zmin1 = math.min(zmin1, z1 - hd1)
    zmax1 = math.max(zmax1, z1 + hd1)

    -- Calculation for the bounding box in the global coordinate space
    _, _, _,
      t00, t01, t02,
      t10, t11, t12,
      t20, t21, t22 = cframe:components()
    local hw2: number = 0.5 * (math.abs(sx * t00) + math.abs(sy * t01) + math.abs(sz * t02))
    local hh2: number = 0.5 * (math.abs(sx * t10) + math.abs(sy * t11) + math.abs(sz * t12))
    local hd2: number = 0.5 * (math.abs(sx * t20) + math.abs(sy * t21) + math.abs(sz * t22))
    local x2: number, y2: number, z2: number = cframe.X, cframe.Y, cframe.Z
    xmin2 = math.min(xmin2, x2 - hw2)
    xmax2 = math.max(xmax2, x2 + hw2)
    ymin2 = math.min(ymin2, y2 - hh2)
    ymax2 = math.max(ymax2, y2 + hh2)
    zmin2 = math.min(zmin2, z2 - hd2)
    zmax2 = math.max(zmax2, z2 + hd2)
	end

	local localBoundingBoxOffset: Vector3 = Vector3.new(
		0.5 * (xmin1 + xmax1),
		0.5 * (ymin1 + ymax1),
		0.5 * (zmin1 + zmax1)
	)
	local localBoundingBoxSize: Vector3 = Vector3.new(
		xmax1 - xmin1,
		ymax1 - ymin1,
		zmax1 - zmin1
	)

	return localBoundingBoxOffset, localBoundingBoxSize
end

return function (pathFolder: Folder): { Point }
  assert(SpawnerPart:IsA('BasePart'), 'Spawner part should be a part')

  -- Add points visualisation
  local PointsFolder: Folder = Instance.new("Folder")
  PointsFolder.Name = "Points"
  PointsFolder.Parent = workspace
  
  -- Add prebuilt objects' colliders
  local CollidersFolder: Folder = Instance.new('Folder')
  CollidersFolder.Name = 'PrebuiltColliders'
  CollidersFolder.Parent = Map
  -- Add other colliders
  for _, part: Instance in ipairs(CollectionService:GetTagged('COLLIDER')) do if part:IsA('BasePart') then part.Parent = CollidersFolder end end

  local random: Random = Random.new()
  local Points: { Point } = {}

  local PrebuiltConfigs: { [Instance]: any? } = {}

  for _, object: Instance in ipairs(PrebuiltObjects:GetChildren()) do
    local objParts: { BasePart } = {}
    for _, part in ipairs(object:GetDescendants()) do
      if part:IsA('BasePart') then table.insert(objParts, part) end
    end
    local cf: CFrame = object:GetPivot()
    local boundingBoxOffset: CFrame, boundingBoxSize: Vector3 = computeTwoBoundingBoxes(object:GetPivot(), objParts)
    boundingBoxSize = Vector3.new(boundingBoxSize.X, 1000, boundingBoxSize.Z)

    local ColliderPart: Part = Instance.new('Part')
    ColliderPart.Name = object.Name .. '_Collider'
    ColliderPart.Size = boundingBoxSize + Vector3.new(PrebuiltStructureMargin * 2, 0, PrebuiltStructureMargin * 2)
    ColliderPart.Anchored = true
    ColliderPart.CFrame = cf * CFrame.new(boundingBoxOffset)
    ColliderPart.Transparency = 0.5
    ColliderPart.CanCollide = true
    ColliderPart.Parent = CollidersFolder

    local objectPoints: { [string]: Point? } = {}
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
            Object = object,
            Collider = ColliderPart,
            Bonus = -100000
          }
          table.insert(Points, Point)
          objectPoints[node.Name] = Point
        end
      end
    end

    local ConfigModule: Instance? = object:FindFirstChild('CONFIG')
    if ConfigModule ~= nil and ConfigModule:IsA('ModuleScript') then
      local Config = require(ConfigModule) :: any
      PrebuiltConfigs[object] = Config
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

  -- Perform connecting phase
  local StartId: number, EndId: number = 1, #Points
  local Start: Point, End: Point = Points[StartId], Points[EndId]
  for i: number, cPoint: Point in ipairs(Points) do
    if cPoint.Special == 'START' then
      Start = cPoint
      StartId = i
    elseif cPoint.Special == 'END' then
      End = cPoint
      EndId = i
    end
  end

  local debugParts: boolean = true
  local genPath = function (path: { Point }): ()
    pathFolder:ClearAllChildren()
    if debugParts then -- Create connections for debug to show the currently checked combination
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
  end

  -- Requirements:
  -- 1. No two points can repeat in path (Do in ortools)
  -- 2. No direct connection between end and start (mitigated by minimum path points)
  -- 3. No intersection with previous path (Done)
  -- 4. No intersection with not existing path between end and start (Done)
  -- 5. Respect minimum and maximum distance (Done)
  -- 6. Do not collide with prebuilt structures (Done)
  -- 7. Point can't be too close to any path (Done)
  -- 8. Bending angle should be appropriate based on the directions (DONE)
  -- 9. Make sure that Next properties are tolerated (DONE)
  -- 10. Measure based on distance and angle (DONE)
  -- 11. Measure based on distance to end over target points (Prepared; handle in ortools)
  -- 12. Measure based on track length (prefer longer tracks) (Handle in ortools)
  -- 13. Account the bonus

  queue:put({ Start }, 0) -- Add the starting point to the queue
  local finalPath: {Point}? = nil

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
      local angle: number = angleBetween(curDirection, lastDirection)
      local normalizedAngle: number = angle / math.pi
      if math.abs(normalizedAngle) >= 2/3 then continue end
      if newPoint.Direction ~= nil and math.abs(angleBetween(curDirection, -newPoint.Direction) / math.pi) > 1/4 then continue end
      if lastPoint.Direction ~= nil and math.abs(angleBetween(curDirection, lastPoint.Direction) / math.pi) > 1/4 then continue end
      
      -- Check if the path does not go through a prebuilt structure (unless the starting node or the ending node belongs to the structure)
      local currentColliders: { BasePart } = {}
      for _, collider: Instance in ipairs(CollidersFolder:GetChildren()) do
        if not collider:IsA('BasePart') then continue end
        if lastPoint.Collider == collider or newPoint.Collider == collider then continue end
        table.insert(currentColliders, collider)
      end

      local params: RaycastParams = RaycastParams.new()
      params.FilterDescendantsInstances = currentColliders
      params.FilterType = Enum.RaycastFilterType.Include
      params.IgnoreWater = true
      local collideRaycast: RaycastResult? = Workspace:Raycast(lastPoint.Position, newPoint.Position - lastPoint.Position, params)
      if collideRaycast ~= nil then continue end

      -- Check if the track does not directly through an object
      local raycast: RaycastResult? = Workspace:Raycast(lastPoint.Position, newPoint.Position - lastPoint.Position)
      if raycast ~= nil then
        if raycast.Instance:IsDescendantOf(PrebuiltObjects) then continue end
      end
      
      local measure: number = distance
      if lastPoint.Characteristics == 0 then
        measure += math.abs(normalizedAngle) * 200
      elseif lastPoint.Characteristics == 1 then
        measure -= math.abs(normalizedAngle) * 100
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

  for object: Instance, config: any? in pairs(PrebuiltConfigs) do
    if config == nil then continue end
    local onTrackGeneratedFunc = config.OnTrackGenerated
    if onTrackGeneratedFunc ~= nil then
      local objectPoints: { Point } = {}
      if finalPath ~= nil then
        for _, point: Point in ipairs(finalPath) do
          if point.Object == object then table.insert(objectPoints, point) end
        end
      end
      pcall(onTrackGeneratedFunc, object, objectPoints, finalPath)
    end
  end

  CollidersFolder:Destroy()
  pathFolder:ClearAllChildren()
  PointsFolder:Destroy()

  return finalPath
end