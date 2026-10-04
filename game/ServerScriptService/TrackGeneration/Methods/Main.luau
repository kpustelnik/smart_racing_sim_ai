--!strict

local Workspace = game:GetService("Workspace")
local CollectionService = game:GetService("CollectionService")
local Map = Workspace:WaitForChild("Map")
local SpawnerPart = Map:WaitForChild("Spawner")
local PrebuiltObjects = Map:WaitForChild('Prebuilt')

local Utilities = game:GetService("ServerScriptService"):WaitForChild("Utilities")
local PriorityQueue = require(Utilities:WaitForChild('PriorityQueue'))
local BoundingBox = require(Utilities:WaitForChild('BoundingBox'))

local Types = require(script.Parent.Parent:WaitForChild("types"))
local PathGeometry = require(script.Parent.Parent:WaitForChild("PathGeometry"))
type Point = Types.Point

-- Options
local PointsNumber: number = 120
local minimumPoints: number = 30
local targetPoints: number = 40
local minimumDistance: number = 130
local maxDistance: number = 500
local maxBendFraction: number = 2/3
local minTurnRadius: number = 60
local trimPercentage: number = 0.6
local minTrimPercentage: number = 0.3
local maxTrimPercentage: number = 0.85
local trimStep: number = 0.1
local pathDistance: number = 100
local PrebuiltStructureMargin = 30
-- local minPrebuilt: number = 4

local angleBetween = PathGeometry.angleBetween
local intersects = PathGeometry.intersects
local getPathDistance = PathGeometry.distanceToSegment


return function (pathFolder: Folder): { Point }?
  assert(SpawnerPart:IsA('BasePart'), 'Spawner part should be a part')

  local queue: PriorityQueue.PriorityQueue<{ Point }> = PriorityQueue.new()

  -- Add points visualisation
  local PointsFolder: Folder = Instance.new("Folder")
  PointsFolder.Name = "Points"
  PointsFolder.Parent = Workspace
  
  -- Add prebuilt objects' colliders
  local CollidersFolder: Folder = Instance.new('Folder')
  CollidersFolder.Name = 'PrebuiltColliders'
  CollidersFolder.Parent = Map
  -- Add other colliders
  for _, part: Instance in ipairs(CollectionService:GetTagged('COLLIDER')) do if part:IsA('BasePart') then part.Parent = CollidersFolder end end

  local seedAttribute: unknown? = Workspace:GetAttribute("SEED")
  local seed: number = if typeof(seedAttribute) == "number" then seedAttribute else os.time()
  local random: Random = Random.new(seed)
  local Points: { Point } = {}

  local PrebuiltConfigs: { [Instance]: any? } = {}

  for _, object: Instance in ipairs(PrebuiltObjects:GetChildren()) do
    if not object:IsA('PVInstance') then continue end
    local objParts: { BasePart } = {}
    for _, part in ipairs(object:GetDescendants()) do
      if part:IsA('BasePart') then table.insert(objParts, part) end
    end
    local cf: CFrame = object:GetPivot()
    local boundingBoxOffset: Vector3, boundingBoxSize: Vector3 = BoundingBox.compute(cf, objParts)
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
          node.CanCollide = false
          node.CanTouch = false
          node.CanQuery = false

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
          local fromPoint: Point? = objectPoints[from]
          local toPoint: Point? = objectPoints[to]
          if fromPoint ~= nil and toPoint ~= nil then
            fromPoint.Next = toPoint
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

  local debugParts: boolean = false
  local genPath = function (path: { Point }): ()
    pathFolder:ClearAllChildren()
    if debugParts then -- Create connections for debug to show the currently checked combination
      for i: number, curPoint: Point in ipairs(path) do
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
  local currentTrim: number = trimPercentage
  local windowBestDepth: number = 0
  local previousWindowBestDepth: number? = nil
  while not queue:empty() do
    local curPath: {Point} = queue:pop()
    windowBestDepth = math.max(windowBestDepth, #curPath)
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
      if math.abs(normalizedAngle) >= maxBendFraction then continue end
      local incomingLength: number = if #curPath > 1
        then (lastPoint.Position - curPath[#curPath - 1].Position).Magnitude
        else math.huge
      local tangentLength: number = math.min(incomingLength, curDirection.Magnitude) / 2
      if math.abs(angle) >= 2 * math.atan(tangentLength / minTurnRadius) then continue end
      if newPoint.Direction ~= nil and math.abs(angleBetween(curDirection, -newPoint.Direction) / math.pi) > 1/4 then continue end
      if lastPoint.Direction ~= nil and math.abs(angleBetween(curDirection, lastPoint.Direction) / math.pi) > 1/4 then continue end
      
      -- Check if the path does not go through a prebuilt structure (unless the starting node or the ending node belongs to the structure)
      local currentColliders: { Instance } = {}
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
      -- Trim the queue
      --[[ Trimming by queue size
      local newSize = queue.current_size * (1 - trimPercentage)
      while queue.current_size > newSize do queue:pop() end
      ]]
      --[[
        Trimming by path size. Every 200 expansions without reaching end, deeper queued partial paths
        are dropped, forcing the search back onto shorter prefixes and into different branches.
      ]]
      if previousWindowBestDepth ~= nil then
        if windowBestDepth > previousWindowBestDepth then
          currentTrim = math.max(minTrimPercentage, currentTrim - trimStep)
        else
          currentTrim = math.min(maxTrimPercentage, currentTrim + trimStep)
        end
      end
      previousWindowBestDepth = windowBestDepth
      windowBestDepth = 0

      local newPathSize: number = math.floor(#curPath * (1 - currentTrim))

      local keptValues: { { Point } } = {}
      local keptPriorities: { number } = {}
      for i = 1, queue.current_size do
        if #queue.heap_val[i] <= newPathSize then
          table.insert(keptValues, queue.heap_val[i])
          table.insert(keptPriorities, queue.heap_pri[i])
        end
      end

      if #keptValues > 0 and #keptValues < queue.current_size then
        queue:initialize()
        for i: number, value: {Point} in keptValues do
          queue:put(value, keptPriorities[i])
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