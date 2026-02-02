

local Workspace = game:GetService("Workspace")
local CollectionService = game:GetService("CollectionService")
local Map = Workspace:WaitForChild("Map")
local SpawnerPart = Map:WaitForChild("Spawner")
local PrebuiltObjects = Map:WaitForChild('Prebuilt')
local HttpService = game:GetService("HttpService")

local Types = require(script.Parent.Parent:WaitForChild("types"))
type Point = Types.Point

-- Options
local PointsNumber: number = 120
local minimumDistance: number = 130
local maxDistance: number = 500
local PrebuiltStructureMargin = 30
local pathDistance: number = 120

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

  -- Prepare data to send to the remote service
  local cpm3D = {}
  local prox3D = {}
  local cpm2D = {}
  local measure2D = {}
  local measure3D = {}
  local next1D = {}
  local distancesToEnd = {}
  local bonusPoints = {}
  local locations = {}
  local remoteData = {
    measure2D = measure2D, -- Points for double combinations
    measure3D = measure3D, -- Points for triple combinations
    canPickMatrix3D = cpm3D, -- Whether two consecutive nodes can be picked
    canPickMatrix2D = cpm2D, -- Whether three consecutive nodes can be picked
    proximity3D = prox3D, -- Whether a node can be picked as next considering previously picked lines
    next1D = next1D, -- Enforced next node of each node
    distancesToEnd = distancesToEnd, -- Distance of each node to the end node
    bonusPoints = bonusPoints, -- Bonus points for each node
    locations = locations, -- Locations of each node (X Y Z)
    startId = StartId, -- Start node id
    endId = EndId, -- End node id
  }

  for a = 1, #Points, 1 do
    local pointA: Point = Points[a]
    locations[a] = {pointA.Position.X, pointA.Position.Y, pointA.Position.Z}
    cpm2D[a] = {}
    prox3D[a] = {}
    cpm3D[a] = {}
    --imatrix[a] = {}
    measure2D[a] = {}
    measure3D[a] = {}
    distancesToEnd[a] = (End.Position - pointA.Position).Magnitude
    bonusPoints[a] = if pointA.Bonus ~= nil then pointA.Bonus else 0
    next1D[a] = 0
    for b = 1, #Points, 1 do
      local pointB: Point = Points[b]
      local abDirection: Vector3 = pointB.Position - pointA.Position

      -- Add measure
      measure2D[a][b] = abDirection.Magnitude
      measure3D[a][b] = {}

      if pointB == pointA.Next then next1D[a] = b end -- (Condition 9)
      cpm2D[a][b] = a ~= b 
      if pointA.Next ~= pointB then -- Skip if b is next of a
        if cpm2D[a][b] then 
          cpm2D[a][b] = abDirection.Magnitude >= minimumDistance and abDirection.Magnitude <= maxDistance -- (Condition 5)
        end
        if cpm2D[a][b] then -- (Condition 4)
          if a ~= StartId and b ~= EndId and intersects(pointA.Position, pointB.Position, End.Position, Start.Position) then
            cpm2D[a][b] = false
          end
        end

        if cpm2D[a][b] then -- (Condition 6)
          -- Check if the path does not go through a prebuilt structure (unless the starting node or the ending node belongs to the structure)
          local currentColliders: { BasePart } = {}

          for _, collider in ipairs(CollidersFolder:GetChildren()) do
            if not collider:IsA('BasePart') then continue end
            if pointA.Collider == collider or pointB.Collider == collider then continue end
            table.insert(currentColliders, collider)
          end

          local params: RaycastParams = RaycastParams.new()
          params.FilterDescendantsInstances = currentColliders
          params.FilterType = Enum.RaycastFilterType.Include
          params.IgnoreWater = true
          local collideRaycast: RaycastResult? = Workspace:Raycast(pointA.Position, abDirection, params)
          if collideRaycast ~= nil then cpm2D[a][b] = false end
        end

        if cpm2D[a][b] then -- (Condition 6 also but more directly)
          -- Check if the track does not directly through an object
          local raycast: RaycastResult? = Workspace:Raycast(pointA.Position, abDirection)
          if raycast ~= nil then
            if raycast.Instance:IsDescendantOf(PrebuiltObjects) then cpm2D[a][b] = false end
          end
        end

        if cpm2D[a][b] and pointA.Direction ~= nil then -- (Condition 8 with start Direction)
          local lastDirection: Vector3 = pointA.Direction
          local curDirection: Vector3 = abDirection
          local normalizedAngle = angleBetween(curDirection, lastDirection) / math.pi
          if math.abs(normalizedAngle) >= 2/3 then cpm2D[a][b] = false end -- Too big bending angle
          if pointB.Direction ~= nil and math.abs(angleBetween(curDirection, -pointB.Direction) / math.pi) > 1/4 then cpm2D[a][b] = false end -- Too big end angle
          if math.abs(normalizedAngle) > 1/4 then cpm2D[a][b] = false end -- Too big start angle

          if pointA.Characteristics == 0 then
            measure2D[a][b] += math.abs(normalizedAngle) * 200
          elseif pointA.Characteristics == 1 then
            measure2D[a][b] -= math.abs(normalizedAngle) * 100
          end
        end
      end
      
      prox3D[a][b] = {}
      cpm3D[a][b] = {}
      -- imatrix[a][b] = {}
      for c = 1, #Points, 1 do
        local pointC: Point = Points[c]

        measure3D[a][b][c] = 0

        local bcDirection: Vector3 = pointC.Position - pointB.Position
        cpm3D[a][b][c] = true -- Assume can connect
        if cpm3D[a][b][c] and pointB.Direction == nil then -- (Condition 8 without start direction)
          local lastDirection: Vector3 = abDirection
          local curDirection: Vector3 = bcDirection
          local normalizedAngle = angleBetween(curDirection, lastDirection) / math.pi
          if math.abs(normalizedAngle) >= 2/3 then cpm3D[a][b][c] = false end -- Too big bending angle
          if pointC.Direction ~= nil and math.abs(angleBetween(curDirection, -pointC.Direction) / math.pi) > 1/4 then cpm3D[a][b][c] = false end -- Too big end angle
          
          if pointB.Characteristics == 0 then
            measure3D[a][b][c] += math.abs(normalizedAngle) * 200
          elseif pointB.Characteristics == 1 then
            measure3D[a][b][c] -= math.abs(normalizedAngle) * 100
          end
        end
        
        -- imatrix[a][b][c] = {}
        prox3D[a][b][c] = getPathDistance(pointA.Position, pointB.Position, pointC.Position) >= pathDistance -- (Condition 7) Point can't be too close to some path (line)
        --[[
        -- Too expensive to calculate - should be done on backend
        for d = 1, #Points, 1 do
          condWait(a, b, c, d)
          local pointD: Point = Points[d]
          imatrix[a][b][c][d] = true
          if pointA.Next ~= pointB then -- Skip if b is next of a
            if a ~= b and a ~= c and a ~= d and b ~= c and b ~= d and c ~= d then
              if intersects(pointA.Position, pointB.Position, pointC.Position, pointD.Position) then -- (Condition 3)
                imatrix[a][b][c][d] = false
              end
            end
          end
        end
        ]]
      end
    end
  end

  error('Remote solver got deprecated; please use the local solver instead.')

  local jsonData: string = HttpService:JSONEncode(remoteData)

  local success: boolean, result = pcall(function()
    return HttpService:RequestAsync({
      Url = "http://localhost:8000/generate_path",
      Method = "POST",
      Headers = {
        ["Content-Type"] = "application/json"
      },
      Body = jsonData
    })
  end)
  if not success then
    error("Failed to contact remote path solver: " .. tostring(result))
  end
  local pathResult = HttpService:JSONDecode(result.Body)
  local finalPath: {Point} = {}
  for _, i in ipairs(pathResult.path) do table.insert(finalPath, Points[i]) end

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