--!strict
--!nolint UnreachableCode

local Workspace = game:GetService("Workspace")
local CollectionService = game:GetService("CollectionService")
local Map = Workspace:WaitForChild("Map")
local SpawnerPart = Map:WaitForChild("Spawner")
local PrebuiltObjects = Map:WaitForChild('Prebuilt')
local HttpService = game:GetService("HttpService")

local Types = require(script.Parent.Parent:WaitForChild("types"))
local PathGeometry = require(script.Parent.Parent:WaitForChild("PathGeometry"))
local BoundingBox = require(game:GetService("ServerScriptService"):WaitForChild("Utilities"):WaitForChild("BoundingBox"))
type Point = Types.Point

-- Options
local PointsNumber: number = 120
local minimumDistance: number = 130
local maxDistance: number = 500
local PrebuiltStructureMargin = 30
local pathDistance: number = 120

local angleBetween = PathGeometry.angleBetween
local intersects = PathGeometry.intersects
local getPathDistance = PathGeometry.distanceToSegment

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
          local currentColliders: { Instance } = {}

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
    local headers: { [string]: string } = { ["Content-Type"] = "application/json" }
    return HttpService:RequestAsync({
      Url = "http://localhost:8000/generate_path",
      Method = "POST",
      Headers = headers,
      Body = jsonData
    })
  end)
  if not success then
    error("Failed to contact remote path solver: " .. tostring(result))
  end
  local pathResult = HttpService:JSONDecode(result.Body :: string)
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