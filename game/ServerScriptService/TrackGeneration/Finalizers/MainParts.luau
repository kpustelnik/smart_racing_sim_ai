--!strict

local Types = require(script.Parent.Parent:WaitForChild("types"))

local AssetService = game:GetService('AssetService')

local Workspace = game:GetService('Workspace')
local Terrain = Workspace:WaitForChild('Terrain')

local Checkpoints = Instance.new("Folder")
Checkpoints.Name = "Checkpoints"
Checkpoints.Parent = Workspace


local function catmullRom(p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: number)
    local t2: number = t * t
    local t3: number = t2 * t
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
        (-p0 + 3*p1 - 3*p2 + p3) * t3
    )
end

local max = function (...: number): number
  local args: { number } = {...}
  local currentMax: number = -math.huge
  for _, v in ipairs(args) do
    if v > currentMax then currentMax = v end
  end
  return currentMax
end

return function (path: { Types.Point }, pathFolder: Folder): ()
  -- Checkpoints
  local checkpointIndex = 0
  local function createCheckpoint(cf)
    local i = checkpointIndex
    checkpointIndex += 1

    local checkpointPart = Instance.new("Part")
    checkpointPart.Name = tostring(i)
    checkpointPart.Color = Color3.new(1, 1, 0)
    checkpointPart.Material = Enum.Material.Neon
    checkpointPart.Size = Vector3.new(20, 30, 2)
    checkpointPart.CFrame = cf * CFrame.new(0, 0, 0)
    checkpointPart.Anchored = true
    checkpointPart.Transparency = 0.5
    checkpointPart.CanCollide = false
    checkpointPart.CanQuery = false
    checkpointPart.CanTouch = false
    checkpointPart.Parent = Checkpoints
  end

  -- Path generation
  for i, curPoint: Types.Point in ipairs(path) do
    local nextPoint: Types.Point? = if i ~= #path then path[i + 1] else nil
    if curPoint.Next == nextPoint then -- Check if it's a skip connection (via structure)
      continue
    end
    if nextPoint ~= nil then
      local prv = if curPoint.Direction ~= nil then curPoint.Position - 400 * Vector3.new(curPoint.Direction.X, 0, curPoint.Direction.Z) elseif i - 1 >= 1 then path[i-1].Position else curPoint.Position
      local a = curPoint.Position
      local b = nextPoint.Position
      local nxt = if nextPoint.Direction ~= nil then nextPoint.Position - 400 * Vector3.new(nextPoint.Direction.X, 0, nextPoint.Direction.Z) elseif i + 2 <= #path then path[i+2].Position else nextPoint.Position

      local startCF = CFrame.lookAt(catmullRom(prv, a, b, nxt, 0), catmullRom(prv, a, b, nxt, 0.002))
      if curPoint.Direction ~= nil then
        startCF = CFrame.lookAt(curPoint.Position, curPoint.Position + curPoint.Direction, Vector3.new(0, 1, 0))
      end
      createCheckpoint(startCF)
      local endCF = CFrame.lookAt(catmullRom(prv, a, b, nxt, 1), catmullRom(prv, a, b, nxt, 1.002))
      if nextPoint.Direction ~= nil then
        endCF = CFrame.lookAt(nextPoint.Position, nextPoint.Position - nextPoint.Direction, Vector3.new(0, 1, 0))
        createCheckpoint(endCF)
      end

      -- Probe the curve points
      local probedPoints: { Vector3 } = {}
      for t = 0, 1, 0.001 do table.insert(probedPoints, catmullRom(prv, a, b, nxt, t)) end
      local t1 = #probedPoints

      -- Probe few extra points
      for t = 1.001, 1.01, 0.001 do table.insert(probedPoints, catmullRom(prv, a, b, nxt, t)) end

      local localBoundingBoxSize: Vector3 = Vector3.new(49, 20, 1)
      local localBoundingBoxOffset: Vector3 = Vector3.new(0, 10, 0)

      local all_lPoints: { Vector3? }, rev_all_lPoints: { Vector3? } = {}, {}
      local all_rPoints: { Vector3? }, rev_all_rPoints: { Vector3? } = {}, {}
      table.insert(all_lPoints, startCF * CFrame.new(-localBoundingBoxSize.X / 2, 0, 0))
      table.insert(all_rPoints, startCF * CFrame.new(localBoundingBoxSize.X / 2, 0, 0))
      table.insert(rev_all_lPoints, endCF * CFrame.new(-localBoundingBoxSize.X / 2, 0, 0))
      table.insert(rev_all_rPoints, endCF * CFrame.new(localBoundingBoxSize.X / 2, 0, 0))
      local runningPositionNow: CFrame = startCF
      local reverseRunningPositionNow: CFrame = endCF
      local ct: number = 0
      local rct: number = t1
      local all_cfs: { CFrame }, rev_all_cfs: { CFrame } = { startCF }, { endCF }
      while true do
        -- Find the start point
        local lastPart = Instance.new('Part')
        lastPart.Size = localBoundingBoxSize
        lastPart.Anchored = true
        lastPart.CFrame = runningPositionNow * CFrame.new(localBoundingBoxOffset)
        lastPart.Transparency = 0.5
        lastPart.Parent = Workspace

        local cf: CFrame? = nil
        while true do 
          ct += 1
          if ct >= rct then break end -- Reached the boundary
          cf = CFrame.lookAt(probedPoints[ct], probedPoints[math.clamp(ct + 1, 1, #probedPoints)])
          
          local overlapParams = OverlapParams.new()
          overlapParams.FilterType = Enum.RaycastFilterType.Include
          overlapParams.FilterDescendantsInstances = { lastPart }
          overlapParams.MaxParts = 1

          local parts = Workspace:GetPartBoundsInBox(cf * CFrame.new(localBoundingBoxOffset), localBoundingBoxSize, overlapParams)
          if #parts == 0 then break end
        end
        lastPart:Destroy()
        if ct >= rct then break end -- Reached the boundary

        local lPoint = cf * CFrame.new(-localBoundingBoxSize.X / 2, 0, 0)
        local rPoint = cf * CFrame.new(localBoundingBoxSize.X / 2, 0, 0)
        table.insert(all_lPoints, lPoint.Position)
        table.insert(all_rPoints, rPoint.Position)
        runningPositionNow = cf

        -- Find the reverse point
        lastPart = Instance.new('Part')
        lastPart.Size = localBoundingBoxSize
        lastPart.Anchored = true
        lastPart.CFrame = reverseRunningPositionNow * CFrame.new(localBoundingBoxOffset)
        lastPart.Transparency = 0.5
        lastPart.Parent = Workspace

        cf = nil
        while true do 
          rct -= 1
          if ct >= rct then break end -- Reached the boundary
          cf = CFrame.lookAt(probedPoints[rct], probedPoints[math.clamp(rct + 1, 1, #probedPoints)])
          
          local overlapParams = OverlapParams.new()
          overlapParams.FilterType = Enum.RaycastFilterType.Include
          overlapParams.FilterDescendantsInstances = { lastPart }
          overlapParams.MaxParts = 1

          local parts = Workspace:GetPartBoundsInBox(cf * CFrame.new(localBoundingBoxOffset), localBoundingBoxSize, overlapParams)
          if #parts == 0 then break end
        end
        lastPart:Destroy()
        if ct >= rct then break end -- Reached the boundary

        lPoint = cf * CFrame.new(-localBoundingBoxSize.X / 2, 0, 0)
        rPoint = cf * CFrame.new(localBoundingBoxSize.X / 2, 0, 0)
        table.insert(rev_all_lPoints, lPoint.Position)
        table.insert(rev_all_rPoints, rPoint.Position)
      end

      for i = #rev_all_cfs, 1, -1 do
        table.insert(all_cfs, rev_all_cfs[i])
        table.insert(all_lPoints, rev_all_lPoints[i])
        table.insert(all_rPoints, rev_all_rPoints[i])
      end
      for _, cf in ipairs(all_cfs) do
        Terrain:FillBlock(cf * CFrame.new(0, -13, 0), Vector3.new(localBoundingBoxSize.X, 0, localBoundingBoxOffset.Z) + Vector3.new(4, 27, 3), Enum.Material.Asphalt)
      end
      for _, cf in ipairs(all_cfs) do
        Terrain:FillBlock(cf * CFrame.new(0, 13 + localBoundingBoxSize.Y / 2, 0), localBoundingBoxSize + Vector3.new(6, 36, 5), Enum.Material.Air)
      end

      -- Fix all points to startCF
      local allPoints = #all_lPoints
      for i = 1, allPoints do
        if all_lPoints[i] ~= 0 then
          all_lPoints[i] = startCF:PointToObjectSpace(all_lPoints[i])
        else
          all_lPoints[i] = nil
        end
        if all_rPoints[i] ~= 0 then
          all_rPoints[i] = startCF:PointToObjectSpace(all_rPoints[i])
        else
          all_rPoints[i] = nil
        end
      end

      local WallSize: number = localBoundingBoxSize.Y

      local roadMesh: EditableMesh = AssetService:CreateEditableMesh()
      local leftWallMesh: EditableMesh = AssetService:CreateEditableMesh()
      local rightWallMesh: EditableMesh = AssetService:CreateEditableMesh()

      local lastLPoint = all_lPoints[1]
      local lastLPointRoadVertex = roadMesh:AddVertex(lastLPoint)
      local lastLBPointWallVertex = leftWallMesh:AddVertex(lastLPoint)
      local lastLTPointWallVertex = leftWallMesh:AddVertex(lastLPoint + Vector3.new(0, WallSize, 0))
      
      local lastRPoint = all_rPoints[1]
      local lastRPointRoadVertex = roadMesh:AddVertex(lastRPoint)
      local lastRBPointWallVertex = rightWallMesh:AddVertex(lastRPoint)
      local lastRTPointWallVertex = rightWallMesh:AddVertex(lastRPoint + Vector3.new(0, WallSize, 0))

      for i = 2, allPoints do
        local lPoint = all_lPoints[i]
        local rPoint = all_rPoints[i]

        -- Left wall
        if lPoint ~= nil then
          local lbPointWallVertex = leftWallMesh:AddVertex(lPoint)
          local ltPointWallVertex = leftWallMesh:AddVertex(lPoint + Vector3.new(0, WallSize, 0))

          -- Inner
          leftWallMesh:AddTriangle(lastLBPointWallVertex, lbPointWallVertex, ltPointWallVertex)
          leftWallMesh:AddTriangle(lastLBPointWallVertex, ltPointWallVertex, lastLTPointWallVertex)
          -- Outer
          leftWallMesh:AddTriangle(lastLBPointWallVertex, ltPointWallVertex, lbPointWallVertex)
          leftWallMesh:AddTriangle(lastLBPointWallVertex, lastLTPointWallVertex, ltPointWallVertex)

          lastLBPointWallVertex = lbPointWallVertex
          lastLTPointWallVertex = ltPointWallVertex
        end

        -- Right wall
        if rPoint ~= nil then
          local rbPointWallVertex = rightWallMesh:AddVertex(rPoint)
          local rtPointWallVertex = rightWallMesh:AddVertex(rPoint + Vector3.new(0, WallSize, 0))

          -- Inner
          rightWallMesh:AddTriangle(lastRBPointWallVertex, rtPointWallVertex, rbPointWallVertex)
          rightWallMesh:AddTriangle(lastRBPointWallVertex, lastRTPointWallVertex, rtPointWallVertex)
          -- Outer
          rightWallMesh:AddTriangle(lastRBPointWallVertex, rbPointWallVertex, rtPointWallVertex)
          rightWallMesh:AddTriangle(lastRBPointWallVertex, rtPointWallVertex, lastRTPointWallVertex)

          lastRBPointWallVertex = rbPointWallVertex
          lastRTPointWallVertex = rtPointWallVertex
        end

        -- Road
        if lPoint ~= nil and rPoint ~= nil then
          local lPointRoadVertex = roadMesh:AddVertex(lPoint)
          local rPointRoadVertex = roadMesh:AddVertex(rPoint)

          roadMesh:AddTriangle(lastLPointRoadVertex, rPointRoadVertex, lPointRoadVertex)
          roadMesh:AddTriangle(lastLPointRoadVertex, lastRPointRoadVertex, rPointRoadVertex)

          lastLPoint = lPoint
          lastLPointRoadVertex = lPointRoadVertex
          lastRPoint = rPoint
          lastRPointRoadVertex = rPointRoadVertex
        elseif lPoint ~= nil then
          local lPointRoadVertex = roadMesh:AddVertex(lPoint)
          roadMesh:AddTriangle(lastLPointRoadVertex, lastRPointRoadVertex, lPointRoadVertex)
          lastLPoint = lPoint
          lastLPointRoadVertex = lPointRoadVertex
        elseif rPoint ~= nil then
          local rPointRoadVertex = roadMesh:AddVertex(rPoint)
          roadMesh:AddTriangle(lastLPointRoadVertex, lastRPointRoadVertex, rPointRoadVertex)
          lastRPoint = rPoint
          lastRPointRoadVertex = rPointRoadVertex
        end
      end

      local roadPart = AssetService:CreateMeshPartAsync(Content.fromObject(roadMesh))
      roadPart.CFrame = startCF
      roadPart.Anchored = true
      roadPart.Parent = pathFolder
      roadPart.Name = "Road"
      roadPart.Color = Color3.new(0.1, 0.1, 0.1)
      roadPart:AddTag('ROAD')

      local leftWallPart = AssetService:CreateMeshPartAsync(Content.fromObject(leftWallMesh))
      leftWallPart.CFrame = startCF
      leftWallPart.Anchored = true
      leftWallPart.Parent = pathFolder
      leftWallPart.Name = "LeftWall"
      leftWallPart.Color = Color3.new(1, 1, 1)
      leftWallPart:AddTag('WALL')

      local rightWallPart = AssetService:CreateMeshPartAsync(Content.fromObject(rightWallMesh))
      rightWallPart.CFrame = startCF
      rightWallPart.Anchored = true
      rightWallPart.Parent = pathFolder
      rightWallPart.Name = "RightWall"
      rightWallPart.Color = Color3.new(1, 1, 1)
      rightWallPart:AddTag('WALL')
    end
  end
end