--!strict

local Types = require(script.Parent.Parent:WaitForChild("types"))
local FinalizerCommon = require(script.Parent.Parent:WaitForChild("FinalizerCommon"))

local AssetService = game:GetService('AssetService')
local ServerScriptService = game:GetService('ServerScriptService')

local Workspace = game:GetService('Workspace')
local Terrain: Terrain = Workspace.Terrain

local Spline = require(ServerScriptService:WaitForChild('Utilities'):WaitForChild('Spline'))

local probeSpacing: number = 0.5

local UP: Vector3 = Vector3.new(0, 1, 0)
local levelLook = FinalizerCommon.levelLook

local function createProbePart(size: Vector3, cframe: CFrame): BasePart
  local part: Part = Instance.new('Part')
  part.Size = size
  part.Anchored = true
  part.CFrame = cframe
  part.Transparency = 1
  part.CanCollide = false
  part.CanTouch = false
  part.Parent = Workspace
  return part
end

local function overlapsVolume(volume: BasePart, cframe: CFrame, size: Vector3): boolean
  local overlapParams: OverlapParams = OverlapParams.new()
  overlapParams.FilterType = Enum.RaycastFilterType.Include
  overlapParams.FilterDescendantsInstances = { volume }
  overlapParams.MaxParts = 1

  return #Workspace:GetPartBoundsInBox(cframe, size, overlapParams) > 0
end

local function placeMesh(mesh: EditableMesh, origin: CFrame, name: string, color: Color3, tag: string, parent: Instance): ()
  local part: MeshPart = AssetService:CreateMeshPartAsync(Content.fromObject(mesh))
  mesh:Destroy()

  part.CFrame = origin
  part.Anchored = true
  part.Name = name
  part.Color = color
  part:AddTag(tag)
  part.Parent = parent
end

return function (path: { Types.Point }, pathFolder: Folder): ()
  local createCheckpoint = FinalizerCommon.checkpointWriter()

  -- Path generation
  for i, curPoint: Types.Point in ipairs(path) do
    local nextPoint: Types.Point? = if i ~= #path then path[i + 1] else nil
    if curPoint.Next == nextPoint then -- Check if it's a skip connection (via structure)
      continue
    end
    if nextPoint ~= nil then
      local span: FinalizerCommon.Span = FinalizerCommon.span(path, i, levelLook)
      local segment: Spline.Segment = span.Segment
      local startCF: CFrame = span.StartCFrame
      local endCF: CFrame = span.EndCFrame

      createCheckpoint(startCF)
      if span.EndTangent ~= nil then createCheckpoint(endCF) end

      local localBoundingBoxSize: Vector3 = Vector3.new(49, 20, 1)
      local localBoundingBoxOffset: Vector3 = Vector3.new(0, 10, 0)
      local halfWidth: number = localBoundingBoxSize.X / 2

      local probedPoints: { Vector3 }, t1: number = Spline.sampleSegment(segment, probeSpacing, 4)

      local function frameAt(index: number): CFrame
        local ahead: Vector3 = probedPoints[math.clamp(index + 1, 1, #probedPoints)]
        return levelLook(probedPoints[index], ahead)
      end

      local all_lPoints: { Vector3 }, rev_all_lPoints: { Vector3 } = {}, {}
      local all_rPoints: { Vector3 }, rev_all_rPoints: { Vector3 } = {}, {}
      local all_cfs: { CFrame }, rev_all_cfs: { CFrame } = { startCF }, { endCF }
      table.insert(all_lPoints, (startCF * CFrame.new(-halfWidth, 0, 0)).Position)
      table.insert(all_rPoints, (startCF * CFrame.new(halfWidth, 0, 0)).Position)
      table.insert(rev_all_lPoints, (endCF * CFrame.new(-halfWidth, 0, 0)).Position)
      table.insert(rev_all_rPoints, (endCF * CFrame.new(halfWidth, 0, 0)).Position)

      local runningPositionNow: CFrame = startCF
      local reverseRunningPositionNow: CFrame = endCF
      local ct: number = 0
      local rct: number = t1

      while true do
        -- Walk forward until the next section clears the previous one
        local forwardVolume: BasePart = createProbePart(
          localBoundingBoxSize,
          runningPositionNow * CFrame.new(localBoundingBoxOffset)
        )
        local forwardCF: CFrame? = nil
        while true do
          ct += 1
          if ct >= rct then break end -- Reached the boundary
          local candidate: CFrame = frameAt(ct)
          if not overlapsVolume(forwardVolume, candidate * CFrame.new(localBoundingBoxOffset), localBoundingBoxSize) then
            forwardCF = candidate
            break
          end
        end
        forwardVolume:Destroy()
        if ct >= rct or forwardCF == nil then break end -- Reached the boundary

        table.insert(all_cfs, forwardCF)
        table.insert(all_lPoints, (forwardCF * CFrame.new(-halfWidth, 0, 0)).Position)
        table.insert(all_rPoints, (forwardCF * CFrame.new(halfWidth, 0, 0)).Position)
        runningPositionNow = forwardCF

        -- Find the reverse point
        local reverseVolume: BasePart = createProbePart(
          localBoundingBoxSize,
          reverseRunningPositionNow * CFrame.new(localBoundingBoxOffset)
        )
        local reverseCF: CFrame? = nil
        while true do
          rct -= 1
          if ct >= rct then break end -- Reached the boundary
          local candidate: CFrame = frameAt(rct)
          if not overlapsVolume(reverseVolume, candidate * CFrame.new(localBoundingBoxOffset), localBoundingBoxSize) then
            reverseCF = candidate
            break
          end
        end
        reverseVolume:Destroy()
        if ct >= rct or reverseCF == nil then break end -- Reached the boundary

        table.insert(rev_all_cfs, reverseCF)
        table.insert(rev_all_lPoints, (reverseCF * CFrame.new(-halfWidth, 0, 0)).Position)
        table.insert(rev_all_rPoints, (reverseCF * CFrame.new(halfWidth, 0, 0)).Position)
        reverseRunningPositionNow = reverseCF
      end

      -- The reverse walk collected end -> middle, so replay it middle -> end.
      for j = #rev_all_cfs, 1, -1 do
        table.insert(all_cfs, rev_all_cfs[j])
        table.insert(all_lPoints, rev_all_lPoints[j])
        table.insert(all_rPoints, rev_all_rPoints[j])
      end

      local sectionSpan: number = localBoundingBoxSize.Z + 3
      for _, cf: CFrame in ipairs(all_cfs) do
        Terrain:FillBlock(
          cf * CFrame.new(0, -13, 0),
          Vector3.new(localBoundingBoxSize.X + 4, 27, sectionSpan),
          Enum.Material.Asphalt
        )
      end
      for _, cf: CFrame in ipairs(all_cfs) do
        Terrain:FillBlock(
          cf * CFrame.new(0, 13 + localBoundingBoxSize.Y / 2, 0),
          Vector3.new(localBoundingBoxSize.X + 6, localBoundingBoxSize.Y + 36, sectionSpan + 2),
          Enum.Material.Air
        )
      end

      -- Fix all points to startCF
      local allPoints: number = #all_lPoints
      for j = 1, allPoints do
        all_lPoints[j] = startCF:PointToObjectSpace(all_lPoints[j])
        all_rPoints[j] = startCF:PointToObjectSpace(all_rPoints[j])
      end

      local WallSize: number = localBoundingBoxSize.Y
      local localUp: Vector3 = startCF:VectorToObjectSpace(UP)

      local roadMesh: EditableMesh = AssetService:CreateEditableMesh()
      local leftWallMesh: EditableMesh = AssetService:CreateEditableMesh()
      local rightWallMesh: EditableMesh = AssetService:CreateEditableMesh()

      local lastLPointRoadVertex: number = roadMesh:AddVertex(all_lPoints[1])
      local lastLBPointWallVertex: number = leftWallMesh:AddVertex(all_lPoints[1])
      local lastLTPointWallVertex: number = leftWallMesh:AddVertex(all_lPoints[1] + WallSize * localUp)

      local lastRPointRoadVertex: number = roadMesh:AddVertex(all_rPoints[1])
      local lastRBPointWallVertex: number = rightWallMesh:AddVertex(all_rPoints[1])
      local lastRTPointWallVertex: number = rightWallMesh:AddVertex(all_rPoints[1] + WallSize * localUp)

      for j = 2, allPoints do
        local lPoint: Vector3 = all_lPoints[j]
        local rPoint: Vector3 = all_rPoints[j]

        -- Left wall
        local lbPointWallVertex: number = leftWallMesh:AddVertex(lPoint)
        local ltPointWallVertex: number = leftWallMesh:AddVertex(lPoint + WallSize * localUp)

        -- Inner
        leftWallMesh:AddTriangle(lastLBPointWallVertex, lbPointWallVertex, ltPointWallVertex)
        leftWallMesh:AddTriangle(lastLBPointWallVertex, ltPointWallVertex, lastLTPointWallVertex)
        -- Outer
        leftWallMesh:AddTriangle(lastLBPointWallVertex, ltPointWallVertex, lbPointWallVertex)
        leftWallMesh:AddTriangle(lastLBPointWallVertex, lastLTPointWallVertex, ltPointWallVertex)

        lastLBPointWallVertex = lbPointWallVertex
        lastLTPointWallVertex = ltPointWallVertex

        -- Right wall
        local rbPointWallVertex: number = rightWallMesh:AddVertex(rPoint)
        local rtPointWallVertex: number = rightWallMesh:AddVertex(rPoint + WallSize * localUp)

        -- Inner
        rightWallMesh:AddTriangle(lastRBPointWallVertex, rtPointWallVertex, rbPointWallVertex)
        rightWallMesh:AddTriangle(lastRBPointWallVertex, lastRTPointWallVertex, rtPointWallVertex)
        -- Outer
        rightWallMesh:AddTriangle(lastRBPointWallVertex, rbPointWallVertex, rtPointWallVertex)
        rightWallMesh:AddTriangle(lastRBPointWallVertex, rtPointWallVertex, lastRTPointWallVertex)

        lastRBPointWallVertex = rbPointWallVertex
        lastRTPointWallVertex = rtPointWallVertex

        -- Road
        local lPointRoadVertex: number = roadMesh:AddVertex(lPoint)
        local rPointRoadVertex: number = roadMesh:AddVertex(rPoint)

        roadMesh:AddTriangle(lastLPointRoadVertex, rPointRoadVertex, lPointRoadVertex)
        roadMesh:AddTriangle(lastLPointRoadVertex, lastRPointRoadVertex, rPointRoadVertex)

        lastLPointRoadVertex = lPointRoadVertex
        lastRPointRoadVertex = rPointRoadVertex
      end

      placeMesh(roadMesh, startCF, "Road", Color3.new(0.1, 0.1, 0.1), 'ROAD', pathFolder)
      placeMesh(leftWallMesh, startCF, "LeftWall", Color3.new(1, 1, 1), 'WALL', pathFolder)
      placeMesh(rightWallMesh, startCF, "RightWall", Color3.new(1, 1, 1), 'WALL', pathFolder)
    end
  end
end
