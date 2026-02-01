--!strict

local Types = require(script.Parent.Parent:WaitForChild("types"))

local ServerStorage = game:GetService('ServerStorage')
local ServerScriptService = game:GetService("ServerScriptService")

local Workspace = game:GetService('Workspace')
local Terrain = Workspace:WaitForChild('Terrain')

local TrackBase = ServerStorage:WaitForChild('TrackBase')
local axis: Vector3 = Vector3.new(0, 0, 1)

local Redupe = ServerScriptService:WaitForChild("Utilities"):WaitForChild("redupe")
local createGhostPreview = require(Redupe:WaitForChild("createGhostPreview"))
local bendPlacement = require(Redupe:WaitForChild("bendPlacement"))
local resizeAlignPairs = require(Redupe:WaitForChild("resizeAlignPairs"))

local Checkpoints = Instance.new("Folder")
Checkpoints.Name = "Checkpoints"
Checkpoints.Parent = Workspace

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
  local prev = nil
  local prevBasis = nil
  for i, curPoint: Types.Point in ipairs(path) do
    local nextPoint: Types.Point? = if i ~= #path then path[i + 1] else nil
    if curPoint.Next == nextPoint then -- Check if it's a skip connection (via structure)
      prev = nil
      continue
    end
    if nextPoint ~= nil then
      local prv = if curPoint.Direction ~= nil then curPoint.Position - 400 * Vector3.new(curPoint.Direction.X, 0, curPoint.Direction.Z) elseif i - 1 >= 1 then path[i-1].Position else curPoint.Position
      local a = curPoint.Position
      local b = nextPoint.Position
      local nxt = if nextPoint.Direction ~= nil then nextPoint.Position - 400 * Vector3.new(nextPoint.Direction.X, 0, nextPoint.Direction.Z) elseif i + 2 <= #path then path[i+2].Position else nextPoint.Position

      local cTrackBase = TrackBase:Clone()
      local TrackBaseTargets = { cTrackBase }
      local TrackBaseParts = {}
      for _, part: Instance in ipairs(cTrackBase:GetDescendants()) do
        if part:IsA("BasePart") then table.insert(TrackBaseParts, part) end
      end

      cTrackBase.Parent = pathFolder
      local startCF = CFrame.lookAt(catmullRom(prv, a, b, nxt, 0), catmullRom(prv, a, b, nxt, 0.002))
      if curPoint.Direction ~= nil then
        startCF = CFrame.lookAt(curPoint.Position, curPoint.Position + curPoint.Direction, Vector3.new(0, 1, 0))
      end
      createCheckpoint(startCF)
      cTrackBase:PivotTo(startCF) -- Position the track base at the start
      local endCF = CFrame.lookAt(catmullRom(prv, a, b, nxt, 1), catmullRom(prv, a, b, nxt, 1.002))
      if nextPoint.Direction ~= nil then
        endCF = CFrame.lookAt(nextPoint.Position, nextPoint.Position - nextPoint.Direction, Vector3.new(0, 1, 0))
        createCheckpoint(endCF)
      end
  
      -- Compute the track base properties
      local localBoundingBoxOffset: Vector3, localBoundingBoxSize: Vector3 = computeTwoBoundingBoxes(startCF, TrackBaseParts)
      local center: CFrame = startCF
      local boundsOffset: Vector3 = localBoundingBoxOffset
      local size: Vector3 = localBoundingBoxSize
      center *= CFrame.new(boundsOffset)
      boundsOffset = Vector3.zero

      local ghostPreview = createGhostPreview(TrackBaseTargets, startCF, localBoundingBoxSize)
      ghostPreview.hide()

      local sizeOnAxis: number = (axis * size).Magnitude

      local placements: { bendPlacement.Placement } = {} :: { bendPlacement.Placement }
      local reversePlacements: { bendPlacement.Placement } = {} :: { bendPlacement.Placement }

      local deltaSize: Vector3 = Vector3.new() --[[ TODO: May adjust the end size ]]
      -- TODO: End Delta position (offset from the main axis may also be put here)

      -- Probe the curve points
      local probedPoints: { Vector3 } = {}
      for t = 0, 1, 0.001 do table.insert(probedPoints, catmullRom(prv, a, b, nxt, t)) end
      local t1 = #probedPoints

      -- Probe few extra points
      for t = 1.001, 1.01 do table.insert(probedPoints, catmullRom(prv, a, b, nxt, t)) end

      local ct: number = 0
      local rct: number = t1 + 1
      local runningPositionNow: CFrame = startCF
      local reverseRunningPositionNow: CFrame = endCF
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
        task.wait() 
        -- lastPart:Destroy()
        if ct >= rct then break end -- Reached the boundary

        -- Calculate the initial placement data
        local copySize: Vector3 = localBoundingBoxSize + (deltaSize * ct / t1)
        local prevSize: Vector3 = localBoundingBoxSize
        if #placements > 0 then prevSize = placements[#placements].Size end
        local placement: bendPlacement.Placement = {
          BoundsOffset = boundsOffset,
          Size = copySize,
          Offset = runningPositionNow:Inverse() * cf,
          PreviousSize = prevSize,
        }

        runningPositionNow = (runningPositionNow * placement.Offset):Orthonormalize()
        placement.CFrame = runningPositionNow
        table.insert(placements, placement)

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
        task.wait() 
        -- lastPart:Destroy()
        if ct >= rct then break end -- Reached the boundary

        -- Calculate the initial placement data
        local reverseCopySize: Vector3 = localBoundingBoxSize + (deltaSize * rct / t1)
        local prevReverseCopySize: Vector3 = localBoundingBoxSize + (deltaSize * (rct - 1) / t1)
        local reversePlacement: bendPlacement.Placement ={
          BoundsOffset = boundsOffset,
          Size = reverseCopySize,
          Offset = reverseRunningPositionNow:Inverse() * cf,
          PreviousSize = prevReverseCopySize,
        }

        reverseRunningPositionNow = (reverseRunningPositionNow * reversePlacement.Offset):Orthonormalize()
        reversePlacement.CFrame = reverseRunningPositionNow
        table.insert(reversePlacements, reversePlacement)
      end
      -- Remove middle placement to avoid overlap
      if #placements > 1 then table.remove(placements, #placements) end

      -- Reconstruct reverse placements
      for i = #reversePlacements, 1, -1 do
        local rp = reversePlacements[i]
        if i == #reversePlacements then -- Remove overlap in the last offset
          local lastPlacement: bendPlacement.Placement = placements[#placements]
          rp.Offset = lastPlacement.CFrame:Inverse() * rp.CFrame
        else
          rp.Offset = reversePlacements[i + 1].CFrame:Inverse() * rp.CFrame -- rp.Offset:Inverse() -- Causing numerical error
        end
        -- ( Skipping CFrame adjustment )
        table.insert(placements, rp)
      end

      -- Place the copies using offsets
      local runningPosition = startCF
      local lastCopy = TrackBaseTargets
      local lastBasis = {
        CFrame = center,
        Offset = boundsOffset,
        Size = localBoundingBoxSize,
      }
      
      local resultsPerTarget = {}
      do for i = 1, #TrackBaseTargets do resultsPerTarget[i] = {} end end
      local resizeAlignPairsCache = {}

      for _, placement in placements do
        runningPosition *= placement.Offset
        runningPosition = runningPosition:Orthonormalize()
        
        local thisCopy = ghostPreview.create(false, runningPosition, size, axis)
        local thisInfo = {
          CFrame = runningPosition,
          Offset = boundsOffset,
          Size = size,
        }
        Terrain:FillBlock(thisInfo.CFrame * CFrame.new(0, -15, 0), size + Vector3.new(7, 27, 3), Enum.Material.Asphalt)
        Terrain:FillBlock(thisInfo.CFrame * CFrame.new(0, 15, 0), size + Vector3.new(3, 36, 5), Enum.Material.Air)

        local resizeAlignResults = resizeAlignPairs(lastCopy, thisCopy, lastBasis, thisInfo, axis, resizeAlignPairsCache)
        for j, resizeAlignResult in resizeAlignResults do
          for _, resultInstance in resizeAlignResult do
            table.insert(resultsPerTarget[j], resultInstance)
          end
        end

        -- This must insert the actual copy after any auxiliary instances
        -- created by resize align.
        for j, resultInstance in thisCopy do table.insert(resultsPerTarget[j], resultInstance) end
        lastCopy = thisCopy
        lastBasis = thisInfo
      end

      if prev ~= nil then
        local firstCopy = TrackBaseTargets
        local firstBasis = {
          CFrame = center,
          Offset = boundsOffset,
          Size = size,
        }
        Terrain:FillBlock(prevBasis.CFrame * CFrame.new(0, -15, 0), size + Vector3.new(7, 27, 3), Enum.Material.Asphalt)
        Terrain:FillBlock(prevBasis.CFrame * CFrame.new(0, 15, 0), size + Vector3.new(2, 36, 3), Enum.Material.Air)
        local alignResults = resizeAlignPairs(prev, firstCopy, prevBasis, firstBasis, axis, resizeAlignPairsCache)
        for j, resizeAlignResult in alignResults do
          for _, resultInstance in resizeAlignResult do
            table.insert(resultsPerTarget[j], resultInstance)
          end
        end
      end
      prevBasis = lastBasis
      prev = lastCopy

      ghostPreview.trim()
    end
  end
end