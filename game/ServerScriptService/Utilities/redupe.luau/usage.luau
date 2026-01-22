local Workspace: Workspace = game:GetService('Workspace')

local base: Model = Workspace.Base
local target: BasePart = Workspace.Target
local startCF: CFrame = base:GetPivot()

local axis: Vector3 = Vector3.new(0, 0, 1)

local ServerScriptService = game:GetService("ServerScriptService")
local redupe = ServerScriptService.Utilities["redupe.luau"]
local createGhostPreview = require(redupe.createGhostPreview)
local bendPlacement = require(redupe.bendPlacement)
local resizeAlignPairs = require(redupe.resizeAlignPairs)
local Constants = require(redupe.Constants)

--[[
	Micro-optimized code for computing a local space and the global space
	bounding box for a set of parts and attachments as fast as possible.
]]
local function computeTwoBoundingBoxes(basisCFrame1: CFrame, allParts: { BasePart })
	local basisX: number, basisY: number, basisZ: number = basisCFrame1.X, basisCFrame1.Y, basisCFrame1.Z
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

	local globalBoundingBoxOffset: Vector3 = Vector3.new(
		0.5 * (xmin2 + xmax2) - basisX,
		0.5 * (ymin2 + ymax2) - basisY,
		0.5 * (zmin2 + zmax2) - basisZ
	)
	local globalBoundingBoxSize: Vector3 = Vector3.new(
		xmax2 - xmin2,
		ymax2 - ymin2,
		zmax2 - zmin2
	)

	return localBoundingBoxOffset, localBoundingBoxSize, globalBoundingBoxOffset, globalBoundingBoxSize
end

local targets: { Instance } = { base }
local allParts = {}
for _, part in ipairs(base:GetDescendants()) do
  if part:IsA("BasePart") then
    table.insert(allParts, part)
  end
end
local localBoundingBoxOffset: Vector3, localBoundingBoxSize: Vector3, globalBoundingBoxOffset: Vector3, globalBoundingBoxSize: Vector3 = computeTwoBoundingBoxes(startCF, allParts)
local center: CFrame = startCF
local boundsOffset: Vector3 = localBoundingBoxOffset
local size: Vector3 = localBoundingBoxSize
center *= CFrame.new(boundsOffset)
boundsOffset = Vector3.zero
--[[
local function getCopyCount(): number?
  local EndCFrame: CFrame = target.CFrame
  local endOffset: Vector3 = EndCFrame:ToObjectSpace(startCF).Position
  local offsetOnAxis: number = (axis * endOffset).Magnitude
  local sizeOnAxis: number = (axis * localBoundingBoxSize).Magnitude
  if math.abs(sizeOnAxis) < 0.01 then
    warn(`Redupe: Spacing settings result in too many copies ({offsetOnAxis / sizeOnAxis})`)
    return nil
  end
  local count: number = math.floor(offsetOnAxis / sizeOnAxis + 0.5) + 1
  assert(count == count, "NaN")
  return count
end
]]

local function computeRedundantRotationLimit(): number
  local angles: Vector3 = Vector3.new(target.CFrame.Rotation:ToEulerAnglesXYZ())
  local nonZeroCount: number = 0
  if math.abs(angles.X) > 0.001 then nonZeroCount += 1 end
  if math.abs(angles.Y) > 0.001 then nonZeroCount += 1 end
  if math.abs(angles.Z) > 0.001 then nonZeroCount += 1 end
  if nonZeroCount ~= 1 then
    -- If the count is, zero, there's definitely no overlap.
    -- If the count is two, having it overlap would require very precisely
    -- chosen analytic angles, almost impossible to achive in practice.
    -- With three, I don't think it's possible to have exact overlap
    -- given the way I do roll application.
    return math.huge
  end

  local theAngle: number = math.abs(angles:Dot(Vector3.one))
  local steps: number = (math.pi * 2) / theAngle
  assert(steps == steps, "NaN")

  -- does steps = p / q for integers p, q?
  for q = 1, 128 do
    local p: number = math.floor(steps * q + 0.5)
    if math.abs(steps - p / q) < 0.001 then
      return math.floor(steps * q + 0.5)
    end
  end
  return math.huge
end

--[[
local function isSinglePartWithSpacingCase(): boolean
  -- Only intrested in single part here
  if not currentSettings.SelectionIsSinglePart then
    return false
  end
  -- Copy count always has interesting spacing
  if not currentSettings.UseSpacing then
    return true
  end
  -- Copy spacing non-defaults
  if currentSettings.CopySpacing ~= 1 or currentSettings.CopyPadding ~= 0 then
    return true
  end
  return false
end
]]

local function shouldResizeAlign(): boolean
  -- Most importantly, don't resize if the user explicitly asked not to
  --[[
  if not currentSettings.ResizeAlign then
    return false
  end
  ]]

  -- Second, we have to resize if we've resized in the past
  local resizeTag: string = Constants.RESIZE_TAG
  for _, target: Instance in targets do
    if target:HasTag(resizeTag) then
      return true
    end
    for _, part: Instance in target:QueryDescendants("BasePart") do
      if part:HasTag(resizeTag) then
        return true
      end
    end
  end

  -- Third, if it's a single part with spacing, ResizeAligning would
  -- effectively ignore the spacing, so don't do it.
  --[[
  if isSinglePartWithSpacingCase() then
    return false
  end
  ]]

  -- Finally, as an optimization, don't resize if there's no rotation
  if target.CFrame.Rotation:FuzzyEq(CFrame.identity) then
    return false
  end

  return true
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
local function catmullRomCentripetal(p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, u: number, alpha: number?)
	alpha = alpha or 0.5
	u = math.clamp(u, 0, 1)

	local EPS = 1e-6

	local function tj(ti: number, a: Vector3, b: Vector3): number
		local d = (b - a).Magnitude
		-- centripetal: ti + (|b-a|^alpha)
		return ti + math.max(d, EPS) ^ alpha
	end

	-- Parameter values based on chord lengths
	local t0 = 0.0
	local t1 = tj(t0, p0, p1)
	local t2 = tj(t1, p1, p2)
	local t3 = tj(t2, p2, p3)

	-- Map u in [0,1] to t in [t1,t2]
	local t = t1 + (t2 - t1) * u

	local function lerpVec(a: Vector3, b: Vector3, s: number): Vector3
		return a + (b - a) * s
	end

	local function lerpT(a: Vector3, b: Vector3, ta: number, tb: number): Vector3
		local denom = (tb - ta)
		if math.abs(denom) < EPS then
			return a -- degenerate; points too close in parameter space
		end
		return lerpVec(a, b, (t - ta) / denom)
	end

	-- Barry–Goldman construction
	local A1 = lerpT(p0, p1, t0, t1)
	local A2 = lerpT(p1, p2, t1, t2)
	local A3 = lerpT(p2, p3, t2, t3)

	local B1 = lerpT(A1, A2, t0, t2)
	local B2 = lerpT(A2, A3, t1, t3)

	local C  = lerpT(B1, B2, t1, t2)
	return C
end

local ghostPreview = createGhostPreview(targets, startCF, localBoundingBoxSize)
local function updatePlacement(): ({{ Instance }}?, CFrame?)
  ghostPreview.hide()

  -- TODO: Adjust the points further on
  local prv: Vector3 = startCF.Position + startCF:VectorToWorldSpace(axis) * 200
  local a: Vector3 = startCF.Position
  local b: Vector3 = target.CFrame.Position
  local nxt: Vector3 = b + target.CFrame.lookVector * 200

  local sizeOnAxis: number = (axis * localBoundingBoxSize).Magnitude

  local placements: { bendPlacement.Placement } = {} :: { bendPlacement.Placement }
  local reversePlacements: { bendPlacement.Placement } = {} :: { bendPlacement.Placement }

  local deltaSize: Vector3 = localBoundingBoxSize --[[ TODO: May add end size ]] - localBoundingBoxSize
  -- TODO: End Delta position (offset from the main axis may also be put here)
  local probedPoints = {}
  for t = 0, 1, 0.001 do
    table.insert(probedPoints, catmullRom(prv, a, b, nxt, t))
  end
  local t1 = #probedPoints
  for t = 1.001, 1.1 do table.insert(probedPoints, catmullRom(prv, a, b, nxt, t)) end
  local endCF = CFrame.lookAt(probedPoints[t1], probedPoints[t1+1])

  local lastPos: Vector3 = startCF.Position
  local reverseLastPos: Vector3 = endCF.Position

  local ct: number = 1
  local rct: number = t1
  local runningPositionNow: CFrame = startCF
  local reverseRunningPositionNow: CFrame = endCF
  while true do
    -- Find the start point
    while true do 
      ct += 1
      if ct > rct then break end
      if (probedPoints[ct] - lastPos).Magnitude > sizeOnAxis * 1 then
        break
      end
    end
    if ct > rct then break end

    if #placements > 0 then
      runningPositionNow *= placements[#placements].Offset
      runningPositionNow = runningPositionNow:Orthonormalize()
    end

    local copySize: Vector3 = localBoundingBoxSize + (deltaSize * ct / t1)
    local prevSize = localBoundingBoxSize
    if #placements > 0 then prevSize = placements[#placements].Size end
    local placement: bendPlacement.Placement = {
      BoundsOffset = boundsOffset,
      Size = copySize,
      Offset = CFrame.new(),
      PreviousSize = prevSize,
    }
    
    local originalOffset = CFrame.new(-axis * sizeOnAxis)
    local testPosition: Vector3
    while true do -- Bend and check if better splice position exists
      placement.Offset = originalOffset
      local curPos = probedPoints[ct]
      local nxtPos = probedPoints[math.clamp(ct + 1, 1, #probedPoints)]
      local cf = CFrame.lookAt(curPos, nxtPos)
      local rotation = runningPositionNow.Rotation:ToObjectSpace(cf.Rotation)
      bendPlacement(placement, axis, rotation, -1, 0, 1)
      testPosition = (runningPositionNow * placement.Offset).Position
      local anyAffected = false
      while ct < rct and (testPosition - probedPoints[ct]).Magnitude > (testPosition - probedPoints[ct + 1]).Magnitude do
        ct += 1
        anyAffected = true
      end
      if not anyAffected then break end 
    end
    if ct > rct then break end
    placement.Offset += Vector3.new(0, probedPoints[ct].Y - testPosition.Y, 0)
    placement.CFrame = runningPositionNow * placement.Offset
    table.insert(placements, placement)
    lastPos = probedPoints[ct]

    -- Find the reverse point
    while true do 
      rct -= 1
      if ct > rct then break end
      if (probedPoints[rct] - reverseLastPos).Magnitude > sizeOnAxis * 1 then
        reverseLastPos = probedPoints[rct]
        break
      end
    end
    if ct > rct then break end

    if #reversePlacements > 0 then
      reverseRunningPositionNow *= reversePlacements[#reversePlacements].Offset
      reverseRunningPositionNow = reverseRunningPositionNow:Orthonormalize()
    end

    local reverseCopySize: Vector3 = localBoundingBoxSize + (deltaSize * rct / t1)
    local prevReverseCopySize: Vector3 = localBoundingBoxSize + (deltaSize * (rct - 1) / t1)
    local reversePlacement: bendPlacement.Placement ={
      BoundsOffset = boundsOffset,
      Size = reverseCopySize,
      Offset = CFrame.new(),
      PreviousSize = prevReverseCopySize,
    }
    
    local reverseOriginalOffset = CFrame.new(axis * sizeOnAxis)
    while true do -- Bend and check if better splice position exists
      reversePlacement.Offset = reverseOriginalOffset
      local curPos = probedPoints[rct]
      local nxtPos = probedPoints[math.clamp(rct + 1, 1, #probedPoints)]
      local cf = CFrame.lookAt(curPos, nxtPos)
      local rotation = reverseRunningPositionNow.Rotation:ToObjectSpace(cf.Rotation)
      bendPlacement(reversePlacement, -axis, rotation, -1, 0, 1)
      testPosition = (reverseRunningPositionNow * reversePlacement.Offset).Position
      local anyAffected = false
      while ct < rct and (testPosition - probedPoints[rct]).Magnitude > (testPosition - probedPoints[rct - 1]).Magnitude do
        rct -= 1
        anyAffected = true
      end
      if not anyAffected then break end 
    end
    if ct > rct then break end
    reversePlacement.Offset += Vector3.new(0, probedPoints[rct].Y - testPosition.Y, 0)
    reversePlacement.CFrame = reverseRunningPositionNow * reversePlacement.Offset
    table.insert(reversePlacements, reversePlacement)
    reverseLastPos = probedPoints[rct]

    --[[
    if copySize.X > 0.001 and copySize.Y > 0.001 and copySize.Z > 0.001 then dont put
    end]]
  end
  if #placements > 1 then table.remove(placements, #placements) end
  print(placements, reversePlacements)

  for i = #reversePlacements, 1, -1 do
    local rp = reversePlacements[i]
    if i == #reversePlacements then -- Remove overlap in the last offset
      local lastPlacement = placements[#placements]
      rp.Offset = lastPlacement.CFrame:Inverse() * reversePlacements[i].CFrame
    else
      rp.Offset = rp.Offset:Inverse()
    end
    table.insert(placements, rp)
  end

  --[[
  for i = 1, copyCount - 1 do
    local mid: Vector3?
    while true do
      t += 0.001
      if t > 1 then break end
      mid = catmullRom(prv, a, b, nxt, t)
      if (mid - lastPos).Magnitude > sizeOnAxis * 1 then
        lastPos = mid
        break
      end
    end
    if mid == nil then
      copyCount = i
      break
    end
    assert(t == t, "NaN")
    local nxtVector3: Vector3 = catmullRom(prv, a, b, nxt, t + 1/(copyCount + 1))

    local realPosition: CFrame = CFrame.lookAt(mid, nxtVector3)
    local copyPosition: CFrame = startCF * CFrame.new(axis * i * sizeOnAxis)

    local p: Part = Instance.new("Part")
    p.CFrame = realPosition
    p.Size = Vector3.new(2, 2, 2)
    p.Parent = Workspace
    local copySize: Vector3 = localBoundingBoxSize + (deltaSize * t)
    if copySize.X > 0.001 and copySize.Y > 0.001 and copySize.Z > 0.001 then
      table.insert(placements, {
        CFrame = copyPosition,
        RealCFrame = realPosition,
        BoundsOffset = boundsOffset,
        Size = copySize,
        Offset = CFrame.new(),
        PreviousSize = Vector3.new(),
      })
    end
  end
  ]]

  local doResizeAlign = shouldResizeAlign()

  -- Place using offsets
  local runningPosition = startCF
  local lastCopy = targets
  local lastBasis = {
    CFrame = center,
    Offset = boundsOffset,
    Size = localBoundingBoxSize,
  }
  local resultsPerTarget = {}
  for i = 1, #targets do
    resultsPerTarget[i] = {}
  end
  local resizeAlignPairsCache = {}
  for i, placement in placements do
    runningPosition *= placement.Offset
    runningPosition = runningPosition:Orthonormalize()
    local thisCopy = ghostPreview.create(false, runningPosition, placement.Size, axis)
    local thisInfo = {
      CFrame = runningPosition,
      Offset = boundsOffset,
      Size = placement.Size,
    }

    if doResizeAlign and true then
      task.wait(0.05)
      local resizeAlignResults = resizeAlignPairs(lastCopy, thisCopy, lastBasis, thisInfo, axis, resizeAlignPairsCache)
      for j, resizeAlignResult in resizeAlignResults do
        for _, resultInstance in resizeAlignResult do
          table.insert(resultsPerTarget[j], resultInstance)
        end
      end
    end

    -- This must insert the actual copy after any auxiliary instances
    -- created by resize align.
    for j, resultInstance in thisCopy do
      table.insert(resultsPerTarget[j], resultInstance)
    end
    lastCopy = thisCopy
    lastBasis = thisInfo
  end

  -- If we came full circle, resizeAlign the last copy to the first
  --[[
  if doResizeAlign and done and #placements >= redundantLimit - 1 then
    local firstCopy = targets
    local firstBasis = {
      CFrame = center,
      Offset = boundsOffset,
      Size = size,
    }
    local alignResults = resizeAlignPairs(lastCopy, firstCopy, lastBasis, firstBasis, draggerContext.PrimaryAxis, resizeAlignPairsCache)
    for j, resizeAlignResult in alignResults do
      for _, resultInstance in resizeAlignResult do
        table.insert(resultsPerTarget[j], resultInstance)
      end
    end
  end
  ]]

  ghostPreview.trim()

  -- Remember the copy count for snapping purposes
--  draggerContext.SnapMultiplier = copyCount - 1
--  currentSettings.FinalCopyCount = math.min(copyCount, redundantLimit)
  return resultsPerTarget, runningPosition
end


-- OWN


print("Before loop")
	updatePlacement(false)
--[[
  ghostPreview.hide()
ghostPreview.trim()]]