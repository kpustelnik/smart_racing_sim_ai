--!native
--!optimize 2
type ResizeAlignInfo = {
	CFrame: CFrame,
	Offset: Vector3,
	Size: Vector3,
}

export type ResizeAlignPairsCache = {
	[number]: boolean?, -- Is Wedge fillable mesh
}

local MIN_ZFIGHT_AREA: number = 0.5
local SPANS_FULL_WIDTH_EPSILON: number = 0.051

local copyInstanceProperties = require(script.Parent:WaitForChild('copyInstanceProperties'))
local isWedgeFillable = require(script.Parent:WaitForChild('isWedgeFillable'))
local Constants = require(script.Parent:WaitForChild('Constants'))

--[[
	The return value `t` is a number such that `r1o + t * r1d` is the point of
	closest approach on the first ray between the two rays specified by the
	arguments.
]]
local function intersectRayRay(r1o: Vector3, r1d: Vector3, r2o: Vector3, r2d: Vector3): boolean
	local n: number =
		(r2o - r1o):Dot(r1d) * r2d:Dot(r2d) +
		(r1o - r2o):Dot(r2d) * r1d:Dot(r2d)
	local d: number =
		r1d:Dot(r1d) * r2d:Dot(r2d) -
		r1d:Dot(r2d) * r1d:Dot(r2d)
	if d == 0 then
		return false
	else
		return true, n / d
	end
end

-- Insersect Ray(a + t*b) with plane (origin: o, normal: n), return t of the interesection
function intersectRayPlane(a: Vector3, b: Vector3, o: Vector3, n: Vector3): number
	return (o - a):Dot(n) / b:Dot(n)
end

local function isRotationWorldAlignedModulo90Degrees(cframe: CFrame): boolean
	local x: Vector3 = cframe.XVector:Abs()
	local y: Vector3 = cframe.YVector:Abs()
	local z: Vector3 = cframe.ZVector:Abs()
	if math.abs(x.X + x.Y + x.Z - 1) > 0.001 then return false end
	if math.abs(y.X + y.Y + y.Z - 1) > 0.001 then return false end
	if math.abs(z.X + z.Y + z.Z - 1) > 0.001 then return false end
	return true
end

local function isCandidateForResizing(a: BasePart, aBasis: ResizeAlignInfo, axis: Vector3): boolean
	local cframeInBasis: CFrame = aBasis.CFrame:ToObjectSpace(a.CFrame)
	if not isRotationWorldAlignedModulo90Degrees(cframeInBasis) then return false end

	-- Already resized once, always allow even if it doesn't exactly span the
	-- bounds along the axis anymore.
	if a:HasTag(Constants.RESIZE_TAG) then return true end

	-- Check that the part exactly spans the bounds along the axis.

	-- Size must match
	local sizeInBasis: Vector3 = aBasis.CFrame:VectorToObjectSpace(a.CFrame:VectorToWorldSpace(a.Size)):Abs()
	local size: number = sizeInBasis:Dot(axis)
	if math.abs(size - aBasis.Size:Dot(axis)) > SPANS_FULL_WIDTH_EPSILON * 2 then -- Deliberately a bit loose of an epsilon here
		return false
	end

	-- Position must be centered
	local positionInBounds: Vector3 = cframeInBasis.Position - aBasis.Offset
	local position: number = positionInBounds:Dot(axis)
	if math.abs(position) > SPANS_FULL_WIDTH_EPSILON then return false end

	-- Checks for appropriate part shape
	if a:IsA("Part") then
		if a.Shape == Enum.PartType.Ball then
			-- Nothing to do for balls
			return false
		elseif a.Shape == Enum.PartType.Cylinder then
			-- Check for correct orientation (can only resize along cylinder axis)
			local cylinderAxis: Vector3 = aBasis.CFrame:VectorToObjectSpace(a.CFrame.XVector):Abs()
			if not cylinderAxis:FuzzyEq(axis) then return false end
		end
	end

	return true
end

local function approxSign(n: number): number
	if math.abs(n) < 0.0001 then
		return 0
	else
		return math.sign(n)
	end
end

local function closestUnitVector(forCFrame: CFrame, to: Vector3): Vector3
	local bestDot: number = -math.huge
	local bestVector: Vector3 = Vector3.zero
	local candidates: { Vector3 } = {
		forCFrame.XVector,
		forCFrame.YVector,
		forCFrame.ZVector,
		-forCFrame.XVector,
		-forCFrame.YVector,
		-forCFrame.ZVector,
	}
	for _: number, candidate: Vector3 in candidates do
		local dot: number = candidate:Dot(to)
		if dot > bestDot then
			bestDot = dot
			bestVector = candidate
		end
	end
	return bestVector
end

local function maybeRemoveWedgeAtLocation(location: Vector3, expectedName: string): ()
	local found: {BasePart} = workspace:GetPartBoundsInRadius(location, 0.001)
	for _: number, part: BasePart in found do
		if part:HasTag(Constants.FILL_TAG) and part.Name == expectedName then
			part.Parent = nil
		end
	end
end

local function getPartSizeIncludingSpecialMeshScale(part: BasePart): (Vector3, Vector3, Vector3)
	local mesh: DataModelMesh? = part:FindFirstChildWhichIsA("DataModelMesh")
	if mesh then
		return part.Size * mesh.Scale, mesh.Scale, mesh.Offset
	else
		return part.Size, Vector3.one, Vector3.zero
	end
end

-- Make a wedge as similar to a given BasePart as possible
local function makeWedge(a: BasePart): Part
	if a:IsA("Part") then
		local copy: Part = Instance.fromExisting(a)
		copy.Shape = Enum.PartType.Wedge
		copy.PivotOffset = CFrame.identity
		return copy
	else
		local wedge: Part = Instance.new("Part")
		copyInstanceProperties(a, wedge)
		wedge.Shape = Enum.PartType.Wedge
		wedge.PivotOffset = CFrame.identity
		return wedge
	end
end

local function makeBall(a: BasePart): Part
	local wedge: Part = makeWedge(a)
	wedge.Shape = Enum.PartType.Ball
	return wedge
end

local function smallestNonZeroComponent(v: Vector3): number
	local best: number = math.huge
	if v.X > 0.01 then best = v.X end
	if v.Y > 0.01 and v.Y < best then best = v.Y end
	if v.Z > 0.01 and v.Z < best then best = v.Z end
	return best
end

local function resizeAlignPair(a: BasePart, b: BasePart, aBasis: ResizeAlignInfo, bBasis: ResizeAlignInfo, axis: Vector3, cache: ResizeAlignPairsCache, cacheIndex: number, resultList: {Instance}?): ()
	if not isCandidateForResizing(a, aBasis, axis) then
		return
	end

	-- aBasis.Basis * offset = bBasis.Basis
	assert(aBasis.CFrame ~= bBasis.CFrame, "Same?")
	local localOffset: CFrame = aBasis.CFrame:ToObjectSpace(bBasis.CFrame)
	local localSign: number = math.sign(localOffset.Position:Dot(axis))
	local localSignedAxis: Vector3 = localSign * axis
	local aWorldAxis: Vector3 = aBasis.CFrame:VectorToWorldSpace(localSignedAxis)
	local bWorldAxis: Vector3 = bBasis.CFrame:VectorToWorldSpace(localSignedAxis)

	local aSize: Vector3, aSizeScale: Vector3, aVisualOffset: Vector3 = getPartSizeIncludingSpecialMeshScale(a)
	local bSize: Vector3, bSizeScale: Vector3, bVisualOffset: Vector3 = getPartSizeIncludingSpecialMeshScale(b)
	local aWorldVisualOffset: Vector3 = a.CFrame:VectorToWorldSpace(aVisualOffset)
	local bWorldVisualOffset: Vector3 = b.CFrame:VectorToWorldSpace(bVisualOffset)

	-- sizingAxis and perpendicular axis in the space of the part
	local sizingAxis: Vector3 = a.CFrame:VectorToObjectSpace(aBasis.CFrame:VectorToWorldSpace(axis)):Abs()
	local perpAxis: Vector3 = Vector3.one - axis
	local sizingPerpToAxis: Vector3 = a.CFrame:VectorToObjectSpace(aBasis.CFrame:VectorToWorldSpace(perpAxis)):Abs()

	-- Side to offset towards in the space of the part
	local relativeOffset: CFrame = aBasis.CFrame:ToObjectSpace(bBasis.CFrame)
	local xDir: number = relativeOffset.XVector:Dot(localSignedAxis)
	local yDir: number = relativeOffset.YVector:Dot(localSignedAxis)
	local zDir: number = relativeOffset.ZVector:Dot(localSignedAxis)
	local directions: Vector3 = Vector3.new(approxSign(xDir), approxSign(yDir), approxSign(zDir))
	local sizingPerpDirections: Vector3 = a.CFrame:VectorToObjectSpace(aBasis.CFrame:VectorToWorldSpace(directions))	
	local offsetPerpCorrectSideA: Vector3 = sizingPerpToAxis * sizingPerpDirections * aSize * 0.5
	local offsetPerpCorrectSideB: Vector3 = sizingPerpToAxis * sizingPerpDirections * bSize * 0.5
	if not offsetPerpCorrectSideA:FuzzyEq(offsetPerpCorrectSideB) then
		-- Not interested in weird asymmetric fills here
		-- We could do something here but since we're doing each resize-align pairwise
		-- only without an overarching plan, this case will oscillate between
		-- different kinds of fill with when called multiple times for different
		-- sized things along the duplication path giving a bad feel.
		return
	end

	-- If the two objects are exactly overlapping we can't determine which side
	-- the rotation was towards.
	if directions == Vector3.zero then
		return
	end

	-- Do intersection
	local alignOuterPointA: Vector3 = a.CFrame:PointToWorldSpace(-offsetPerpCorrectSideA)
	local alignOuterPointB: Vector3 = b.CFrame:PointToWorldSpace(-offsetPerpCorrectSideB)
	local alignInnerPointA: Vector3 = a.CFrame:PointToWorldSpace(offsetPerpCorrectSideA)
	local alignInnerPointB: Vector3 = b.CFrame:PointToWorldSpace(offsetPerpCorrectSideB)
	
	local wedgeHeightA: number, wedgeHeightB: number;
	local aBaseOuterLength: number, aBaseInnerLength: number, bBaseOuterLength: number, bBaseInnerLength: number;
	if (aWorldAxis - bWorldAxis).Magnitude < 0.001 then
		-- Directions are effectively parallel, go straight to second case rather
		-- than attempting ray intersection.
		wedgeHeightA = -1
		wedgeHeightB = -1
	else
		-- Directions are not parallel (but still possibly very close to
		-- parallel), do ray intersection.
		local good1: boolean, good2: boolean, good3: boolean, good4: boolean;
		good1, aBaseOuterLength = intersectRayRay(alignOuterPointA, aWorldAxis, alignOuterPointB, -bWorldAxis)
		good2, bBaseOuterLength = intersectRayRay(alignOuterPointB, -bWorldAxis, alignOuterPointA, aWorldAxis)
		good3, aBaseInnerLength = intersectRayRay(alignInnerPointA, aWorldAxis, alignInnerPointB, -bWorldAxis)
		good4, bBaseInnerLength = intersectRayRay(alignInnerPointB, -bWorldAxis, alignInnerPointA, aWorldAxis)
		
		if not good1 or not good2 or not good3 or not good4 then
			-- Parallel case, shouldn't hit this because we already checked for
			-- the directions parallel enough to result in zero denominator.
			warn("Failed to intersect rays for resize-align, shouldn't hit this case, submit a bug report.")
			wedgeHeightA = -1
			wedgeHeightB = -1
		else
			-- First attempt at wedge heights via ray intersection
			wedgeHeightA = aBaseInnerLength - aBaseOuterLength
			wedgeHeightB = bBaseInnerLength - bBaseOuterLength
		end
	end

	if wedgeHeightA < 0 or wedgeHeightB < 0 then
		-- Bring A up to B rather than having them meet in the middle, will be almost
		-- the same thing because the ends are already well aligned.
		local basisSizeOnAxisA: number = (aBasis.Size:Dot(axis) * 0.5)
		local basisSizeOnAxisB: number = (bBasis.Size:Dot(axis) * 0.5)
		local pointOnBasisA: Vector3 = aBasis.CFrame.Position + aWorldAxis * basisSizeOnAxisA
		local pointOnBasisB: Vector3 = bBasis.CFrame.Position - bWorldAxis * basisSizeOnAxisB
		local outerAToSelf: number = intersectRayPlane(alignOuterPointA, aWorldAxis, pointOnBasisA, aWorldAxis)
		local outerAToOther: number = intersectRayPlane(alignOuterPointA, aWorldAxis, pointOnBasisB, bWorldAxis)
		local outerAExtra: number = outerAToOther - outerAToSelf
		aBaseOuterLength = outerAToSelf + 0.5 * outerAExtra

		local innerAToSelf: number = intersectRayPlane(alignInnerPointA, aWorldAxis, pointOnBasisA, aWorldAxis)
		local innerAToOther: number = intersectRayPlane(alignInnerPointA, aWorldAxis, pointOnBasisB, bWorldAxis)
		local innerAExtra: number = innerAToOther - innerAToSelf
		aBaseInnerLength = innerAToSelf + 0.5 * innerAExtra

		local outerBToSelf: number = intersectRayPlane(alignOuterPointB, -bWorldAxis, pointOnBasisB, -bWorldAxis)
		local outerBToOther: number = intersectRayPlane(alignOuterPointB, -bWorldAxis, pointOnBasisA, aWorldAxis)
		local outerBExtra: number = outerBToOther - outerBToSelf
		bBaseOuterLength = outerBToSelf + 0.5 * outerBExtra

		local innerBToSelf: number = intersectRayPlane(alignInnerPointB, -bWorldAxis, pointOnBasisB, -bWorldAxis)
		local innerBToOther: number = intersectRayPlane(alignInnerPointB, -bWorldAxis, pointOnBasisA, aWorldAxis)
		local innerBExtra: number = innerBToOther - innerBToSelf
		bBaseInnerLength = innerBToSelf + 0.5 * innerBExtra
		
		-- Second attempt at wedge heights via plane intersection
		wedgeHeightA = aBaseInnerLength - aBaseOuterLength
		wedgeHeightB = bBaseInnerLength - bBaseOuterLength
	end

	if wedgeHeightA < 0 or wedgeHeightB < 0 then
		warn("Failed to compute wedge heights for resize-align using either approach")
		wedgeHeightA = math.max(0, wedgeHeightA)
		wedgeHeightB = math.max(0, wedgeHeightB)
	end

	local offsetUnsigned: Vector3 = offsetPerpCorrectSideA:Abs()
	local mainOffset: number = math.max(offsetUnsigned.X, offsetUnsigned.Y, offsetUnsigned.Z)
	local zFightAreaA: number = wedgeHeightA * mainOffset * 0.5
	local zFightAreaB: number = wedgeHeightB * mainOffset * 0.5
	local zFightArea: number = math.max(zFightAreaA, zFightAreaB)

	local nonWedgeFillableMesh: boolean = false
	if a:IsA("MeshPart") or a:IsA("PartOperation") then
		local cachedIsWedgeFillable: boolean? = cache[cacheIndex]
		if cachedIsWedgeFillable == nil then
			cachedIsWedgeFillable = isWedgeFillable(a)
			cache[cacheIndex] = cachedIsWedgeFillable
		end
		nonWedgeFillableMesh = not cachedIsWedgeFillable
	end

	-- Always create wedge if it's a cylinder (we already checked that it's the right orientation
	-- in isCandidateForResizing)
	local enoughZFightForWedge: boolean = not directions:FuzzyEq(Vector3.zero) and zFightArea > MIN_ZFIGHT_AREA
	local isCylinder: boolean = a:IsA("Part") and a.Shape == Enum.PartType.Cylinder
	local hasGoodWedgeHeights: boolean = wedgeHeightA > 0 and wedgeHeightB > 0
	local createFill: boolean = (enoughZFightForWedge or isCylinder) and hasGoodWedgeHeights and not nonWedgeFillableMesh

	local aOriginalLength: number = (aSize * sizingAxis).Magnitude
	local bOriginalLength: number = (bSize * sizingAxis).Magnitude

	local aDeltaLength: number = (createFill and aBaseOuterLength or aBaseInnerLength) - aOriginalLength * 0.5
	local bDeltaLength: number = (createFill and bBaseOuterLength or bBaseInnerLength) - bOriginalLength * 0.5
	if isCylinder then
		aDeltaLength += wedgeHeightA * 0.5
		bDeltaLength += wedgeHeightB * 0.5
	end

	-- Remove any fill wedges that were making this side of the model whole
	-- before from a previous redupe operation. This will happen if you redupe
	-- in that direction and then delete the result but that leaves behind an
	-- unpaired resizealigned edge with wedges.
	local checkForWedgeToRemoveLocationA: Vector3 = a.Position + aWorldAxis * aOriginalLength * 0.5
	local checkForWedgeToRemoveLocationB: Vector3 = b.Position - bWorldAxis * bOriginalLength * 0.5
	maybeRemoveWedgeAtLocation(checkForWedgeToRemoveLocationA, a.Name .. Constants.FILL_NAME_SUFFIX)
	maybeRemoveWedgeAtLocation(checkForWedgeToRemoveLocationB, b.Name .. Constants.FILL_NAME_SUFFIX)

	if createFill then
		if isCylinder then
			-- For cylinders fill intersection with one ball
			local ball: Part = makeBall(a)
			ball.Name ..= Constants.FILL_NAME_SUFFIX
			ball.CFrame = CFrame.fromMatrix(
				a.Position + aWorldAxis * (aBaseOuterLength + wedgeHeightA * 0.5),
				closestUnitVector(a.CFrame, -offsetPerpCorrectSideA):Cross(aWorldAxis),
				aWorldAxis
			) + aWorldVisualOffset
			ball.Size = Vector3.one * smallestNonZeroComponent(sizingPerpToAxis * aSize)
			ball:AddTag(Constants.FILL_TAG)
			ball.Parent = a.Parent
		else
			-- For boxes fill intersection with two wedges
			local desiredZVectorForA: Vector3 = a.CFrame:VectorToWorldSpace(-offsetPerpCorrectSideA)
			local wedgeA: Part = makeWedge(a)
			wedgeA.Name ..= Constants.FILL_NAME_SUFFIX
			wedgeA.CFrame = CFrame.fromMatrix(
				a.Position + aWorldAxis * (aBaseOuterLength + wedgeHeightA * 0.5),
				closestUnitVector(a.CFrame, desiredZVectorForA):Cross(aWorldAxis),
				aWorldAxis
			) + aWorldVisualOffset
			local aPerSizeInBasis: Vector3 = aBasis.CFrame:VectorToObjectSpace(a.CFrame:VectorToWorldSpace(sizingPerpToAxis * aSize))
			local aSizeInBasis: Vector3 = axis * wedgeHeightA + aPerSizeInBasis
			wedgeA.Size = wedgeA.CFrame:VectorToObjectSpace(aBasis.CFrame:VectorToWorldSpace(aSizeInBasis)):Abs()
			wedgeA:AddTag(Constants.FILL_TAG)
			wedgeA.Parent = a.Parent
			if resultList then
				table.insert(resultList, wedgeA)
			end

			local desiredZVectorForB: Vector3 = b.CFrame:VectorToWorldSpace(-offsetPerpCorrectSideB)
			local wedgeB: Part = makeWedge(b)
			wedgeB.Name ..= Constants.FILL_NAME_SUFFIX
			wedgeB.CFrame = CFrame.fromMatrix(
				b.Position - bWorldAxis * (bBaseOuterLength + wedgeHeightB * 0.5),
				-closestUnitVector(b.CFrame, desiredZVectorForB):Cross(bWorldAxis),
				-bWorldAxis
			) + bWorldVisualOffset
			local bPerSizeInBasis: Vector3 = bBasis.CFrame:VectorToObjectSpace(b.CFrame:VectorToWorldSpace(sizingPerpToAxis * bSize))
			local bSizeInBasis: Vector3 = axis * wedgeHeightB + bPerSizeInBasis
			wedgeB.Size = wedgeB.CFrame:VectorToObjectSpace(bBasis.CFrame:VectorToWorldSpace(bSizeInBasis)):Abs()
			wedgeB:AddTag(Constants.FILL_TAG)
			wedgeB.Parent = b.Parent
			if resultList then
				table.insert(resultList, wedgeB)
			end
		end
	end

	local aNewSize: Vector3 = a.Size + (sizingAxis * aDeltaLength) / aSizeScale
	if aNewSize:Dot(sizingAxis) <= 0.001 then
		a.Parent = nil
	else
		a.Size = aNewSize
		a.CFrame += aWorldAxis * aDeltaLength * 0.5
		a:AddTag(Constants.RESIZE_TAG)
	end

	local bNewSize: Vector3 = b.Size + (sizingAxis * bDeltaLength) / bSizeScale
	if bNewSize:Dot(sizingAxis) <= 0.001 then
		b.Parent = nil
	else
		b.Size = bNewSize
		b.CFrame -= bWorldAxis * bDeltaLength * 0.5
		b:AddTag(Constants.RESIZE_TAG)
	end
end

local function filterChildList(children: {Instance}): {Instance}
	local filteredList: {Instance} = {}
	local fillTag: string = Constants.FILL_TAG
	for _, ch: Instance in children do
		if not ch.Archivable then
			continue
		end
		if ch:HasTag(fillTag) then
			continue
		end
		table.insert(filteredList, ch)
	end
	return filteredList
end

local function resizeAlignPairsRecursive(a: {Instance}, b: {Instance}, aBasis: ResizeAlignInfo, bBasis: ResizeAlignInfo, axis: Vector3, cache: ResizeAlignPairsCache, cacheIndex: number, resultList: {Instance}?): number
	-- assert(#a == #b, "Mismatched number of instances to resize-align recursive")
	for i: number, itemA: Instance in a do
		local itemB: Instance? = nil
		for _, x in b do
			if x.Name == itemA.Name then
				itemB = x
				break
			end
		end
		if itemB == nil then
			warn("Item missing")
			continue
		end
		if itemA.ClassName ~= itemB.ClassName then
			warn("Mismatched instance types in resize-align:", itemA.ClassName, itemB.ClassName)
			continue
		end
		if itemA.Name ~= itemB.Name then
			warn("Mismatched instance names in resize-align:", itemA.Name, itemB.Name)
			continue
		end
		if itemA:IsA("BasePart") and itemB:IsA("BasePart") then
			resizeAlignPair(itemA, itemB, aBasis, bBasis, axis, cache, cacheIndex, resultList)
			cacheIndex += 1
		else
			local childListA: {Instance} = filterChildList(itemA:GetChildren())
			local childListB: {Instance} = filterChildList(itemB:GetChildren())
			cacheIndex = resizeAlignPairsRecursive(childListA, childListB, aBasis, bBasis, axis, cache, cacheIndex, nil)
		end
	end
	return cacheIndex
end

local function resizeAlignPairs(a: {Instance}, b: {Instance}, aBasis: ResizeAlignInfo, bBasis: ResizeAlignInfo, axis: Vector3, cache: ResizeAlignPairsCache): {{Instance}}
	assert(#a == #b, "Mismatched number of instances to resize-align")
	local allResults: {{Instance}} = {}
	for i: number, item: Instance in a do
		local resultsForItem: {Instance} = {}
		resizeAlignPairsRecursive({a[i]}, {b[i]}, aBasis, bBasis, axis, cache, 1, resultsForItem)
		allResults[i] = resultsForItem
	end
	return allResults
end

return resizeAlignPairs