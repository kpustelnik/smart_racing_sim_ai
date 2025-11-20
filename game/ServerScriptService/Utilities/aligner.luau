-- https://github.com/stravant/roblox-resizealign/blob/main/src/main.server.lua

type Face = {
	Object: BasePart,
	Normal: Enum.NormalId,
	IsWedge: boolean?,
}

local function otherNormals(dir: Vector3)
	if math.abs(dir.X) > 0 then
		return Vector3.new(0, 1, 0), Vector3.new(0, 0, 1)
	elseif math.abs(dir.Y) > 0 then
		return Vector3.new(1, 0, 0), Vector3.new(0, 0, 1)
	else
		return Vector3.new(1, 0, 0), Vector3.new(0, 1, 0)
	end
end

local function extend(v: Vector3, amount: number)
	return v.Unit * (v.Magnitude + amount) 
end

local function getFacePoints(face: Face)
	local hsize = face.Object.Size / 2
	local cf = face.Object.CFrame
	if face.IsWedge then
		return {
			cf:PointToWorldSpace((Vector3.xAxis + Vector3.yAxis + Vector3.zAxis) * hsize),
			cf:PointToWorldSpace((Vector3.xAxis + Vector3.yAxis + Vector3.zAxis) * hsize),
			cf:PointToWorldSpace((-Vector3.xAxis - Vector3.yAxis - Vector3.zAxis) * hsize),
			cf:PointToWorldSpace((-Vector3.xAxis - Vector3.yAxis - Vector3.zAxis) * hsize),
		}
	else
		local faceDir = Vector3.fromNormalId(face.Normal)
		local faceA, faceB = otherNormals(faceDir)
		faceDir, faceA, faceB = faceDir*hsize, faceA*hsize, faceB*hsize
		--
		return {
			cf:PointToWorldSpace(faceDir + faceA + faceB);
			cf:PointToWorldSpace(faceDir + faceA - faceB);
			cf:PointToWorldSpace(faceDir - faceA - faceB);
			cf:PointToWorldSpace(faceDir - faceA + faceB);
		}
	end
end

local function copyPartProps (fromPart: BasePart, toPart: BasePart)
	toPart.Anchored     = fromPart.Anchored
	toPart.Massless     = fromPart.Massless
	toPart.RootPriority = fromPart.RootPriority
	toPart.CustomPhysicalProperties = fromPart.CustomPhysicalProperties
	--
	toPart.CanCollide   = fromPart.CanCollide
	toPart.CanTouch     = fromPart.CanTouch
	toPart.CanQuery     = fromPart.CanQuery
	toPart.CollisionGroup = fromPart.CollisionGroup
	--
	toPart.Color        = fromPart.Color
	toPart.CastShadow   = fromPart.CastShadow
	toPart.Material     = fromPart.Material
	toPart.Reflectance  = fromPart.Reflectance
	toPart.Transparency = fromPart.Transparency
	toPart.MaterialVariant = fromPart.MaterialVariant
end

local function getPoints(part: BasePart): {Vector3}
	local hsize = part.Size / 2
	local cf = part.CFrame
	local points = {}
	for i = -1, 1, 2 do
		for j = -1, 1, 2 do
			for k = -1, 1, 2 do
				table.insert(points, cf:PointToWorldSpace(Vector3.new(i, j, k) * hsize))
			end
		end
	end
	return points
end

local function getNormal(face: Face)
	if face.IsWedge then
		local hsize = face.Object.Size / 2
		local cf = face.Object.CFrame
		return (cf.YVector * hsize.Z - cf.ZVector * hsize.Y).Unit
	else
		return face.Object.CFrame:VectorToWorldSpace(Vector3.fromNormalId(face.Normal))
	end
end

local function getDimension(face: Face)
	if face.IsWedge then
		-- Can't subtract from a wedge face, only extend
		return Vector3.zero
	else
		local dir = Vector3.fromNormalId(face.Normal)
		return Vector3.new(math.abs(dir.X), math.abs(dir.Y), math.abs(dir.Z))
	end
end

function cl0(n)
	return (n > 0) and n or 0
end
function realDistanceFrom(point: Vector3, part: BasePart)
	local p = part.CFrame:PointToObjectSpace(part.Position)
	local hz = part.Size/2
	local sep = Vector3.new(cl0(math.abs(p.x)-hz.x), cl0(math.abs(p.y)-hz.y), cl0(math.abs(p.z)-hz.z))
	return sep.magnitude
end

function getClosestPointTo(part: BasePart, points: {Vector3}): Vector3
	local closestDistance = math.huge
	local closestPoint = nil
	for _, point in points do
		local dist = realDistanceFrom(point, part)
		if dist < closestDistance then
			closestDistance = dist
			closestPoint = point
		end
	end
	return closestPoint
end

function getFurthestPointTo(part: BasePart, points: {Vector3}): Vector3
	local furthestDistance = -math.huge
	local furthestPoint = nil
	for _, point in points do
		local dist = realDistanceFrom(point, part)
		if dist > furthestDistance then
			furthestDistance = dist
			furthestPoint = point
		end
	end
	return furthestPoint
end

local function getBasis(face: Face)
	if face.IsWedge then
		return face.Object.Position, getNormal(face)
	else
		local hsize = face.Object.Size / 2
		local faceDir = Vector3.fromNormalId(face.Normal)
		local faceNormal = face.Object.CFrame:VectorToWorldSpace(faceDir)
		local facePoint = face.Object.CFrame:PointToWorldSpace(faceDir * hsize)
		return facePoint, faceNormal
	end
end

-- Get the point in the list most "out" of the face
function getPositivePointToFace(face, points: {Vector3}): Vector3
	local basePoint, normal = getBasis(face)
	local maxDist = -math.huge
	local maxPoint = nil
	for _, point in points do
		local dist = (point - basePoint):Dot(normal)
		if dist > maxDist then
			maxDist = dist
			maxPoint = point
		end
	end
	return maxPoint
end

function getNegativePointToFace(face, points: {Vector3}): Vector3
	local basePoint, normal = getBasis(face)
	local minDist = math.huge
	local minPoint = nil
	for _, point in pairs(points) do
		local dist = (point - basePoint):Dot(normal)
		if dist < minDist then
			minDist = dist
			minPoint = point
		end
	end
	return minPoint
end

function resizePart(face: Face, delta: number)
	if face.IsWedge then
		-- Create a new part extruding wedge face
		local point, normal = getBasis(face)
		local part = Instance.new("Part")
		copyPartProps(face.Object, part)
		part.CFrame = CFrame.fromMatrix(point + normal * 0.5 * delta, face.Object.CFrame.XVector, normal)
		local size = face.Object.Size
		part.Size = Vector3.new(size.X, delta, math.sqrt(size.Y^2 + size.Z^2))
		part.TopSurface = Enum.SurfaceType.Smooth
		part.BottomSurface = Enum.SurfaceType.Smooth
		part.Parent = face.Object.Parent
		part.Name = face.Object.Name.."_Extended"
	else
		local axis = Vector3.fromNormalId(face.Normal)
		face.Object.Size += Vector3.new(math.abs(axis.X), math.abs(axis.Y), math.abs(axis.Z)) * delta
		face.Object.CFrame *= CFrame.new(axis * (delta / 2))
	end
end

------------------
--IMPLEMENTATION--
------------------

function otherNormalIds(normalId: Enum.NormalId)
	if normalId == Enum.NormalId.Top or normalId == Enum.NormalId.Bottom then
		return Enum.NormalId.Right, Enum.NormalId.Left, Enum.NormalId.Back, Enum.NormalId.Front
	elseif normalId == Enum.NormalId.Right or normalId == Enum.NormalId.Left then
		return Enum.NormalId.Top, Enum.NormalId.Bottom, Enum.NormalId.Back, Enum.NormalId.Front
	elseif normalId == Enum.NormalId.Front or normalId == Enum.NormalId.Back then
		return Enum.NormalId.Top, Enum.NormalId.Bottom, Enum.NormalId.Right, Enum.NormalId.Left
	end
end
function getFaceSize(part: BasePart, normalId: Enum.NormalId)
	local size = part.Size
	local vec = Vector3.fromNormalId(normalId)
	local x = (1 - math.abs(vec.X)) * size.X
	local y = (1 - math.abs(vec.Y)) * size.Y
	local z = (1 - math.abs(vec.Z)) * size.Z
	return (if x == 0 then 1 else x) * (if y == 0 then 1 else y) * (if z == 0 then 1 else z)
end
function getFacePolygon(part: BasePart, normalId: Enum.NormalId)
	local halfSize = 0.5 * part.Size
	local cf = part.CFrame
	local basePosition = cf:PointToWorldSpace(Vector3.fromNormalId(normalId) * halfSize)
	--
	local x, negx, y, negy = otherNormalIds(normalId)
	local offset_x = cf:VectorToWorldSpace(Vector3.fromNormalId(x) * halfSize)
	local offset_y = cf:VectorToWorldSpace(Vector3.fromNormalId(y) * halfSize)
	--
	return {
		{
			a = basePosition + offset_x + offset_y,
			b = basePosition + offset_x - offset_y,
			c = basePosition - offset_x + offset_y,
			d = basePosition - offset_x - offset_y,
			n = x,
		},
		{
			a = basePosition - offset_x + offset_y,
			b = basePosition - offset_x - offset_y,
			c = basePosition + offset_x + offset_y,
			d = basePosition + offset_x - offset_y,
			n = negx,
		},
		{
			a = basePosition + offset_y + offset_x,
			b = basePosition + offset_y - offset_x,
			c = basePosition - offset_y + offset_x,
			d = basePosition - offset_y - offset_x,
			n = y,
		},
		{
			a = basePosition - offset_y + offset_x,
			b = basePosition - offset_y - offset_x,
			c = basePosition + offset_y + offset_x,
			d = basePosition + offset_y - offset_x,
			n = negy,
		},
	}
end

--[[
	The return value `t` is a number such that `r1o + t * r1d` is the point of
	closest approach on the first ray between the two rays specified by the
	arguments.
]]
function intersectRayRay(r1o, r1d, r2o, r2d)
	local n =
		(r2o - r1o):Dot(r1d) * r2d:Dot(r2d) +
		(r1o - r2o):Dot(r2d) * r1d:Dot(r2d)
	local d =
		r1d:Dot(r1d) * r2d:Dot(r2d) -
		r1d:Dot(r2d) * r1d:Dot(r2d)
	if d == 0 then
		return false
	else
		return true, n / d
	end
end

local function isCylinder(part: BasePart)
	return part:IsA("Part") and part.Shape == Enum.PartType.Cylinder
end


local function fillJoint(faceA: Face, faceB: Face, fillPoint, fillAxis: Vector3, pointsA, pointsB, offsetA, offsetB)
	local maxProj = -math.huge
	local minProj = math.huge
	local maxRadius = -math.huge
	for _, point in pointsA do
		local modPoint = point + offsetA
		local proj = (modPoint - fillPoint):Dot(fillAxis)
		maxProj = math.max(maxProj, proj)
		minProj = math.min(minProj, proj)
		
		-- Take the radius from the first face selected
		local toAxis = (modPoint - (fillPoint + fillAxis * proj)).Magnitude
		maxRadius = math.max(maxRadius, toAxis)
	end
	for _, point in pointsB do
		local modPoint = point + offsetB
		local proj = (modPoint - fillPoint):Dot(fillAxis)
		maxProj = math.max(maxProj, proj)
		minProj = math.min(minProj, proj)
	end
	--
	local centerPoint = fillPoint + fillAxis * (0.5 * (minProj + maxProj))
	local length = (maxProj - minProj)
	local radius = maxRadius
	--
	local cyl = Instance.new("Part")
	if isCylinder(faceA.Object) and isCylinder(faceB.Object) then
		cyl.Shape = Enum.PartType.Ball
	else
		cyl.Shape = Enum.PartType.Cylinder
	end
	cyl.TopSurface = Enum.SurfaceType.Smooth
	cyl.BottomSurface = Enum.SurfaceType.Smooth
	copyPartProps(faceB.Object, cyl)
	cyl.Size = Vector3.new(length, 2 * radius, 2 * radius)
	cyl.CFrame = CFrame.fromMatrix(centerPoint, fillAxis, getNormal(faceB))
	cyl.Parent = faceB.Object.Parent
end

local OUTER_TOUCH = "Outer Touch"
local INNER_TOUCH = "Inner Touch"
local MIDDLE_JOIN = "Rounded Join"
local BUTT_JOINT = "Butt Joint"
local EXTEND_UP_TO = "Extend Up To"
local EXTEND_INTO = "Extend Into"

-- Calculate the result
function doExtend(faceA, faceB, mode)
	local pointsA = getFacePoints(faceA)
	local pointsB = getFacePoints(faceB)
	local localDimensionA = getDimension(faceA)
	local localDimensionB = getDimension(faceB)
	local dirA = getNormal(faceA)
	local dirB = getNormal(faceB)
	
	-- Compare the directions
	local a, b, c = dirA:Dot(dirA), dirA:Dot(dirB), dirB:Dot(dirB)
	local denom = a*c - b*b
	local isParallel = math.abs(denom) < 0.001
	
	-- Find the points to extend out to meet
	local extendPointA, extendPointB;
	if mode.Mode == EXTEND_INTO or mode.Mode == OUTER_TOUCH or mode.Mode == BUTT_JOINT or (isParallel and mode.Mode == MIDDLE_JOIN) then
		extendPointA = getPositivePointToFace(faceB, pointsA)
		extendPointB = getPositivePointToFace(faceA, pointsB)
	elseif mode.Mode == EXTEND_UP_TO or mode.Mode == INNER_TOUCH then
		extendPointA = getNegativePointToFace(faceB, pointsA)
		extendPointB = getNegativePointToFace(faceA, pointsB)
	elseif mode.Mode == MIDDLE_JOIN then
		-- First pick a point and radius based on A
		extendPointA = getBasis(faceA)
		local fillAxis = dirA:Cross(dirB).Unit
		local radiusA = -math.huge
		for _, point in pointsA do
			local projPoint = extendPointA + fillAxis * (point - extendPointA):Dot(fillAxis)
			radiusA = math.max(radiusA, (point - projPoint).Magnitude)
		end
		
		-- Next extend out as far as possible in B but then pull back by the A radius
		local centerPointB = getBasis(faceB)
		extendPointB = getPositivePointToFace(faceA, pointsB)
		local proj = (extendPointB - centerPointB):Dot(fillAxis)
		local projPoint = centerPointB + fillAxis * proj
		local distanceToAxis = (extendPointB - projPoint).Magnitude
		local frac = radiusA / distanceToAxis
		extendPointB = extendPointB:Lerp(projPoint, frac)
	else
		assert(false, "unreachable")
	end
	
	-- Find the closest distance between the rays (extendPointA, dirA) and (extendPointB, dirB):
	-- See: http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment
	local startSep = extendPointB - extendPointA
	local d, e = dirA:Dot(startSep), dirB:Dot(startSep)

	-- Is this a degenerate case?
	if isParallel then
		-- Parts are parallel, extend faceA to faceB
		local lenA = (extendPointA - extendPointB):Dot(getNormal(faceB))
		if faceA.IsWedge then
			if lenA < 0 then
				return
			end
		else
			local extendableA = (localDimensionA * faceA.Object.Size).magnitude
			if getNormal(faceA):Dot(getNormal(faceB)) > 0 then
				lenA = -lenA
			end
			if lenA < -extendableA then
				return
			end
		end
		resizePart(faceA, lenA)
		return
	end

	-- Get the distances to extend by
	local lenA = -(b*e - c*d) / denom
	local lenB = -(a*e - b*d) / denom

	if mode.Mode == EXTEND_INTO or mode.Mode == EXTEND_UP_TO then
		-- We need to find a different lenA, which is the intersection of
		-- extendPointA to the plane faceB:
		-- dist to plane (point, normal) = - (ray_dir . normal) / ((ray_origin - point) . normal)
		local denom2 = dirA:Dot(dirB)
		if math.abs(denom2) > 0.0001 then
			lenA = - (extendPointA - extendPointB):Dot(dirB) / denom2
			lenB = 0
		else
			-- Perpendicular
			-- Project all points of faceB onto faceA and extend by that much
			local points = getPoints(faceB.Object)
			if mode.Mode == EXTEND_UP_TO then
				local smallestLen = math.huge
				for _, v in pairs(points) do
					local dist = (v - extendPointA):Dot(getNormal(faceA))
					if dist < smallestLen then
						smallestLen = dist
					end
				end
				lenA = smallestLen
			elseif mode.Mode == EXTEND_INTO then
				local largestLen = -math.huge
				for _, v in pairs(points) do
					local dist = (v - extendPointA):Dot(getNormal(faceA))
					if dist > largestLen then
						largestLen = dist
					end
				end
				lenA = largestLen
			else
				assert(false, "Unreachable")
			end
			lenB = 0
		end
	end

	-- Are both extents doable?
	-- Note: Negative amounts to extend by *are* allowed, but only
	-- up to the size of the part on the dimension being extended on.
	local extendableA = (localDimensionA * faceA.Object.Size).magnitude
	local extendableB = (localDimensionB * faceB.Object.Size).magnitude
	if lenA < -extendableA then
		return
	end
	if lenB < -extendableB then
		return
	end

	-- Both are doable, execute:
	resizePart(faceA, lenA)
	resizePart(faceB, lenB)
	
	-- Handle filling a joint
	if mode.Mode == MIDDLE_JOIN then
		local fillAxis = dirA:Cross(dirB).Unit
		fillJoint(faceA, faceB, extendPointA + dirA * lenA, fillAxis, pointsA, pointsB, dirA * lenA, dirB * lenB)
	end

	-- For a butt joint, we want to resize back one of the parts by the thickness 
	-- of the other part on that axis. Renize the first part (A), such that it
	-- "butts up against" the second part (B).
	if mode.Mode == BUTT_JOINT then
		-- Find the width of B on the axis A, which is the amount to resize by
		local points = getFacePoints(faceB)
		local minV =  math.huge
		local maxV = -math.huge
		for _, v in pairs(points) do
			local proj = (v - extendPointA):Dot(dirA)
			if proj < minV then minV = proj end
			if proj > maxV then maxV = proj end
		end
		resizePart(faceA, -(maxV - minV))
	end
end

return doExtend