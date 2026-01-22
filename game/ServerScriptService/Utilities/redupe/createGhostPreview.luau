--!strict
--!native
local ReplicatedFirst: ReplicatedFirst = game:GetService("ReplicatedFirst")
local DraggerService: DraggerService = game:GetService("DraggerService")
local Packages: Folder = ReplicatedFirst:WaitForChild("Packages")
local DraggerFramework: Instance = require(Packages:WaitForChild("draggerframework")) :: Instance
local JointMaker = (require :: any)(DraggerFramework:WaitForChild("Utility"):WaitForChild("JointMaker"))

export type GhostPreview = {
	hide: () -> (),
	create: (isPreview: boolean, positionOffset: CFrame, sizeOffset: Vector3, extrudeAxis: Vector3) -> { Instance },
	trim: () -> (),
}

type OriginalSizeInfo = {
	Size: Vector3,
	CFrame: CFrame,
	OriginalMeshScale: Vector3?,
	DataModelMesh: DataModelMesh?,
}
type OriginalSizeMap = { [Instance]: OriginalSizeInfo }

type WithEnabled = {
	Enabled: boolean,
}

type PoolItem = {
	instances: { Instance },
	jointsToReenable: { WithEnabled },
	originalSizeInfo: OriginalSizeMap,
}

local GHOST_TRANSPARENCY: number = 0.5

local function cloneTargets(targets: { Instance }): { Instance }
	local clones: { Instance } = {}
	for i: number, target: Instance in targets do clones[i] = target:Clone() end
	return clones
end

local function largestAxis(v: Vector3): Vector3
	local abs: Vector3 = v:Abs()
	if abs.X >= abs.Y and abs.X >= abs.Z then
		return Vector3.xAxis
	elseif abs.Y >= abs.Z then
		return Vector3.yAxis
	else
		return Vector3.zAxis
	end
end

local function enableJointsAndJoinToWorld(item: PoolItem): ()
	for _, joint: WithEnabled in item.jointsToReenable do
		joint.Enabled = true
	end

	if DraggerService ~= nil and DraggerService.JointsEnabled then
		local parts: { BasePart } = {}
		for _, instance in item.instances do
			for _, part: BasePart in (instance:QueryDescendants("BasePart") :: any) :: {BasePart} do
				table.insert(parts, part)
			end
		end
		local jointMaker = JointMaker.new(false)
		jointMaker:pickUpParts(parts)
		local jointPairs = jointMaker:computeJointPairs()
		jointPairs:createJoints()
	end
end

-- Returns the amount of size on the axis of interest in basis, and the "rest" of the size
local function sizeToWorldSpace(basis: CFrame, axisOfInterest: Vector3, size: Vector3): (number, Vector3)
	-- Take the amount of size on the axis, and leave the rest in rest
	local xFactor: number = basis.XVector:Dot(axisOfInterest)
	local sizeOnX: number = math.abs(xFactor) * size.X
	local sizeOffX: number = (1 - math.abs(xFactor)) * size.X

	local yFactor: number = basis.YVector:Dot(axisOfInterest)
	local sizeOnY: number = math.abs(yFactor) * size.Y
	local sizeOffY: number = (1 - math.abs(yFactor)) * size.Y

	local zFactor: number = basis.ZVector:Dot(axisOfInterest)
	local sizeOnZ: number = math.abs(zFactor) * size.Z
	local sizeOffZ: number = (1 - math.abs(zFactor)) * size.Z

	local onAxis: number = sizeOnX + sizeOnY + sizeOnZ
	local rest: Vector3 = Vector3.new(sizeOffX, sizeOffY, sizeOffZ)
	return onAxis, rest
end

local function sizeToObjectSpace(basis: CFrame, axisOfInterest: Vector3, sizeOnAxis: number): Vector3
	local xFactor: number = basis.XVector:Dot(axisOfInterest)
	local yFactor: number = basis.YVector:Dot(axisOfInterest)
	local zFactor: number = basis.ZVector:Dot(axisOfInterest)

	local totalFactor: number = math.abs(xFactor) + math.abs(yFactor) + math.abs(zFactor)
	if totalFactor == 0 then return Vector3.zero end

	local xPortion: number = (math.abs(xFactor) / totalFactor)
	local yPortion: number = (math.abs(yFactor) / totalFactor)
	local zPortion: number = (math.abs(zFactor) / totalFactor)

	return Vector3.new(xPortion, yPortion, zPortion) * sizeOnAxis
end

local EXACTLY_CENTERED_DISAMBIGUATION = 0.01
local function extrudeModel(model: Model, basis: CFrame, sizeDelta: Vector3, original: OriginalSizeMap, isPreview: boolean): ()
	-- Find joints that need to be adjusted afterwards. We need to record
	local toAdjustC0C1: { [JointInstance]: CFrame } = {}
	for _, joint: JointInstance in (model:QueryDescendants("JointInstance") :: any) :: {JointInstance} do
		local part0: BasePart? = joint.Part0
		local part1: BasePart? = joint.Part1
		if part0 and part1 then
			local relativeTo: CFrame = part0.CFrame:Lerp(part1.CFrame, 0.5)
			toAdjustC0C1[joint] = relativeTo:ToObjectSpace(part0.CFrame * joint.C0)
		end
	end

	-- Special case for returning the model to how it was if there's no size delta, because
	-- we don't have an axis to extrude on in that case.
	if sizeDelta:FuzzyEq(Vector3.zero) then
		for _, part: BasePart in (model:QueryDescendants("BasePart") :: any) :: {BasePart} do
			local originalInfo: OriginalSizeInfo? = original[part]
			if originalInfo then
				part.Size = originalInfo.Size
				part.CFrame = originalInfo.CFrame
				if originalInfo.DataModelMesh and originalInfo.OriginalMeshScale then
					originalInfo.DataModelMesh.Scale = originalInfo.OriginalMeshScale
				end
			end
		end
	else
		-- Extrude on the axis with the largest delta
		local extrudeAxis: Vector3 = largestAxis(sizeDelta)
		local extrudeAmount: number = sizeDelta:Dot(extrudeAxis)

		-- We only want basis to represent a rotation since we're pivoting the model to the origin
		-- to have a stable basis for scaling.
		basis = basis.Rotation
		for _, part: BasePart in (model:QueryDescendants("BasePart") :: any) :: {BasePart} do
			-- Info to scale from
			local originalInfo: OriginalSizeInfo = original[part]
			if not originalInfo then
				local mesh: DataModelMesh? = part:FindFirstChildWhichIsA("DataModelMesh")
				local newInfo: OriginalSizeInfo = table.freeze({
					Size = part.Size,
					CFrame = part.CFrame,
					DataModelMesh = mesh,
					OriginalMeshScale = if mesh then mesh.Scale else nil,
				})
				original[part] = newInfo
				originalInfo = newInfo
			end

			-- Since the model is at origin, the raw part CFrame is already in local space
			local localCFrame: CFrame = originalInfo.CFrame
			local localSize: number, restOfSize: Vector3 = sizeToWorldSpace(originalInfo.CFrame, extrudeAxis, originalInfo.Size)

			local positionOnAxis: number = localCFrame.Position:Dot(extrudeAxis)
			local halfSize: number = 0.5 * localSize
			local a: number = positionOnAxis + halfSize + EXACTLY_CENTERED_DISAMBIGUATION
			local b: number = positionOnAxis - halfSize + EXACTLY_CENTERED_DISAMBIGUATION
			if a * b < 0 then
				-- Spans zero, extrude
				local newSizeOnAxis: number = localSize + extrudeAmount
				if newSizeOnAxis > 0 then
					local newSize: Vector3 = sizeToObjectSpace(originalInfo.CFrame, extrudeAxis, newSizeOnAxis) + restOfSize
					part.Size = newSize
					if originalInfo.DataModelMesh and originalInfo.OriginalMeshScale then
						originalInfo.DataModelMesh.Scale = (newSize / originalInfo.Size) * originalInfo.OriginalMeshScale
					end
					-- Make sure it's shown if it was hidden previously
					part.LocalTransparencyModifier = 0
				else
					-- Collasped to zero, hide or remove it
					if isPreview then
						part.LocalTransparencyModifier = 1
					else
						part.Parent = nil
					end
				end
			else
				local sign: number = math.sign(localCFrame.Position:Dot(extrudeAxis))
				local motion: Vector3 = extrudeAxis * extrudeAmount * 0.5 * sign
				part.CFrame = localCFrame + motion
			end
		end
	end

	-- Adjust joints with a C0/C1
	for joint: JointInstance, localCenter: CFrame in toAdjustC0C1 do
		local part0: BasePart? = joint.Part0
		local part1: BasePart? = joint.Part1
		if part0 and part1 then
			local relativeTo: CFrame = part0.CFrame:Lerp(part1.CFrame, 0.5)
			local worldCenter: CFrame = relativeTo * localCenter
			joint.C0 = (part0.CFrame:Inverse() * worldCenter):Orthonormalize()
			joint.C1 = (part1.CFrame:Inverse() * worldCenter):Orthonormalize()
		end
	end
end

local function disableJoints(hierarchy: Instance, jointsToReenable: { WithEnabled }): ()
	for _, joint in (hierarchy:QueryDescendants("JointInstance") :: any) :: {JointInstance} do
		if joint.Enabled then
			joint.Enabled = false
			table.insert(jointsToReenable, (joint :: any) :: WithEnabled)
		end
		if joint.Part0 and not joint.Part0:IsDescendantOf(hierarchy) then joint:Destroy() end
		if joint.Part1 and not joint.Part1:IsDescendantOf(hierarchy) then joint:Destroy() end
	end
	for _, weldConstraint in (hierarchy:QueryDescendants("WeldConstraint") :: any) :: {WeldConstraint} do
		if weldConstraint.Enabled then
			weldConstraint.Enabled = false
			table.insert(jointsToReenable, (weldConstraint :: any) :: WithEnabled)
		end
		if weldConstraint.Part0 and not weldConstraint.Part0:IsDescendantOf(hierarchy) then
			weldConstraint:Destroy()
		end
		if weldConstraint.Part1 and not weldConstraint.Part1:IsDescendantOf(hierarchy) then
			weldConstraint:Destroy()
		end
	end
end

local function createGhostPreview(targets: { Instance }, cframe: CFrame, size: Vector3): GhostPreview
	local pool: { PoolItem } = {}
	local placed: { PoolItem } = {}
	local toTrim: number? = nil

	-- Clone the targets to get a known structure that will match
	-- during subsequent clones. (The first clone may lose some
	-- things like Archivable = false Instances)
	local clonedTargets: { Instance } = cloneTargets(targets)

	local function getItem(isPreview: boolean): PoolItem
		if isPreview and #pool > 0 then
			local existingItem: PoolItem? = table.remove(pool)
			assert(existingItem, "Pool was not empty")
			-- Not cloned targets here because we want the
			-- original parent to parent to.
			for i: number, target: Instance in targets do
				existingItem.instances[i].Parent = target.Parent
			end
			return existingItem
		end

		local newItem: PoolItem = {
			instances = {},
			jointsToReenable = {},
			originalSizeInfo = {},
		}
		for i: number, clonedTarget: Instance in clonedTargets do
			local copy: Instance = clonedTarget:Clone()
			if isPreview then
				copy.Archivable = false
				if copy:IsA("Model") or copy:IsA("Folder") then
					for _, descendant: Instance in copy:GetDescendants() do
						if descendant:IsA("BasePart") and descendant.Transparency < 1 then
							descendant.Transparency = GHOST_TRANSPARENCY
						end
					end
				elseif copy:IsA("BasePart") then
					copy.Transparency = GHOST_TRANSPARENCY
				else
					warn(`Unsupported target type {copy.ClassName} for ghost preview`)
				end
			end
			disableJoints(copy, newItem.jointsToReenable)
			copy.Parent = targets[i].Parent
			newItem.instances[i] = copy
		end
		return newItem
	end

	-- Deparent the unused item in the pool
	local function trim(): ()
		if toTrim then
			for i = toTrim, #pool do
				local item: PoolItem = pool[i]
				for _, instance: Instance in item.instances do instance.Parent = nil end
			end
			toTrim = nil
		end
	end

	local function hide(): ()
		trim()
		toTrim = #pool + 1
		for _, item: PoolItem in placed do table.insert(pool, item) end
		table.clear(placed)
	end

	local function adjustSingleInstance(item: Instance, target: Instance, extrudeAxis: Vector3, original: OriginalSizeMap, targetCFrame: CFrame, targetSize: Vector3, isPreview: boolean): boolean
		local scale: Vector3 = targetSize / size
		if target:IsA("Model") then
			local itemModel: Model = item :: Model
			local targetModel: Model = target :: Model
			local pivotInBasis: CFrame = cframe:ToObjectSpace(targetModel:GetPivot())
			itemModel:PivotTo(CFrame.identity)
			extrudeModel(itemModel, cframe, (targetSize - size), original, isPreview)
			itemModel:PivotTo(targetCFrame * pivotInBasis)
			return true
		elseif target:IsA("BasePart") then
			local itemPart: BasePart = item :: BasePart
			local targetPart: BasePart = target :: BasePart
			local offsetFromBase: CFrame = cframe:ToObjectSpace(targetPart:GetPivot())
			itemPart:PivotTo(targetCFrame * offsetFromBase)
			itemPart.Size = scale * targetPart.Size
			return true
		else
			return false
		end
	end

	local function adjustItemRecursive(item: Instance, target: Instance, extrudeAxis: Vector3, original: OriginalSizeMap, targetCFrame: CFrame, targetSize: Vector3, isPreview: boolean): ()
		local didAdjust: boolean = adjustSingleInstance(item, target, extrudeAxis, original, targetCFrame, targetSize, isPreview)
		if not didAdjust then
			-- Recurse to children
			local itemChildren: { Instance } = item:GetChildren()
			local targetChildren: { Instance } = target:GetChildren()

			-- This invariant works because we clone the targets one additional
			-- time upfront. So every clone is a "clone of a clone" which should
			-- be idempotent in structure.
			assert(#itemChildren == #targetChildren, "Mismatched children count in ghost preview adjustment")

			for i = 1, #itemChildren do
				adjustItemRecursive(itemChildren[i], targetChildren[i], extrudeAxis, original, targetCFrame, targetSize, isPreview)
			end
		end
	end

	local function create(isPreview: boolean, targetCFrame: CFrame, targetSize: Vector3, extrudeAxis: Vector3): { Instance }
		local itemToSpawn: PoolItem = getItem(isPreview)
		for i: number, clonedTarget: Instance in clonedTargets do
			local item: Instance = itemToSpawn.instances[i]
			adjustItemRecursive(item, clonedTarget, extrudeAxis, itemToSpawn.originalSizeInfo, targetCFrame, targetSize, isPreview)
		end
		-- Non-preview is permanent, don't need them in the placed list
		if isPreview then
			table.insert(placed, itemToSpawn)
		else
			enableJointsAndJoinToWorld(itemToSpawn)
		end
		return itemToSpawn.instances
	end

	return {
		hide = hide,
		create = create,
		trim = trim,
	}
end

return createGhostPreview