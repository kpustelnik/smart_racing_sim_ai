--!strict
--!native
--[[
Copyright 2025 Mark Langen (@stravant)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
]]
local ReplicatedFirst: ReplicatedFirst = game:GetService("ReplicatedFirst")
local DraggerService: DraggerService? = game:GetService("DraggerService")
local Packages: Folder = ReplicatedFirst:WaitForChild("Packages")
local DraggerFramework: Instance = require(Packages:WaitForChild("draggerframework")) :: Instance
local JointMaker = (require :: any)(DraggerFramework:WaitForChild("Utility"):WaitForChild("JointMaker"))

export type GhostPreview = {
	hide: () -> (),
	create: (isPreview: boolean, scaleMode: string, positionOffset: CFrame, sizeOffset: Vector3, extrudeAxis: Vector3) -> { Instance },
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
	local xFactor: number = math.abs(basis.XVector:Dot(axisOfInterest))
	local sizeOnX: number = xFactor * size.X
	local sizeOffX: number = (1 - xFactor) * size.X

	local yFactor: number = math.abs(basis.YVector:Dot(axisOfInterest))
	local sizeOnY: number = yFactor * size.Y
	local sizeOffY: number = (1 - yFactor) * size.Y

	local zFactor: number = math.abs(basis.ZVector:Dot(axisOfInterest))
	local sizeOnZ: number = zFactor * size.Z
	local sizeOffZ: number = (1 - zFactor) * size.Z

	local onAxis: number = sizeOnX + sizeOnY + sizeOnZ
	local rest: Vector3 = Vector3.new(sizeOffX, sizeOffY, sizeOffZ)
	return onAxis, rest
end

local function sizeToObjectSpace(basis: CFrame, axisOfInterest: Vector3, size: Vector3, sizeOnAxis: number): Vector3
	local xFactor: number = math.abs(basis.XVector:Dot(axisOfInterest)) * size.X
	local yFactor: number = math.abs(basis.YVector:Dot(axisOfInterest)) * size.Y
	local zFactor: number = math.abs(basis.ZVector:Dot(axisOfInterest)) * size.Z

	local totalFactor: number = xFactor + yFactor + zFactor
	if totalFactor == 0 then return Vector3.zero end

	local xPortion: number = xFactor / totalFactor
	local yPortion: number = yFactor / totalFactor
	local zPortion: number = zFactor / totalFactor

	return Vector3.new(xPortion, yPortion, zPortion) * sizeOnAxis
end

local function getOriginalInfo(part: BasePart, original: OriginalSizeMap): OriginalSizeInfo
	local originalInfo: OriginalSizeInfo? = original[part]
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
	return originalInfo :: OriginalSizeInfo
end

local function resetToOriginalInfo(part: BasePart, original: OriginalSizeMap): ()
	local originalInfo: OriginalSizeInfo? = original[part]
	if originalInfo then
		part.Size = originalInfo.Size
		part.CFrame = originalInfo.CFrame
		part.LocalTransparencyModifier = 0
		if originalInfo.DataModelMesh and originalInfo.OriginalMeshScale then
			originalInfo.DataModelMesh.Scale = originalInfo.OriginalMeshScale
		end
	end
end

local EXACTLY_CENTERED_DISAMBIGUATION = 0.01
local function extrudePart(part: BasePart, originalInfo: OriginalSizeInfo, extrudeAxis: Vector3, extrudeAmount: number, isPreview: boolean): (CFrame, Vector3)
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
			local newSize: Vector3 = sizeToObjectSpace(originalInfo.CFrame, extrudeAxis, originalInfo.Size, newSizeOnAxis) + restOfSize
			if originalInfo.DataModelMesh and originalInfo.OriginalMeshScale then
				originalInfo.DataModelMesh.Scale = (newSize / originalInfo.Size) * originalInfo.OriginalMeshScale
			end
			-- Make sure it is shown if it was hidden previously
			part.LocalTransparencyModifier = 0
			return originalInfo.CFrame, newSize
		else
			-- Collasped to zero, hide or remove it
			if isPreview then
				part.LocalTransparencyModifier = 1
			else
				part.Parent = nil
			end
			return originalInfo.CFrame, originalInfo.Size
		end
	else
		local sign: number = math.sign(localCFrame.Position:Dot(extrudeAxis))
		local motion: Vector3 = extrudeAxis * extrudeAmount * 0.5 * sign
		local newCFrame: CFrame = localCFrame + motion
		part.LocalTransparencyModifier = 0
		return newCFrame, originalInfo.Size
	end
end

local function resizeWithWorldScale(basis: CFrame, size: Vector3, worldScale: Vector3): Vector3
	local xDir: Vector3 = basis.XVector:Abs()
	local yDir: Vector3 = basis.YVector:Abs()
	local zDir: Vector3 = basis.ZVector:Abs()
	return size * Vector3.new(
		(xDir * worldScale).Magnitude,
		(yDir * worldScale).Magnitude,
		(zDir * worldScale).Magnitude
	)
end

local function resizeModelViaExtrude(model: Model, scaleFactor: Vector3, sizeDelta: Vector3, original: OriginalSizeMap, isPreview: boolean): ()
	-- Special case for returning the model to how it was if there is no size delta, because
	-- we do not have an axis to extrude on in that case.
	if sizeDelta:FuzzyEq(Vector3.zero) then
		for _, part: BasePart in (model:QueryDescendants("BasePart") :: any) :: {BasePart} do
			resetToOriginalInfo(part, original)
		end
	else
		-- Extrude on the axis with the largest delta
		local extrudeAxis: Vector3 = largestAxis(sizeDelta)
		local otherAxis: Vector3 = Vector3.one - extrudeAxis
		local scaleForOtherAxis: Vector3 = otherAxis * scaleFactor + extrudeAxis
		local extrudeAmount: number = sizeDelta:Dot(extrudeAxis)

		for _, part: BasePart in (model:QueryDescendants("BasePart") :: any) :: {BasePart} do
			-- Info to scale from
			local originalInfo: OriginalSizeInfo = getOriginalInfo(part, original)

			-- Extrude on extrude axis
			local cframe: CFrame, size: Vector3 = extrudePart(part, originalInfo, extrudeAxis, extrudeAmount, isPreview)

			-- Stretch on other axis
			size = resizeWithWorldScale(cframe, size, scaleForOtherAxis)
			local position: Vector3 = cframe.Position
			cframe = cframe.Rotation + position * scaleForOtherAxis

			-- Apply
			part.Size = size
			part.CFrame = cframe
		end
	end
end

local function resizeModelViaStretch(model: Model, scaleFactor: Vector3, original: OriginalSizeMap, _isPreview: boolean): ()
	-- Special case for returning the model to how it was if there is no size delta, because
	-- we do not have an axis to extrude on in that case.
	if scaleFactor:FuzzyEq(Vector3.one) then
		for _, part: BasePart in (model:QueryDescendants("BasePart") :: any) :: {BasePart} do
			resetToOriginalInfo(part, original)
		end
	else
		for _, part: BasePart in (model:QueryDescendants("BasePart") :: any) :: {BasePart} do
			-- Info to scale from
			local originalInfo: OriginalSizeInfo = getOriginalInfo(part, original)

			-- Since the model is at origin, the raw part CFrame is already in local space
			local localCFrame: CFrame = originalInfo.CFrame
			local localSize: Vector3 = originalInfo.Size
			local newSize: Vector3 = resizeWithWorldScale(originalInfo.CFrame, localSize, scaleFactor)
			part.Size = newSize
			part.CFrame = localCFrame.Rotation + localCFrame.Position * scaleFactor
			if originalInfo.DataModelMesh and originalInfo.OriginalMeshScale then
				originalInfo.DataModelMesh.Scale = (newSize / originalInfo.Size) * originalInfo.OriginalMeshScale
			end
			-- Make sure it is shown if it was hidden previously (cannot happen in this branch
			-- but can happen if we switched from extrude to stretch)
			part.LocalTransparencyModifier = 0
		end
	end
end

-- Return the most powerful scaling (up or down) being applied on an axis
local function largestScale(scaleFactor: Vector3): number
	local maxScale: number = math.max(scaleFactor.X, scaleFactor.Y, scaleFactor.Z)
	local minScale: number = math.min(scaleFactor.X, scaleFactor.Y, scaleFactor.Z)
	if 1 / minScale > maxScale then
		return minScale
	else
		return maxScale
	end
end

local function resizeModelViaScaleTo(model: Model, scaleFactor: Vector3, original: OriginalSizeMap): ()
	-- Restore parts before we do anything if we previously did an edit to the model
	if next(original) then
		for _, part: BasePart in (model:QueryDescendants("BasePart") :: any) :: {BasePart} do
			resetToOriginalInfo(part, original)
		end
		-- Clear so we do not keep restoring unnecessarily
		table.clear(original)
	end

	model:ScaleTo(model:GetScale() * largestScale(scaleFactor))
end

local function scaleModel(model: Model, scaleMode: string, _basis: CFrame, originalSize: Vector3, newSize: Vector3, original: OriginalSizeMap, isPreview: boolean): Vector3
	-- Find joints that need to be adjusted afterwards.
	local toAdjustC0C1: { [JointInstance]: CFrame } = {}
	for _, joint: JointInstance in (model:QueryDescendants("JointInstance") :: any) :: {JointInstance} do
		local part0: BasePart? = joint.Part0
		local part1: BasePart? = joint.Part1
		if part0 and part1 then
			local relativeTo: CFrame = part0.CFrame:Lerp(part1.CFrame, 0.5)
			toAdjustC0C1[joint] = relativeTo:ToObjectSpace(part0.CFrame * joint.C0)
		end
	end

	local scaleFactor: Vector3 = newSize / originalSize
	local appliedScale: Vector3
	if scaleMode == "Extrude" then
		local sizeDelta: Vector3 = newSize - originalSize
		resizeModelViaExtrude(model, scaleFactor, sizeDelta, original, isPreview)
		appliedScale = Vector3.one
	elseif scaleMode == "Stretch" then
		resizeModelViaStretch(model, scaleFactor, original, isPreview)
		appliedScale = scaleFactor
	elseif scaleMode == "Uniform" then
		local initialScale: number = model:GetScale()
		resizeModelViaScaleTo(model, scaleFactor, original)
		appliedScale = Vector3.one * (model:GetScale() / initialScale)
	else
		error(`Unknown scale mode {scaleMode}`)
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

	return appliedScale
end

local function disableJoints(hierarchy: Instance, jointsToReenable: { WithEnabled }): ()
	for _, joint: JointInstance in (hierarchy:QueryDescendants("JointInstance") :: any) :: {JointInstance} do
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

	local function adjustSingleInstance(item: Instance, target: Instance, scaleMode: string, _extrudeAxis: Vector3, original: OriginalSizeMap, targetCFrame: CFrame, targetSize: Vector3, isPreview: boolean): boolean
		local scale: Vector3 = targetSize / size
		if target:IsA("Model") then
			local itemModel: Model = item :: Model
			local targetModel: Model = target :: Model
			local pivotInBasis: CFrame = cframe:ToObjectSpace(targetModel:GetPivot())
			itemModel:PivotTo(CFrame.identity)
			-- Note, this call only adds extra work in ScaleTo mode which will be
			-- uncommon and faster than the other modes to execute so do it for
			-- implementation simplicity. We are going to scale it again after so
			-- work could be saved here by not doing two ScaleTo calls each frame.
			itemModel:ScaleTo(targetModel:GetScale())
			local appliedScale: Vector3 = scaleModel(itemModel, scaleMode, cframe, size, targetSize, original, isPreview)
			itemModel:PivotTo(targetCFrame * (pivotInBasis.Rotation + pivotInBasis.Position * appliedScale))
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

	local function adjustItemRecursive(item: Instance, target: Instance, scaleMode: string, extrudeAxis: Vector3, original: OriginalSizeMap, targetCFrame: CFrame, targetSize: Vector3, isPreview: boolean): ()
		local didAdjust: boolean = adjustSingleInstance(item, target, scaleMode, extrudeAxis, original, targetCFrame, targetSize, isPreview)
		if not didAdjust then
			-- Recurse to children
			local itemChildren: { Instance } = item:GetChildren()
			local targetChildren: { Instance } = target:GetChildren()

			-- This invariant works because we clone the targets one additional
			-- time upfront. So every clone is a "clone of a clone" which should
			-- be idempotent in structure.
			assert(#itemChildren == #targetChildren, "Mismatched children count in ghost preview adjustment")

			for i = 1, #itemChildren do
				adjustItemRecursive(itemChildren[i], targetChildren[i], scaleMode, extrudeAxis, original, targetCFrame, targetSize, isPreview)
			end
		end
	end

	local function create(isPreview: boolean, scaleMode: string, targetCFrame: CFrame, targetSize: Vector3, extrudeAxis: Vector3): { Instance }
		local itemToSpawn: PoolItem = getItem(isPreview)
		for i: number, clonedTarget: Instance in clonedTargets do
			local item: Instance = itemToSpawn.instances[i]
			adjustItemRecursive(item, clonedTarget, scaleMode, extrudeAxis, itemToSpawn.originalSizeInfo, targetCFrame, targetSize, isPreview)
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