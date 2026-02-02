--!strict

export type Placement = {
	CFrame: CFrame,
	BoundsOffset: Vector3,
	Size: Vector3,
	PreviousSize: Vector3,
	Offset: CFrame,
}

local function approxSign(n: number): number
	if math.abs(n) < 0.001 then return 0
  else return math.sign(n)
	end
end

local function fixZeroSize(v: Vector3): Vector3
	return Vector3.new(
		(v.X > 0.01) and v.X or 1,
		(v.Y > 0.01) and v.Y or 1,
		(v.Z > 0.01) and v.Z or 1
	)
end

local function bendPlacement(placement: Placement, axis: Vector3, relativeBend: CFrame, touchSide: number, paddingAmount: number, spacingMultiplier: number): ()
	local relativeOffset: CFrame = placement.Offset
	local referenceSize: Vector3 = placement.PreviousSize
	local paraSize: number = referenceSize:Dot(axis)
	local perpSize: Vector3 = referenceSize - axis * paraSize

	local forwardAxis: Vector3 = if axis:Dot(relativeOffset.Position) > 0 then axis else -axis
	local xDir: number = relativeBend.XVector:Dot(forwardAxis)
	local yDir: number = relativeBend.YVector:Dot(forwardAxis)
	local zDir: number = relativeBend.ZVector:Dot(forwardAxis)
	local directions: Vector3 = Vector3.new(approxSign(xDir), approxSign(yDir), approxSign(zDir))

	-- Apply padding and spacing before using the reference size
	-- (bit messy, I shouldn't really be scaling the perp components here
	-- but they don't get used in the subsequent code)
	referenceSize *= spacingMultiplier
	referenceSize += axis * paddingAmount
	referenceSize = fixZeroSize(referenceSize) -- Ensure the settings didn't result in zero size

	local perpOffset: CFrame = CFrame.new(perpSize * directions * touchSide * 0.5)
	local paraOffset: CFrame = CFrame.new(forwardAxis * referenceSize * 0.5)
	local offsetA: CFrame = perpOffset * paraOffset
	local offsetB: CFrame = offsetA:Inverse() * relativeOffset

	local boundsOffset: CFrame = CFrame.new(placement.BoundsOffset)
	placement.Offset = offsetA * boundsOffset * relativeBend * offsetB * boundsOffset:Inverse()
end

return bendPlacement