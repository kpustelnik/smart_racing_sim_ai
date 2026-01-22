--!strict

local Workspace: Workspace = game:GetService('Workspace')

-- Does it make sense to wedge fill a given part when extruding? It only makes sense if the part extends
-- out to all 6 corners of its bounding box.
local WEDGE_FILLABLE_CLOSENESS_TO_CORNER: number = 0.02
local corners: { Vector3 } = {
	Vector3.new(1, 1, 1),
	Vector3.new(1, 1, -1),
	Vector3.new(1, -1, 1),
	Vector3.new(1, -1, -1),
	Vector3.new(-1, 1, 1),
	Vector3.new(-1, 1, -1),
	Vector3.new(-1, -1, 1),
	Vector3.new(-1, -1, -1),
}

local WEDGE_FILLABLE_TEST_PART: Part?;
local function getTestPart(position: Vector3): Part
  if not WEDGE_FILLABLE_TEST_PART then
    local cp: Part = Instance.new("Part")
    cp = Instance.new("Part")
    cp.Size = Vector3.one * 0.01
    WEDGE_FILLABLE_TEST_PART = cp
  end
  local cp: Part? = WEDGE_FILLABLE_TEST_PART
  if not cp then error("Unreachable") end
  cp.Position = position
  return cp
end

local function isWedgeFillable(a: BasePart): boolean
	for _, corner: Vector3 in corners do
		local offset: Vector3 = corner * (a.Size * 0.5 - Vector3.one * WEDGE_FILLABLE_CLOSENESS_TO_CORNER)
		local worldPoint: Vector3 = a.CFrame:PointToWorldSpace(offset)
		local params: OverlapParams = OverlapParams.new()
		params.FilterType = Enum.RaycastFilterType.Include
		params:AddToFilter(a)
		if #Workspace:GetPartsInPart(getTestPart(worldPoint), params) == 0 then return false end
	end
	return true
end

return isWedgeFillable