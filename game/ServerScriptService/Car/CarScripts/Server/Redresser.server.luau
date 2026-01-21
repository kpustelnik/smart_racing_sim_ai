--!strict
-- FIN
-- Redresser Server Script by Roblox (Adjusted)

local Workspace: Workspace = game:GetService('Workspace')
if not script:IsDescendantOf(Workspace) then return end -- Check if the car was already spawned

local car: Folder = script.Parent.Parent.Parent
local chassis: BasePart = car:WaitForChild('Chassis') :: BasePart
local redress: Folder = car:WaitForChild('Redress') :: Folder
local redressOrientation: AlignOrientation = redress:WaitForChild('RedressOrientation') :: AlignOrientation

local FLIPPED_LIMIT: number = -0.5
local UPRIGHT_LIMIT: number = 0.7
local MOVING_LIMIT: number = 10

local redressParameters: { [string]: number } = {
	maxTimeFlipped = 2,
	redressMaxTorque = 500000,
	redressResponsiveness = 10,
}

local timeFlipped: number = 0

local function isFlipped(): boolean
	local chassisUp: Vector3 = chassis.CFrame.UpVector
	return chassisUp:Dot(Vector3.yAxis) < FLIPPED_LIMIT
end

local function isUpright(): boolean
	local chassisUp: Vector3 = chassis.CFrame.UpVector
	return chassisUp:Dot(Vector3.yAxis) > UPRIGHT_LIMIT
end

local function isMoving(): boolean
	return chassis.AssemblyLinearVelocity.Magnitude > MOVING_LIMIT
end

-- Enable the RedressOrientation constraint until the car has flipped upright
local function redressAsync(): ()
	local forward: Vector3 = chassis.CFrame.LookVector
	local cframe = CFrame.lookAt(Vector3.zero, forward, Vector3.yAxis)
	redressOrientation.CFrame = cframe
	redressOrientation.Enabled = true
	repeat task.wait() until isUpright()
	redressOrientation.Enabled = false
end

local function initialize(): ()
	-- Cache the attribute values in a table and update them when the attribute changes.
	-- Parameters aren't going to change very often so there's no reason to call :GetAttribute() constantly.
	for parameter: string in redressParameters do
		redress:GetAttributeChangedSignal(parameter):Connect(function(): ()
			local value: unknown? = redress:GetAttribute(parameter)
			if typeof(value) == 'number' then
				redressParameters[parameter] = value :: number
			end
		end)
		local cvalue: unknown? = redress:GetAttribute(parameter)
		if typeof(cvalue) == 'number' then redressParameters[parameter] = cvalue :: number end
	end

	task.spawn(function(): ()
		while true do
			local deltaTime: number = task.wait(0.1)
			-- If the car is currently flipped, keep track of the amount of time it has been flipped
			if isFlipped() and not isMoving() then
				timeFlipped += deltaTime
			else
				timeFlipped = 0
			end
			-- Once the time flipped exceeds the limit, flip the car back over
			if timeFlipped >= redressParameters.maxTimeFlipped then
				timeFlipped = 0
				redressAsync()
			end
		end
	end)
end

initialize()
