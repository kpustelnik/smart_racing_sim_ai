--!strict
-- FIN
-- Handles updating car controls based on input attributes (by Roblox)

local Constants = require(script.Parent:WaitForChild('Constants'))
local Units = require(script.Parent:WaitForChild('Units'))
local function lerp(a: number, b: number, t: number): number return a + (b - a) * t end

local car: Folder = script.Parent.Parent
local chassis: BasePart = car:WaitForChild('Chassis') :: BasePart
local inputs: Configuration = car:WaitForChild('Inputs') :: Configuration
local engine: Folder = car:WaitForChild('Engine') :: Folder
local steering: Folder = car:WaitForChild('Steering') :: Folder
local wheels: Folder = car:WaitForChild('Wheels') :: Folder

local ReadNumberAttribute = function (i: Instance, attrName: string): number?
	local value: unknown? = i:GetAttribute(attrName)
	if value ~= nil and typeof(value) == "number" then
		return value :: number
	end
	return nil
end

local wheelParts: { BasePart } = {
	wheels:WaitForChild('WheelFR'):WaitForChild('Wheel') :: BasePart,
	wheels:WaitForChild('WheelFL'):WaitForChild('Wheel') :: BasePart,
	wheels:WaitForChild('WheelRR'):WaitForChild('Wheel') :: BasePart,
	wheels:WaitForChild('WheelRL'):WaitForChild('Wheel') :: BasePart,
}

local engineParameters: { [string]: number } = {
	forwardMaxSpeed = 0,
	reverseMaxSpeed = 0,
	acceleration = 0,
	deceleration = 0,
	braking = 0,
	engineSpeedCorrection = 0,
	minSpeedTorque = 0,
	maxSpeedTorque = 0,
	handBrakeTorque = 0,
	nitroTorque = 0,
	nitroTime = 0,
	nitroRechargeTime = 0,
	nitroMaxSpeed = 0,
	nitroAcceleration = 100,
}

local steeringParameters: { [string]: number } = {
	steeringReduction = 0,
}

local wheelParameters: { [string]: number } = {
	density = 0, -- Wheel density
	elasticity = 0, -- Wheel elasticity
	staticFriction = 0, -- Wheel friction while it is not sliding
	kineticFriction = 0, -- Wheel friction while it is sliding
	slipThreshold = 0, -- Threshold for the wheel to start sliding
}

-- Calculate wheel radius based on the front right wheel
local wheelRadius: number = wheelParts[1].Size.Y / 2

-- Return the current chassis speed on the local Z axis in miles per hour
local function getChassisForwardSpeed(): number
	local relativeVelocity: Vector3 = chassis.CFrame:VectorToObjectSpace(chassis.AssemblyLinearVelocity)
	local milesPerHour: number = Units.studsPerSecondToMilesPerHour(-relativeVelocity.Z)
	return milesPerHour
end

-- Update the amount of nitro available
local NitroVelocity: LinearVelocity = engine:WaitForChild('NitroVelocity') :: LinearVelocity
local function updateNitro(deltaTime: number): ()
	local nitroInput: unknown? = inputs:GetAttribute(Constants.NITRO_INPUT_ATTRIBUTE)
	local boolNitroInput: boolean = nitroInput ~= nil and typeof(nitroInput) == "boolean" and nitroInput :: boolean
	local engineNitro: number? = ReadNumberAttribute(engine, Constants.ENGINE_NITRO_ATTRIBUTE)
	if engineNitro == nil then return end

	local isNitroEnabled: boolean = boolNitroInput and engineNitro > 0
	-- If the nitro input is enabled, lower the nitro level
	if isNitroEnabled then
		local amount: number = 1 / engineParameters.nitroTime
		engineNitro = math.max(engineNitro - amount * deltaTime, 0)

		-- If the nitro level reaches 0, disable the nitro input
		if engineNitro == 0 then
			inputs:SetAttribute(Constants.NITRO_INPUT_ATTRIBUTE, false)
			nitroInput = false
		end
	else
		-- Recharge the nitro level slowly
		local amount: number = 1 / engineParameters.nitroRechargeTime
		engineNitro = math.min(engineNitro + amount * deltaTime, 1)
	end
	engine:SetAttribute(Constants.ENGINE_NITRO_ATTRIBUTE, engineNitro)
	engine:SetAttribute(Constants.NITRO_ENABLED_ATTRIBUTE, isNitroEnabled)

	-- Update the nitro constraint
	NitroVelocity.Enabled = isNitroEnabled
	NitroVelocity.VectorVelocity = Vector3.new(0, 0, -engineParameters.nitroMaxSpeed)
end

-- Update the car's target speed and wheel motors' torque and angular velocity
local WheelFRMotor: CylindricalConstraint = engine:WaitForChild('WheelFRMotor') :: CylindricalConstraint
local WheelFLMotor: CylindricalConstraint = engine:WaitForChild('WheelFLMotor') :: CylindricalConstraint
local WheelRRMotor: CylindricalConstraint = engine:WaitForChild('WheelRRMotor') :: CylindricalConstraint
local WheelRLMotor: CylindricalConstraint = engine:WaitForChild('WheelRLMotor') :: CylindricalConstraint

local function updateEngine(deltaTime: number): ()
	local handBrakeInput: unknown? = inputs:GetAttribute(Constants.HAND_BRAKE_INPUT_ATTRIBUTE)
	local boolHandBrakeInput: boolean = handBrakeInput ~= nil and typeof(handBrakeInput) == "boolean" and handBrakeInput :: boolean

	local throttleInput: number? = ReadNumberAttribute(inputs, Constants.THROTTLE_INPUT_ATTRIBUTE)
	if throttleInput == nil then return end
	local nThrottleInput: number = math.clamp(throttleInput :: number, -1, 1)
	local throttleDirection: number = math.sign(nThrottleInput)

	local engineSpeed: number? = ReadNumberAttribute(engine, Constants.ENGINE_SPEED_ATTRIBUTE)
	if engineSpeed == nil then return end

	local isNitroEnabled: unknown? = engine:GetAttribute(Constants.NITRO_ENABLED_ATTRIBUTE)
	local boolIsNitroEnabled: boolean = isNitroEnabled ~= nil and typeof(isNitroEnabled) == "boolean" and isNitroEnabled :: boolean

	local maxSpeed: number = math.max(engineParameters.forwardMaxSpeed, engineParameters.reverseMaxSpeed)
	local targetSpeed: number = nThrottleInput * (if throttleDirection >= 0 then engineParameters.forwardMaxSpeed else engineParameters.reverseMaxSpeed)
	local forwardSpeed: number = getChassisForwardSpeed()
	local isBraking: boolean = throttleDirection ~= 0
		and math.abs(forwardSpeed) > 1
		and throttleDirection ~= math.sign(forwardSpeed)
	local acceleration: number = engineParameters.acceleration

	-- Adjust the acceleration based on whether the car is using nitro, braking, or slowing down
	if boolIsNitroEnabled then
		acceleration = engineParameters.nitroAcceleration
		targetSpeed = engineParameters.nitroMaxSpeed
	elseif isBraking then
		-- When the car is switching directions, we apply braking until it comes to a stop.
		-- This is essentially engine braking, and accelerates the engine speed to 0.
		acceleration = engineParameters.braking
		targetSpeed = 0
	elseif math.abs(targetSpeed) < math.abs(engineSpeed) then
		-- When the car is slowing down in either direction, we use a separate deceleration
		-- parameter to slow the car down slowly, rather than using the higher acceleration value.
		-- This allows the car to glide to a stop when releasing the throttle.
		acceleration = engineParameters.deceleration
	end

	-- Proportionally adjust the engine speed towards the car's actual speed. This prevents the
	-- wheels from continuing to spin after e.g. the car runs into a wall and stops suddenly.
	engineSpeed = lerp(engineSpeed, forwardSpeed, math.min(deltaTime * engineParameters.engineSpeedCorrection, 1))
	if targetSpeed > engineSpeed then
		engineSpeed = math.min(engineSpeed + acceleration * deltaTime, targetSpeed)
	else
		engineSpeed = math.max(engineSpeed - acceleration * deltaTime, targetSpeed)
	end
	engine:SetAttribute(Constants.ENGINE_SPEED_ATTRIBUTE, engineSpeed)

	-- Calculate the torque and angular velocity required to reach the target linear velocity
	local chassisSpeed: number = getChassisForwardSpeed()
	local torqueAlpha: number = math.min(chassisSpeed, maxSpeed) / maxSpeed
	local torque: number = lerp(engineParameters.minSpeedTorque, engineParameters.maxSpeedTorque, torqueAlpha)
	if boolIsNitroEnabled then torque = engineParameters.nitroTorque end
	local studsSpeed: number = Units.milesPerHourToStudsPerSecond(engineSpeed)
	local angularVelocity: number = studsSpeed / wheelRadius

	-- Update the wheel motor constraints
	WheelFRMotor.AngularVelocity = angularVelocity
	WheelFRMotor.MotorMaxTorque = torque
	WheelFLMotor.AngularVelocity = -angularVelocity
	WheelFLMotor.MotorMaxTorque = torque
	if boolHandBrakeInput then
		WheelRRMotor.AngularVelocity = 0
		WheelRRMotor.MotorMaxTorque = engineParameters.handBrakeTorque
		WheelRLMotor.AngularVelocity = 0
		WheelRLMotor.MotorMaxTorque = engineParameters.handBrakeTorque
	else
		WheelRRMotor.AngularVelocity = angularVelocity
		WheelRRMotor.MotorMaxTorque = torque
		WheelRLMotor.AngularVelocity = -angularVelocity
		WheelRLMotor.MotorMaxTorque = torque
	end
end

-- Update the steering rack position based on the steering input and current speed of the car
local SteeringRack: PrismaticConstraint = steering:WaitForChild('SteeringRack') :: PrismaticConstraint

local function updateSteering(): ()
	local steeringInput: number? = ReadNumberAttribute(inputs, Constants.STEERING_INPUT_ATTRIBUTE)
	if steeringInput == nil then return end
	local nSteeringInput: number = steeringInput :: number
	local maxSpeed: number = math.max(engineParameters.forwardMaxSpeed, engineParameters.reverseMaxSpeed)
	local speed: number = math.min(getChassisForwardSpeed(), maxSpeed)
	-- Steering is reduced at higher speeds to provide smoother handling
	local steeringFactor: number = math.max(1 - (speed / maxSpeed) * steeringParameters.steeringReduction, 0)
	local steeringAmount: number = nSteeringInput * steeringFactor

	if steeringAmount > 0 then
		SteeringRack.TargetPosition = steeringAmount * SteeringRack.LowerLimit
	else
		SteeringRack.TargetPosition = -steeringAmount * SteeringRack.UpperLimit
	end
end

local function updateWheelFriction(): ()
	for _, wheelPart: BasePart in wheelParts do
		-- Find the difference between the wheel's actual linear speed and the speed it should be moving
		-- based on its current angular velocity.
		local speed: number = wheelPart.AssemblyLinearVelocity.Magnitude
		local angularVelocity: Vector3 = wheelPart.AssemblyAngularVelocity
		local targetSpeed: number = wheelRadius * angularVelocity.Magnitude
		local speedDiff: number = math.abs(speed - targetSpeed)

		local isKineticFriction: boolean = speedDiff > wheelParameters.slipThreshold
		local friction: number = if isKineticFriction then wheelParameters.kineticFriction else wheelParameters.staticFriction
		wheelPart.CustomPhysicalProperties = PhysicalProperties.new(wheelParameters.density, friction, wheelParameters.elasticity)
	end
end

local function initialize(): ()
	-- Cache all parameters in tables and update them when the corresponding Attribute updates.
	-- Parameters aren't going to change very often so there's no reason to call :GetAttribute() constantly.
	for parameter: string in engineParameters do
		engine:GetAttributeChangedSignal(parameter):Connect(function (): ()
			local value: unknown? = engine:GetAttribute(parameter)
			if value ~= nil and typeof(value) == 'number' then
				engineParameters[parameter] = value
			end
		end)
		local nvalue: unknown? = engine:GetAttribute(parameter)
		if nvalue ~= nil and typeof(nvalue) == 'number' then
			engineParameters[parameter] = nvalue
		end
	end

	for parameter: string in steeringParameters do
		steering:GetAttributeChangedSignal(parameter):Connect(function(): ()
			local value: unknown? = steering:GetAttribute(parameter)
			if value ~= nil and typeof(value) == 'number' then
				steeringParameters[parameter] = value
			end
		end)
		local nvalue: unknown? = steering:GetAttribute(parameter)
		if nvalue ~= nil and typeof(nvalue) == 'number' then
			steeringParameters[parameter] = nvalue
		end
	end

	for parameter: string in wheelParameters do
		wheels:GetAttributeChangedSignal(parameter):Connect(function(): ()
			local value: unknown? = wheels:GetAttribute(parameter)
			if value ~= nil and typeof(value) == 'number' then
				wheelParameters[parameter] = value
			end
		end)
		local nvalue: unknown? = wheels:GetAttribute(parameter)
		if nvalue ~= nil and typeof(nvalue) == 'number' then
			wheelParameters[parameter] = nvalue
		end
	end
end

local Controller = {}

function Controller:reset(): ()
	-- Reset engine attributes
	engine:SetAttribute(Constants.ENGINE_SPEED_ATTRIBUTE, 0)
	engine:SetAttribute(Constants.ENGINE_NITRO_ATTRIBUTE, 1)
	-- Reset motors
	WheelFRMotor.AngularVelocity = 0
	WheelFRMotor.MotorMaxTorque = engineParameters.minSpeedTorque
	WheelFLMotor.AngularVelocity = 0
	WheelFLMotor.MotorMaxTorque = engineParameters.minSpeedTorque
	WheelRRMotor.AngularVelocity = 0
	WheelRRMotor.MotorMaxTorque = engineParameters.minSpeedTorque
	WheelRLMotor.AngularVelocity = 0
	WheelRLMotor.MotorMaxTorque = engineParameters.minSpeedTorque
	-- Reset nitro
	NitroVelocity.Enabled = false
	-- Reset steering
	SteeringRack.TargetPosition = 0
	-- Reset wheel friction
	for _, wheelPart: BasePart in wheelParts do
		wheelPart.CustomPhysicalProperties = PhysicalProperties.new(wheelParameters.density, wheelParameters.staticFriction, wheelParameters.elasticity)
	end
end

function Controller:update(deltaTime: number): ()
	updateNitro(deltaTime)
	updateEngine(deltaTime)
	updateSteering()
	updateWheelFriction()
end

initialize()

return Controller
