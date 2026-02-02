--!strict
-- FIN
-- Redresser Server Script by Roblox (Adjusted)

local Workspace: Workspace = game:GetService('Workspace')
if not script:IsDescendantOf(Workspace) then return end -- Check if the car was already spawned

local car: Folder = script.Parent.Parent.Parent
local engine: Folder = car:WaitForChild('Engine') :: Folder
local steering: Folder = car:WaitForChild('Steering') :: Folder
local suspension: Folder = car:WaitForChild('Suspension') :: Folder
local antiRoll: Folder = car:WaitForChild('AntiRoll') :: Folder
local redress: Folder = car:WaitForChild('Redress') :: Folder

local ReadNumberAttribute = function (i: Instance, attrName: string): number?
	local value: unknown? = i:GetAttribute(attrName)
	if value ~= nil and typeof(value) == "number" then
		return value :: number
	end
	return nil
end

-- Update suspension constraints
local WheelFRSuspension: SpringConstraint = suspension:WaitForChild("WheelFRSuspension") :: SpringConstraint
local WheelFLSuspension: SpringConstraint = suspension:WaitForChild("WheelFLSuspension") :: SpringConstraint
local WheelRRSuspension: SpringConstraint = suspension:WaitForChild("WheelRRSuspension") :: SpringConstraint
local WheelRLSuspension: SpringConstraint = suspension:WaitForChild("WheelRLSuspension") :: SpringConstraint

local WheelFRMotor: CylindricalConstraint = engine:WaitForChild("WheelFRMotor") :: CylindricalConstraint
local WheelFLMotor: CylindricalConstraint = engine:WaitForChild("WheelFLMotor") :: CylindricalConstraint
local WheelRRMotor: CylindricalConstraint = engine:WaitForChild("WheelRRMotor") :: CylindricalConstraint
local WheelRLMotor: CylindricalConstraint = engine:WaitForChild("WheelRLMotor") :: CylindricalConstraint

local function updateSuspension(): ()
	local frontLength: number? = ReadNumberAttribute(suspension, "frontSuspensionLength")
	local frontStiffness: number? = ReadNumberAttribute(suspension, "frontSuspensionStiffness")
	local frontDamping: number? = ReadNumberAttribute(suspension, "frontSuspensionDamping")
	local rearLength: number? = ReadNumberAttribute(suspension, "rearSuspensionLength")
	local rearStiffness: number? = ReadNumberAttribute(suspension, "rearSuspensionStiffness")
	local rearDamping: number? = ReadNumberAttribute(suspension, "rearSuspensionDamping")
	local frontCamber: number? = ReadNumberAttribute(suspension, "frontCamber")
	local rearCamber: number? = ReadNumberAttribute(suspension, "rearCamber")

	if frontLength ~= nil then
		WheelFRSuspension.FreeLength = frontLength
		WheelFLSuspension.FreeLength = frontLength
	end
	if rearLength ~= nil then
		WheelRRSuspension.FreeLength = rearLength
		WheelRLSuspension.FreeLength = rearLength
	end
	if frontDamping ~= nil then
		WheelFRSuspension.Damping = frontDamping
		WheelFLSuspension.Damping = frontDamping
	end
	if rearDamping ~= nil then
		WheelRRSuspension.Damping = rearDamping
		WheelRLSuspension.Damping = rearDamping
	end
	if frontStiffness ~= nil then
		WheelFRSuspension.Stiffness = frontStiffness
		WheelFLSuspension.Stiffness = frontStiffness
	end
	if rearStiffness ~= nil then
		WheelRRSuspension.Stiffness = rearStiffness
		WheelRLSuspension.Stiffness = rearStiffness
	end

	if frontCamber ~= nil then
		WheelFRMotor.InclinationAngle = 90 + frontCamber
		WheelFLMotor.InclinationAngle = 90 + frontCamber
	end
	if rearCamber ~= nil then
		WheelRRMotor.InclinationAngle = 90 + rearCamber
		WheelRLMotor.InclinationAngle = 90 + rearCamber
	end
end

-- Update anti roll bar constraints
local AntiRollBarFL: SpringConstraint = antiRoll:WaitForChild("AntiRollBarFL") :: SpringConstraint
local AntiRollBarFR: SpringConstraint = antiRoll:WaitForChild("AntiRollBarFR") :: SpringConstraint
local AntiRollBarRL: SpringConstraint = antiRoll:WaitForChild("AntiRollBarRL") :: SpringConstraint
local AntiRollBarRR: SpringConstraint = antiRoll:WaitForChild("AntiRollBarRR") :: SpringConstraint
local function updateAntiRoll(): ()
	local frontDamping: number? = ReadNumberAttribute(antiRoll, "frontAntiRollBarDamping")
	local frontStiffness: number? = ReadNumberAttribute(antiRoll, "frontAntiRollBarStiffness")
	local rearStiffness: number? = ReadNumberAttribute(antiRoll, "rearAntiRollBarStiffness")
	local rearDamping: number? = ReadNumberAttribute(antiRoll, "rearAntiRollBarDamping")

	if frontDamping ~= nil then
		AntiRollBarFL.Damping = frontDamping
		AntiRollBarFR.Damping = frontDamping
	end
	if frontStiffness ~= nil then
		AntiRollBarFR.Stiffness = frontStiffness
		AntiRollBarFL.Stiffness = frontStiffness
	end
	if rearDamping ~= nil then
		AntiRollBarRR.Damping = rearDamping
		AntiRollBarRL.Damping = rearDamping
	end
	if rearStiffness ~= nil then
		AntiRollBarRR.Stiffness = rearStiffness
		AntiRollBarRL.Stiffness = rearStiffness
	end
end

-- Update steering constraints
local SteeringRack: PrismaticConstraint = steering:WaitForChild("SteeringRack") :: PrismaticConstraint

local function updateSteering(): ()
	local length: number? = ReadNumberAttribute(steering, "steeringRackLength")
	local responsiveness: number? = ReadNumberAttribute(steering, "steeringRackResponsiveness")
	local speed: number? = ReadNumberAttribute(steering, "steeringRackSpeed")
	local maxForce: number? = ReadNumberAttribute(steering, "steeringRackMaxForce")

	if length ~= nil then
		SteeringRack.UpperLimit = length
		SteeringRack.LowerLimit = -length
	end
	if responsiveness ~= nil then
		SteeringRack.LinearResponsiveness = responsiveness
	end
	if speed ~= nil then
		SteeringRack.Speed = speed
	end
	if maxForce ~= nil then
		SteeringRack.ServoMaxForce = maxForce
	end
end

-- Update redress constraints
local RedressOrientation: AlignOrientation = redress:WaitForChild("RedressOrientation") :: AlignOrientation
local function updateRedress(): ()
	local maxTorque: number? = ReadNumberAttribute(redress, "redressMaxTorque")
	local responsiveness: number? = ReadNumberAttribute(redress, "redressResponsiveness")

	if maxTorque ~= nil then
		RedressOrientation.MaxTorque = maxTorque
	end
	if responsiveness ~= nil then
		RedressOrientation.Responsiveness = responsiveness
	end
end

local function initialize(): ()
	suspension.AttributeChanged:Connect(updateSuspension)
	updateSuspension()
	antiRoll.AttributeChanged:Connect(updateAntiRoll)
	updateAntiRoll()
	steering.AttributeChanged:Connect(updateSteering)
	updateSteering()
	redress.AttributeChanged:Connect(updateRedress)
	updateRedress()
end

initialize()
