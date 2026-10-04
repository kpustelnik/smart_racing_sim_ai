--!strict
-- FIN

local Constants = {
	-- Attributes used to get/set input values
	NITRO_INPUT_ATTRIBUTE = "nitroInput",
	HAND_BRAKE_INPUT_ATTRIBUTE = "handBrakeInput",
	THROTTLE_INPUT_ATTRIBUTE = "throttleInput",
	STEERING_INPUT_ATTRIBUTE = "steeringInput",
	-- Attributes prefixed with _ are used to share data easily between systems and not intended to be edited
	ENGINE_SPEED_ATTRIBUTE = "_speed",
	ENGINE_NITRO_ATTRIBUTE = "_nitro",
	NITRO_ENABLED_ATTRIBUTE = "_nitroEnabled",
}

return Constants