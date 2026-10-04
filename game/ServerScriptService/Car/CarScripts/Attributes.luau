--!strict
-- Attribute helpers shared by the car scripts

local Attributes = {}

function Attributes.readNumber(instance: Instance, attribute: string): number?
	local value: unknown? = instance:GetAttribute(attribute)
	if typeof(value) == "number" then
		return value :: number
	end
	return nil
end

function Attributes.bindNumbers(source: Instance, parameters: { [string]: number }): ()
	for parameter: string in parameters do
		local function refresh(): ()
			local value: number? = Attributes.readNumber(source, parameter)
			if value ~= nil then parameters[parameter] = value end
		end
		source:GetAttributeChangedSignal(parameter):Connect(refresh)
		refresh()
	end
end

return Attributes
