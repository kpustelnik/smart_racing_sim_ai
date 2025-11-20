if not script:IsDescendantOf(game.Workspace) then return end

local car = script.Parent.Parent.Parent
local inputEvent = car.Scripts.SetInputEvent
local inputs = car.Inputs

inputEvent.OnServerEvent:Connect(function(player, attributeName: string, value)
  -- Make sure the attribute name is valid
  if typeof(attributeName) ~= "string" then
    return
  end

  -- Make sure the value is of a valid type
  if typeof(value) ~= "number" and typeof(value) ~= "boolean" then
    return
  end

  -- Set the attribute on the car
  inputs:SetAttribute(attributeName, value)
end)