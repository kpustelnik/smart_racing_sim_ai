--!strict
-- FIN
local RECOLOR_TAG: string = 'Recolor'

return function (model: Model, color: Color3): ()
  for _, Descendant: Instance in ipairs(model:GetDescendants()) do
    if Descendant:IsA('BasePart') and Descendant:HasTag(RECOLOR_TAG) then
      Descendant.Color = color
    end
  end
end