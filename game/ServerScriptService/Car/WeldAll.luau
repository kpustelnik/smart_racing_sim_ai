--!nonstrict
--  Utility script
local toWeld = game.Workspace.Truck.Body
local weldto = game.Workspace.Truck.Chassis

for _, p: Instance in ipairs(toWeld:GetDescendants()) do
  if p:IsA('WeldConstraint') then p:Destroy() end
end

for _, part: Instance in pairs(toWeld:GetDescendants()) do
  if part:IsA('BasePart') then
    local weld: WeldConstraint = Instance.new('WeldConstraint')
    weld.Part0 = weldto
    weld.Part1 = part
    weld.Parent = part

    part.Massless = true
    part.CanCollide = false
    part.CanQuery = false
    part.CanTouch = false
    part.Anchored = false
  end
end