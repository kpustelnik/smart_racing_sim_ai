local StarterGui: StarterGui = game:GetService('StarterGui')

while true do
  pcall(function()
    StarterGui:SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
  end)
  task.wait()
end