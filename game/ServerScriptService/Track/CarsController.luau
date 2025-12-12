local Workspace = game:GetService('Workspace')
local CarsSpawnersFolder = Workspace:WaitForChild('CarSpawns')
local CarsFolder = Workspace:WaitForChild('Cars')

local ServerStorage = game:GetService('ServerStorage')
local CarModels = ServerStorage:WaitForChild('CarModels')
local CarScripts = script.Parent.Parent:WaitForChild('Car'):WaitForChild('CarScripts')

local CarsController = {}

CarsController.GetSpawns = function (): { BasePart }
  local Spawns: { BasePart } = {}
  for _, spawn: Instance in ipairs(CarsSpawnersFolder:GetChildren()) do
    if spawn:IsA('BasePart') then
      table.insert(Spawns, spawn)
    end
  end
  return Spawns
end

CarsController.PickSpawns = function (count: number): { BasePart }
  local r: Random = Random.new()
  local AllSpawns: { BasePart } = CarsController.GetSpawns()
  local PickedSpawns: { BasePart } = {}
  for _ = 1, count do
    local pickedSpawn: BasePart?
    while true do
      if #AllSpawns == 0 then
        pickedSpawn = nil
        break
      end
      local pickedSpawnI = r:NextInteger(1, #AllSpawns)
      pickedSpawn = AllSpawns[pickedSpawnI]
      table.remove(AllSpawns, pickedSpawnI)
      if pickedSpawn ~= nil then break end
    end

    if pickedSpawn == nil then continue end
    table.insert(PickedSpawns, pickedSpawn)
  end
  return PickedSpawns
end

CarsController.SpawnCars = function (carTypes: { string }, presetCarSpawns: { BasePart }?): { Model }
  local carSpawns: { BasePart } = if presetCarSpawns ~= nil then presetCarSpawns else CarsController.PickSpawns(#carTypes)

  local spawnedCars: { Model } = {}
  for i: number, carType: string in ipairs(carTypes) do
    local carModel: Instance? = CarModels:FindFirstChild(carType)
    if carModel == nil or not carModel:IsA('Model') then continue end
    local carSpawn: BasePart? = carSpawns[i]
    if carSpawn == nil then continue end

    local carModelSpawned: Model = carModel:Clone()
    carModelSpawned.Parent = CarsFolder
    carModelSpawned:PivotTo(carSpawn.CFrame)

    local Scripts: Folder = CarScripts:Clone()
    Scripts.Name = 'Scripts'
    Scripts.Parent = carModelSpawned
    
    table.insert(spawnedCars, carModelSpawned)
  end

  return spawnedCars
end

CarsController.LoadConfig = function (car: Model): any
  local ConfigModule: Instance? = car:FindFirstChild("Config")
  if ConfigModule == nil or not ConfigModule:IsA('ModuleScript') then return end

  return require(ConfigModule) :: any
end

type RayConfig = {
  Source: Part,
  Direction: Vector3,
  IncludeNormal: boolean,
  IgnoreCars: boolean
}

CarsController.GetRaycasts = function (car: Model): { number }?
  local Config = CarsController.LoadConfig(car)
  if Config == nil then return end

  local ConfigRays: { RayConfig } = Config.Rays
  local Result: { number } = {}
  for _, rc: RayConfig in ipairs(ConfigRays) do
    local originCFrame: CFrame = CFrame.new(rc.Source.Position)
    local _, Y: number, _ = rc.Source.CFrame:ToEulerAnglesYXZ()
    originCFrame *= CFrame.Angles(0, Y, 0) -- Only preserve the Y rotation

    local targetPos: Vector3 = (originCFrame * CFrame.new(rc.Direction)).Position
    local startPos: Vector3 = originCFrame.Position

    local params: RaycastParams = RaycastParams.new()
    params.FilterType = Enum.RaycastFilterType.Exclude
    params.RespectCanCollide = true
    if rc.IgnoreCars then
      params.FilterDescendantsInstances = { CarsFolder }
    else
      params.FilterDescendantsInstances = { car }
    end

    local raycastResult: RaycastResult<BasePart>? = Workspace:Raycast(startPos, targetPos - startPos, params)

    local hitPosition: Vector3 = if raycastResult ~= nil then raycastResult.Position else targetPos
    
    local part = Instance.new("Part")
    part.CanCollide = false
    part.Anchored = true
    part.CanTouch = false
    part.CanQuery = false
    part.BrickColor = BrickColor.new("Really red")
    part.Size = Vector3.new(1, 1, (hitPosition - startPos).Magnitude)
    part.Parent = car
    part.CFrame = CFrame.lookAt((hitPosition + startPos) / 2, hitPosition)
    part.Transparency = 0.5
    task.defer(function(): () task.wait(); part:Destroy() end)

    local Distance: number = (hitPosition - startPos).Magnitude
    table.insert(Result, Distance / rc.Direction.Magnitude)
    local Dot: number = 1
    if raycastResult ~= nil then Dot = (targetPos - startPos).Unit:Dot(raycastResult.Normal) end
    if rc.IncludeNormal then table.insert(Result, Dot) end
  end

  return Result
end

return CarsController