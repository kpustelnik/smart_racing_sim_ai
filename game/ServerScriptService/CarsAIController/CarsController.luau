local Workspace = game:GetService('Workspace')
local CollectionService = game:GetService('CollectionService')
local CarsFolder: Instance = Workspace:WaitForChild('Cars')

local CheckpointsController = require(script.Parent:WaitForChild('CheckpointsController'))

local ServerStorage = game:GetService('ServerStorage')
local CarModels: Instance = ServerStorage:WaitForChild('CarModels')
local AllCarScripts = script.Parent.Parent:WaitForChild('Car')
local CarScripts: Folder = AllCarScripts:WaitForChild('CarScripts')
local Recolor = require(AllCarScripts:WaitForChild('Recolor'))

local CarsController = {}

CarsController.GetSpawns = function (): { BasePart }
  local Spawns: { BasePart } = {}
  for _, spawn: Instance in ipairs(CollectionService:GetTagged('SPAWN')) do
    if spawn:IsA('BasePart') then table.insert(Spawns, spawn) end
  end
  return Spawns
end

CarsController.PickSpawns = function (count: number, ignoreWithParts: { Instance }?): { BasePart }
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
      local pickedSpawnI: number = r:NextInteger(1, #AllSpawns)
      pickedSpawn = AllSpawns[pickedSpawnI]
      table.remove(AllSpawns, pickedSpawnI)
      if pickedSpawn == nil then continue end

      if ignoreWithParts ~= nil then
        local hasConflict: boolean = false
        for _, touchingPart: Instance in ipairs(pickedSpawn:GetTouchingParts()) do
          for _, parent: Instance in ipairs(ignoreWithParts) do
            if touchingPart:IsDescendantOf(parent) then
              hasConflict = true
              break
            end
          end
          if hasConflict then break end
        end
        if hasConflict then
          pickedSpawn = nil
          continue
        end
      end
    end

    if pickedSpawn == nil then continue end
    table.insert(PickedSpawns, pickedSpawn)
  end
  return PickedSpawns
end

CarsController.SpawnCars = function (carTypes: { string }, presetCarSpawns: { BasePart }?, collisionGroup: string?): { Model }
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

    if collisionGroup ~= nil then
      for _, v: Instance in pairs(carModelSpawned:GetDescendants()) do
        if v:IsA('BasePart') then v.CollisionGroup = collisionGroup end
      end
    end

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
  local Config: any = CarsController.LoadConfig(car)
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
      local Env: unknown? = car:GetAttribute('ENV')
      local SameEnvCars: { Instance } = {}
      if Env ~= nil and typeof(Env) == 'string' then
        for _, otherCar: Instance in pairs(CarsFolder:GetChildren()) do
          if otherCar:IsA('Model') and otherCar:GetAttribute('ENV') == Env then
            table.insert(SameEnvCars, otherCar)
          end
        end
      end
      params.FilterDescendantsInstances = SameEnvCars
    end

    local raycastResult: RaycastResult<BasePart>? = Workspace:Raycast(startPos, targetPos - startPos, params)

    local hitPosition: Vector3 = if raycastResult ~= nil then raycastResult.Position else targetPos
    
    --[[ Debug parts
    local part: Part = Instance.new("Part")
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
    ]]

    local Distance: number = (hitPosition - startPos).Magnitude
    table.insert(Result, Distance / rc.Direction.Magnitude)
    local Dot: number = 1
    if raycastResult ~= nil then Dot = (targetPos - startPos).Unit:Dot(raycastResult.Normal) end
    if rc.IncludeNormal then table.insert(Result, Dot) end
  end

  return Result
end

local EnvCGPrefix: string = 'CARS_'
CarsController.GetEnvCollisionGroup = function (envId: string): string
  return EnvCGPrefix .. envId
end

CarsController.IsEnvCollisionGroup = function (groupName: string): boolean
  return string.sub(groupName, 1, #EnvCGPrefix) == EnvCGPrefix
end

export type CarData = {
  Connections: { RBXScriptConnection },
  
  Model: Model?,
  CarEngine: Instance,
  CarInputs: Instance,
  CarController: any,

  IsStateUpdated: boolean,
  LastBestProgress: number,
  LastBestProgressTime: number,
  LastReportedProgress: number,
  CurrentLap: number,
  LastCheckpoint: number,
  LastObservation: CarObservation?,
  LastUpdateTime: number?,

  TruncationPending: boolean,
  TerminationPending: boolean,

  StartTime: number,
  FreezeStart: number?
}

CarsController.CleanupCar = function (Id: string): ()
  local carData: CarData? = CarsController.CurrentCars[Id]
  if carData == nil then return end

  while #carData.Connections > 0 do
    local conn: RBXScriptConnection = table.remove(carData.Connections)
    conn:Disconnect()
  end
  if carData.Model ~= nil then
    pcall(function() carData.Model:Destroy() end)
    carData.Model = nil
  end
  CarsController.CurrentCars[Id] = nil
end

CarsController.ApplyHighlightColor = function (model: Model, color: Color3?): ()
  local highlight: Instance? = model:FindFirstChildWhichIsA('Highlight')
  if color == nil then
    if highlight ~= nil then
      highlight:Destroy()
    end
    return
  end

  if highlight == nil then
    highlight = Instance.new('Highlight')
    highlight.Adornee = model
    highlight.Parent = model
    highlight.FillTransparency = 0.7
    highlight.OutlineTransparency = 1
  end

  highlight.FillColor = color
end

CarsController.ApplyColor = function (model: Model, color: Color3): () return Recolor(model, color) end

CarsController.SetupCarEvents = function (Id: string): ()
  local carData: CarData? = CarsController.CurrentCars[Id]
  if carData == nil then return end
  local carModel: Model? = carData.Model
  if carModel == nil then return end

  local connections: { RBXScriptConnection } = carData.Connections

  -- Cleanup the car on destroy
  local destroyConn: RBXScriptConnection = carModel.Destroying:Connect(function (): ()
    carData.Model = nil
    carData.TerminationPending = true
  end)
  table.insert(connections, destroyConn)

  -- Walls collision detection
  local wallTouchedHandler = function (hit: BasePart): ()
    if hit:IsDescendantOf(carModel) then carData.TerminationPending = true end
  end
  local Walls = CollectionService:GetTagged('WALL')
  for _, Wall: Instance in ipairs(Walls) do
    if not Wall:IsA('BasePart') then continue end

    local wallTouchedConn: RBXScriptConnection = Wall.Touched:Connect(wallTouchedHandler)
    table.insert(connections, wallTouchedConn)
  end
end

export type CarObservation = { number }

CarsController.RetrieveObservation = function (Id: string): (CarObservation?, CarObservation)
  local carData: CarData? = CarsController.CurrentCars[Id]
  if carData == nil then return end
  local carModel: Model? = carData.Model
  if carModel == nil then return end

  local carCFrame: CFrame = carModel:GetPivot()
  local carPosition = carCFrame.Position
  local carOrientation = carCFrame.LookVector
    
  local _, _, trackOrientation: number? = CheckpointsController.GetTrackProgressState(carPosition, carOrientation)
  local carEngine: Instance = carData.CarEngine

  local lastObservation = carData.LastObservation

  local obs: CarObservation = {}
  local rc: {number}? = CarsController.GetRaycasts(carModel)
  if rc == nil then error('Failed to get raycasts for car ' .. Id) end
  for _, value: number in ipairs(rc) do table.insert(obs, value) end -- Copy over

  local nitroStatus: number = carEngine:GetAttribute('_nitro') :: any
  local velocityStatus: number = carEngine:GetAttribute('_speed') :: any
  local maxVelocity: number = carEngine:GetAttribute('forwardMaxSpeed') :: any
  table.insert(obs, nitroStatus) -- Nitro status
  table.insert(obs, math.clamp(velocityStatus / maxVelocity, -1, 1)) -- Relative velocity
  
  local realOrientation: number = if trackOrientation ~= nil then trackOrientation / math.pi else 0
  local isReverse: number = if realOrientation > 0.5 or realOrientation < -0.5 then 1 else 0
  table.insert(obs, isReverse)
  
  carData.LastObservation = obs
  return obs, lastObservation
end

CarsController.CurrentCars = {} :: { [string]: CarData? }
CarsController.SetupCar = function (Id: string, EnvId: string, Model: string, Spawn: BasePart): CarData?
  CarsController.CleanupCar(Id)

  local CGName: string = CarsController.GetEnvCollisionGroup(EnvId)
  local CarInstance: Model? = CarsController.SpawnCars({ Model }, Spawn, CGName)[1]
  if CarInstance == nil then return nil end

  CarInstance:SetAttribute('ENV', EnvId)

  local carInputs = CarInstance:WaitForChild('Inputs')
  local carEngine = CarInstance:WaitForChild('Engine')
  local carController = require(CarInstance:WaitForChild("Scripts"):WaitForChild("Controller"))
  local ct: number = tick()
  local carData: CarData = {
    Connections = {},

    Model = CarInstance,
    CarInputs = carInputs,
    CarEngine = carEngine,
    CarController = carController,
    
    IsStateUpdated = false,
    LastBestProgress = 0,
    LastBestProgressTime = ct,
    LastReportedProgress = 0,
    CurrentLap = 0,
    LastCheckpoint = 1,
    LastObservation = nil,
    LastUpdateTime = nil,

    TerminationPending = false,
    TruncationPending = false,

    StartTime = ct,
    FreezeStart = nil
  }

  CarsController.CurrentCars[Id] = carData

  CarsController.SetupCarEvents(Id)
  CarsController.RetrieveObservation(Id)
  return carData
end

CarsController.GetCarData = function (Id: string): CarData? return CarsController.CurrentCars[Id] end

return CarsController