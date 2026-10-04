--!strict
local Workspace = game:GetService('Workspace')
local CollectionService = game:GetService('CollectionService')
local CarsFolder: Instance = Workspace:WaitForChild('Cars')

local CheckpointsController = require(script.Parent:WaitForChild('CheckpointsController'))

local ServerStorage = game:GetService('ServerStorage')
local CarModels: Instance = ServerStorage:WaitForChild('CarModels')
local AllCarScripts = script.Parent.Parent:WaitForChild('Car')
local CarScripts: Folder = AllCarScripts:WaitForChild('CarScripts') :: Folder
local Recolor = require(AllCarScripts:WaitForChild('Recolor'))

local CarsController = {}

local IncludePrevAction: boolean = true
local ACTION_DIM: number = 3

local function readNumberAttribute(instance: Instance, attribute: string, fallback: number): number
  local value: unknown? = instance:GetAttribute(attribute)
  if typeof(value) ~= 'number' then return fallback end

  local number: number = value :: number
  if number ~= number then return fallback end -- NaN
  return number
end

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
    local pickedSpawn: BasePart? = nil
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
      break
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
  IgnoreCars: boolean,
  IgnoreXZRotation: boolean,
  IgnoreTrack: boolean?,
  Offset: Vector3?
}

local EnableDebugPartsRays: boolean = false
local EnableDebugPartsRayHits: boolean = false
local DefaultRaycastOffset = Vector3.new(0, 7, 0)
CarsController.GetRaycasts = function (car: Model): { number }?
  local Config: any = CarsController.LoadConfig(car)
  if Config == nil then return end

  local filters: { [string]: RaycastParams } = {}
  local function filterFor(ignoreCars: boolean, ignoreTrack: boolean): RaycastParams
    local key: string = `{ignoreCars}/{ignoreTrack}`
    local cached: RaycastParams? = filters[key]
    if cached ~= nil then return cached end

    local params: RaycastParams = RaycastParams.new()
    params.FilterType = Enum.RaycastFilterType.Exclude
    params.RespectCanCollide = true
    local IgnoredDescendants: { Instance } = {}
    if ignoreCars then
      table.insert(IgnoredDescendants, CarsFolder)
    else
      local Env: unknown? = car:GetAttribute('ENV')
      if Env ~= nil and typeof(Env) == 'string' then
        for _, otherCar: Instance in pairs(CarsFolder:GetChildren()) do
          if otherCar:IsA('Model') and otherCar:GetAttribute('ENV') ~= Env then -- Ignore cars from other environments
            table.insert(IgnoredDescendants, otherCar)
          end
        end
      end
      table.insert(IgnoredDescendants, car) -- Ignore the car itself
    end
    -- Ignore the track if cars should be detected
    if ignoreTrack then
      for _, roadPart: Instance in ipairs(CollectionService:GetTagged('ROAD')) do
        if roadPart:IsA('BasePart') then table.insert(IgnoredDescendants, roadPart) end
      end
    end
    params.FilterDescendantsInstances = IgnoredDescendants

    filters[key] = params
    return params
  end

  local ConfigRays: { RayConfig } = Config.Rays
  local Result: { number } = {}
  for _, rc: RayConfig in ipairs(ConfigRays) do
    local originCFrame: CFrame = rc.Source.CFrame
    if rc.IgnoreXZRotation then
      originCFrame = CFrame.new(rc.Source.Position)
      local _, Y: number, _ = rc.Source.CFrame:ToEulerAnglesYXZ()
      originCFrame *= CFrame.Angles(0, Y, 0) -- Only preserve the Y rotation
    end

    local targetPos: Vector3 = (originCFrame * CFrame.new(rc.Direction)).Position
    local startPos: Vector3 = originCFrame.Position

    local IgnoreTrack: boolean = if rc.IgnoreTrack ~= nil then rc.IgnoreTrack else true
    local params: RaycastParams = filterFor(rc.IgnoreCars == true, IgnoreTrack)

    local offset = if rc.Offset ~= nil then rc.Offset else if rc.IgnoreCars then DefaultRaycastOffset else Vector3.new(0, 1, 0)
    local rayOrigin: Vector3 = startPos + offset
    local rayVector: Vector3 = targetPos - startPos
    local raycastResult: RaycastResult<BasePart>? = Workspace:Raycast(rayOrigin, rayVector, params)

    local hitPosition: Vector3 = if raycastResult ~= nil then raycastResult.Position else rayOrigin + rayVector
    
    if EnableDebugPartsRays then -- Debug parts
      local part: Part = Instance.new("Part")
      part.CanCollide = false
      part.Anchored = true
      part.CanTouch = false
      part.CanQuery = false
      part.BrickColor = if rc.IgnoreCars then BrickColor.new("Really red") else BrickColor.new("Bright blue")
      part.Size = Vector3.new(1, 1, (hitPosition - rayOrigin).Magnitude)
      part.Parent = car
      part.CFrame = CFrame.lookAt((hitPosition + rayOrigin) / 2, hitPosition)
      part.Transparency = 0.5
      task.defer(function(): () task.wait(0.1); part:Destroy() end)
    end
    if EnableDebugPartsRayHits and raycastResult ~= nil then -- Debug hit parts
      local part: Part = Instance.new("Part")
      part.CanCollide = false
      part.Anchored = true
      part.CanTouch = false
      part.CanQuery = false
      part.BrickColor = if rc.IgnoreCars then BrickColor.new("Bright yellow") else BrickColor.new("Bright green")
      part.Size = Vector3.new(1, 1, 1)
      part.Parent = car
      part.CFrame = CFrame.new(hitPosition)
      task.defer(function(): () task.wait(0.1); part:Destroy() end)
    end

    local Distance: number = (hitPosition - rayOrigin).Magnitude
    table.insert(Result, Distance / rc.Direction.Magnitude)
    local Dot: number = 1
    if raycastResult ~= nil then Dot = rayVector.Unit:Dot(raycastResult.Normal) end
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
  LastLoopStartTime: number,
  LastReportedProgress: number,
  CurrentLap: number,
  LastCheckpoint: number,
  LastObservation: CarObservation?,
  LastAction: { number }?,
  FrozenVelocities: { [BasePart]: { Linear: Vector3, Angular: Vector3 } }?,
  LastUpdateTime: number?,
  IsTouchingWall: boolean,
  IsTouchingAgent: boolean,

  TruncationPending: boolean,
  TerminationPending: boolean,

  PrevNextDistance: number?,
  PrevFollowingDistance: number?,

  StartTime: number,
  FreezeStart: number?
}

CarsController.CurrentCars = {} :: { [string]: CarData? }

CarsController.CleanupCar = function (Id: string): ()
  local carData: CarData? = CarsController.CurrentCars[Id]
  if carData == nil then return end

  while #carData.Connections > 0 do
    local conn: RBXScriptConnection? = table.remove(carData.Connections)
    if conn ~= nil then conn:Disconnect() end
  end

  local carModel: Model? = carData.Model
  if carModel ~= nil then
    pcall(function() carModel:Destroy() end)
    carData.Model = nil
  end
  CarsController.CurrentCars[Id] = nil
end

CarsController.ApplyHighlightColor = function (model: Model, color: Color3?): ()
  local existing: Highlight? = model:FindFirstChildWhichIsA('Highlight') :: Highlight?
  if color == nil then
    if existing ~= nil then
      existing:Destroy()
    end
    return
  end

  local highlight: Highlight = existing or Instance.new('Highlight')
  if existing == nil then
    highlight.Adornee = model
    highlight.Parent = model
    highlight.FillTransparency = 0.7
    highlight.OutlineTransparency = 1
  end

  highlight.FillColor = color
end

CarsController.ApplyColor = function (model: Model, color: Color3): () return Recolor(model, color) end

local wallConnections: { RBXScriptConnection } = {}

local function findCarDataForPart(part: BasePart): CarData?
  local model: Instance? = part:FindFirstAncestorOfClass('Model')
  while model ~= nil do
    if model.Parent == CarsFolder then
      local agentId: unknown? = model:GetAttribute('AGENT_ID')
      if typeof(agentId) == 'string' then return CarsController.CurrentCars[agentId] end
      return nil
    end
    model = model:FindFirstAncestorOfClass('Model')
  end
  return nil
end

CarsController.EnsureWallEventsConnected = function (): ()
  if #wallConnections > 0 then return end

  local function onWallTouched(hit: BasePart): ()
    local carData: CarData? = findCarDataForPart(hit)
    if carData ~= nil then carData.IsTouchingWall = true end
  end

  for _, wall: Instance in ipairs(CollectionService:GetTagged('WALL')) do
    if not wall:IsA('BasePart') then continue end
    table.insert(wallConnections, wall.Touched:Connect(onWallTouched))
  end
end

CarsController.DisconnectWallEvents = function (): ()
  for _, connection: RBXScriptConnection in ipairs(wallConnections) do connection:Disconnect() end
  table.clear(wallConnections)
end

CarsController.SetupCarEvents = function (Id: string): ()
  local carData: CarData? = CarsController.CurrentCars[Id]
  if carData == nil then return end
  local carModel: Model? = carData.Model
  if carModel == nil then return end

  local connections: { RBXScriptConnection } = carData.Connections

  -- Cleanup the car on destroy
  local destroyConn: RBXScriptConnection = carModel.Destroying:Connect(function (): ()
    carData.Model = nil
    carData.TruncationPending = true
  end)
  table.insert(connections, destroyConn)

  -- Cars collision detectionue end
  local chassis: Instance? = carModel:FindFirstChild('Chassis')
  if chassis ~= nil and chassis:IsA('BasePart') then
    local carPartTouchedConn: RBXScriptConnection = chassis.Touched:Connect(function (hit: BasePart): ()
      for _, otherCarModel: Instance in ipairs(CarsFolder:GetChildren()) do
        if otherCarModel:IsA('Model') and otherCarModel ~= carModel and hit:IsDescendantOf(otherCarModel) then
          local env: unknown? = otherCarModel:GetAttribute('ENV')
          if env ~= nil and env == carModel:GetAttribute('ENV') then
            carData.IsTouchingAgent = true
            break
          end
        end
      end
    end)
    table.insert(connections, carPartTouchedConn)
  end
end

export type CarObservation = { number }

CarsController.RetrieveObservation = function (Id: string): (CarObservation?, CarObservation?)
  local carData: CarData? = CarsController.CurrentCars[Id]
  if carData == nil then return nil, nil end
  local carModel: Model? = carData.Model
  if carModel == nil then return nil, nil end

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

  local nitroStatus: number = readNumberAttribute(carEngine, '_nitro', 0)
  local velocityStatus: number = readNumberAttribute(carEngine, '_speed', 0)
  local maxVelocity: number = readNumberAttribute(carEngine, 'forwardMaxSpeed', 0)
  table.insert(obs, nitroStatus) -- Nitro status
  -- Relative velocity
  table.insert(obs, if maxVelocity > 0 then math.clamp(velocityStatus / maxVelocity, -1, 1) else 0)
  
  local realOrientation: number = if trackOrientation ~= nil then trackOrientation / math.pi else 0
  local isReverse: number = if realOrientation > 0.5 or realOrientation < -0.5 then 1 else 0
  table.insert(obs, isReverse)

  if IncludePrevAction then
    -- Zeroes on the first step of an episode, when there is no preceding action.
    local lastAction: { number }? = carData.LastAction
    for i = 1, ACTION_DIM do
      table.insert(obs, if lastAction ~= nil then lastAction[i] else 0)
    end
  end
  
  carData.LastObservation = obs
  return obs, lastObservation
end

CarsController.SetupCar = function (Id: string, EnvId: string, Model: string, Spawn: BasePart): CarData?
  CarsController.CleanupCar(Id)

  local CGName: string = CarsController.GetEnvCollisionGroup(EnvId)
  local CarInstance: Model? = CarsController.SpawnCars({ Model }, { Spawn }, CGName)[1]
  if CarInstance == nil then return nil end

  CarInstance:SetAttribute('ENV', EnvId)
  CarInstance:SetAttribute('AGENT_ID', Id)

  local carInputs = CarInstance:WaitForChild('Inputs')
  local carEngine = CarInstance:WaitForChild('Engine')
  local carController = (require :: any)(CarInstance:WaitForChild("Scripts"):WaitForChild("Controller"))
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
    LastLoopStartTime = ct,
    LastReportedProgress = 0,
    CurrentLap = 0,
    LastCheckpoint = 1,
    LastObservation = nil,
    LastAction = nil,
    FrozenVelocities = nil,
    LastUpdateTime = nil,
    IsTouchingWall = false,
    IsTouchingAgent = false,

    TerminationPending = false,
    TruncationPending = false,

    PrevNextDistance = nil,
    PrevFollowingDistance = nil,

    StartTime = ct,
    FreezeStart = nil
  }

  CarsController.CurrentCars[Id] = carData

  CarsController.EnsureWallEventsConnected()
  CarsController.SetupCarEvents(Id)
  CarsController.RetrieveObservation(Id)
  return carData
end

CarsController.GetCarData = function (Id: string): CarData? return CarsController.CurrentCars[Id] end

return CarsController