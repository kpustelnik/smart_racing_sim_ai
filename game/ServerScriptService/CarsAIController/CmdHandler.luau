--!strict
local RunService = game:GetService('RunService')
local HttpService = game:GetService('HttpService')
local PhysicsService = game:GetService('PhysicsService')
local CarsController = require(script.Parent:WaitForChild('CarsController'))
local CheckpointsController = require(script.Parent:WaitForChild('CheckpointsController'))

local CmdHandler = {}
CmdHandler.__index = CmdHandler

export type BackendMessage = {
  command: string,
  data: any,
  envid: string?
}

local EnvColors = {
  Color3.new(1, 0, 0),
  Color3.new(0, 1, 0),
  Color3.new(0, 0, 1),
  Color3.new(1, 1, 0),
  Color3.new(1, 0, 1),
  Color3.new(0, 1, 1),
}
local LastEnvColor = 0
local NoProgressTimeout: number = 5 -- TODO: Add as global parameter

-- Creates a new environment
type CreateEnvData = {}
CmdHandler['CREATE_ENV'] = function (self: CmdHandler, data: CreateEnvData, envId: string?): ()
  if envId == nil then return end
  if self.Envs[envId] ~= nil then
    warn(`Environment {envId} already exists`)
    return
  end
  LastEnvColor = (LastEnvColor % #EnvColors) + 1
  local curEnvColor: Color3 = EnvColors[LastEnvColor]
  self.Envs[envId] = {
    Agents = {},
    Color = curEnvColor
  }

  local CollisionGroupName: string = CarsController.GetEnvCollisionGroup(envId)
  local suc: boolean, res = pcall(function()
    PhysicsService:RegisterCollisionGroup(CollisionGroupName)

    for _: number, existingCGData: any in ipairs(PhysicsService:GetRegisteredCollisionGroups()) do
      local existingCG: string = existingCGData.name
      if CarsController.IsEnvCollisionGroup(existingCG) and existingCG ~= CollisionGroupName then
        PhysicsService:CollisionGroupSetCollidable(CollisionGroupName, existingCG, false) -- Disable collisions between different envs
      end
    end
  end)
  if not suc then
    warn(`Failed to create collision group for env {envId}: {res}`)
  end

  -- Enable / disable cars collisions within the env
  CmdHandler['UPDATE_COLLISIONS'](self, { } :: UpdateCollisionsData, envId)
end

type UpdateCollisionsData = {
  enable_collisions: boolean?
}
CmdHandler['UPDATE_COLLISIONS'] = function (self: CmdHandler, data: UpdateCollisionsData, envId: string?): () 
  local enableCollisions: boolean? = data.enable_collisions
  if enableCollisions ~= nil then
    self.EnableEnvCollisions = enableCollisions
  end
  for cEnvId: string, _: EnvData in pairs(self.Envs) do
    local CollisionGroupName: string = CarsController.GetEnvCollisionGroup(cEnvId)
    local suc: boolean, res = pcall(function()
      PhysicsService:CollisionGroupSetCollidable(CollisionGroupName, CollisionGroupName, self.EnableEnvCollisions)
    end)
    if not suc then
      warn(`Failed to update collision settings for env {envId}: {res}`)
    end
  end
end

-- Spawns selected agents into the environment
type SpawnAgentsData = { agents: { string } }
CmdHandler['SPAWN_AGENTS'] = function (self: CmdHandler, data: SpawnAgentsData, envId: string?): () 
  if envId == nil then return end
  if self.Envs[envId] == nil then
    CmdHandler['CREATE_ENV'](self, {} :: CreateEnvData, envId) -- Ensure env exists
  end
  local envData: EnvData = self.Envs[envId]

  local currentAgentModels: { Instance } = {}
  for _, agentId: string in ipairs(envData.Agents) do
    local carData: CarsController.CarData? = CarsController.GetCarData(agentId)
    if carData ~= nil then
      local carModel: Model? = carData.Model
      if carModel ~= nil then table.insert(currentAgentModels, carData.Model :: Instance) end
    end
  end

  local pickedSpawns: { BasePart } = CarsController.PickSpawns(#data.agents, if self.EnableEnvCollisions then currentAgentModels else nil)
  if #pickedSpawns < #data.agents and self.EnableEnvCollisions then
    pickedSpawns = CarsController.PickSpawns(#data.agents, nil) -- Fallback to global spawns
  end
  
  local agents: { string } = data.agents
  for i: number, agentId: string in ipairs(agents) do
    local ci = table.find(envData.Agents, agentId)
    if ci == nil then table.insert(envData.Agents, agentId) end

    local curSpawn: BasePart? = pickedSpawns[i]
    if curSpawn == nil then continue end

    local carData: CarsController.CarData? = CarsController.SetupCar(agentId, envId, self.CarModel, curSpawn)
    if carData == nil then continue end
    local carModel: Model? = carData.Model
    if carModel == nil then continue end

    CarsController.ApplyHighlightColor(carModel, envData.Color)
  end
end

-- Resets selected agents in the environment
type ResetData = { agents: { string } }
CmdHandler['RESET_AGENTS'] = function (self: CmdHandler, data: ResetData, envId: string?): ()
  if envId == nil then return end
  local agents: { string } = data.agents
  return CmdHandler['SPAWN_AGENTS'](self, { agents = agents } :: SpawnAgentsData, envId)
end

type RemoveAgentsData = { agents: { string } }
CmdHandler['REMOVE_AGENTS'] = function (self: CmdHandler, data: RemoveAgentsData, envId: string?): ()
  if envId == nil then return end
  local envData: EnvData? = self.Envs[envId]
  if envData == nil then return end

  local agents: { string } = data.agents
  for _, agentId: string in ipairs(agents) do
    CarsController.CleanupCar(agentId)

    local ci = table.find(envData.Agents, agentId)
    if ci ~= nil then
      table.remove(envData.Agents, ci)
    end
  end
end

type CloseData = {}
CmdHandler['CLOSE'] = function (self: CmdHandler, data: CloseData, envId: string?): ()
  if envId == nil then return end
  local envData: EnvData? = self.Envs[envId]
  if envData == nil then return end

  CmdHandler['REMOVE_AGENTS'](self, { agents = table.clone(envData.Agents) } :: RemoveAgentsData, envId)
  self.Envs[envId] = nil

  local CollisionGroupName: string = CarsController.GetEnvCollisionGroup(envId)
  pcall(function()
    PhysicsService:UnregisterCollisionGroup(CollisionGroupName)
  end)
end

type CloseAllData = {}
CmdHandler['CLOSE_ALL'] = function (self: CmdHandler, data: CloseAllData, envId: string?): ()
  for cEnvId: string in pairs(self.Envs) do
    CmdHandler['CLOSE'](self, {} :: CloseData, cEnvId)
  end
end

type ActionData = { [string]: { number } }
CmdHandler['ACTION'] = function (self: CmdHandler, data: ActionData, envId: string?): ()
  for agentId: string, agentData in pairs(data) do
    local carData: CarsController.CarData? = CarsController.GetCarData(agentId)
    if carData == nil then continue end

    carData.CarInputs:SetAttribute('throttleInput', agentData[1])
    carData.CarInputs:SetAttribute('steeringInput', agentData[2])
    carData.CarInputs:SetAttribute('nitroInput', agentData[3] > 0.5)
    carData.IsStateUpdated = true

    -- Update car physics
    local lastUpdateTime: number? = carData.LastUpdateTime
    if lastUpdateTime ~= nil then
      local dt: number = tick() - lastUpdateTime
      carData.CarController:update(dt)
    end
    carData.LastUpdateTime = tick()
  end
end

type FreezeData = {
  status: boolean
}
CmdHandler['FREEZE'] = function (self: CmdHandler, data: FreezeData, envId: string?): ()
  for _: string, envData in pairs(self.Envs) do
    for _, agentId: string in ipairs(envData.Agents) do
      local carData: CarsController.CarData? = CarsController.GetCarData(agentId)
      if carData == nil then continue end

      -- Anchor / unanchor the car model
      local carModel: Model? = carData.Model
      if carModel ~= nil then
        for _, part: BasePart in ipairs(carModel:GetDescendants()) do
          if part:IsA('BasePart') then
            part.Anchored = data.status
          end
        end
      end

      local cur: number = tick()
      if data.status then -- Start the freeze if not started already
        if carData.FreezeStart == nil then
          carData.FreezeStart = cur
        end
      else
        if carData.FreezeStart ~= nil then -- Stop the freeze and revert timings
          local fs: number = carData.FreezeStart
          local diff: number = cur - fs

          carData.LastBestProgressTime += diff
          carData.StartTime += diff
          if carData.LastUpdateTime ~= nil then
            carData.LastUpdateTime += diff
          end
        end
        carData.FreezeStart = nil
      end
    end
  end
end

CmdHandler['PING'] = function (self: CmdHandler, _: FreezeData, envId: string?): () --[[ Do nothing. It's just for maintaining the communication. ]] end

CmdHandler.Handle = function (self: CmdHandler, msg: BackendMessage): ()
  local command: string = msg.command
  local data: any = msg.data
  local envid: string? = msg.envid

  local handleFunc: any = CmdHandler[command]
  if handleFunc then
    local success: boolean, res = pcall(handleFunc, self, data, envid)
    if not success then warn(`Error handling command {command} for env {envid} - {res}`) end
  else
    warn(`Unknown command received: {command}`)
  end
end

CmdHandler.Cleanup = function (self: CmdHandler): ()
  CmdHandler['CLOSE_ALL'](self, {} :: CloseAllData, nil)
  if self.HeartbeatEvent ~= nil then
    self.HeartbeatEvent:Disconnect()
    self.HeartbeatEvent = nil
  end
end

CmdHandler.GetCarEnv = function (self: CmdHandler, carData: CarsController.CarData): string?
  for EnvId: string, EnvData: EnvData in pairs(self.Envs) do
    for _, AgentId: string in ipairs(EnvData.Agents) do
      local cCarData: CarsController.CarData? = CarsController.GetCarData(AgentId)
      if cCarData == nil then continue end
      if cCarData == carData then return EnvId end
    end
  end
  return nil
end

CmdHandler.GetNextCar = function (self: CmdHandler, carData: CarsController.CarData): (CarsController.CarData?, number?)
  local EnvId: string = self:GetCarEnv(carData) :: string
  if EnvId == nil then return nil end
  local EnvData: EnvData? = self.Envs[EnvId]
  if EnvData == nil then return nil end

  local carPivot: CFrame = carData.Model:GetPivot()
  local carTrackProgress: number = CheckpointsController.GetTrackProgressState(carPivot.Position, carPivot.LookVector)

  local nextCarData: CarsController.CarData? = nil
  local nextCarProgress: number = math.huge
  for _, AgentId: string in ipairs(EnvData.Agents) do
    local cCarData: CarsController.CarData? = CarsController.GetCarData(AgentId)
    if cCarData == nil then continue end
    local cCarModel: Model? = cCarData.Model
    if cCarModel == nil or cCarModel == carData.Model then continue end

    local cCarPivot: CFrame = cCarModel:GetPivot()
    local cProgress: number = CheckpointsController.GetTrackProgressState(cCarPivot.Position, cCarPivot.LookVector)
    if cProgress > carTrackProgress and cProgress < nextCarProgress then
      nextCarData = cCarData
      nextCarProgress = cProgress
    end
  end

  if nextCarData == nil then return nil, nil end
  return nextCarData, nextCarProgress
end

CmdHandler.GetFollowingCar = function (self: CmdHandler, carData: CarsController.CarData): (CarsController.CarData?, number?)
  local EnvId: string = self:GetCarEnv(carData) :: string
  if EnvId == nil then return nil end
  local EnvData: EnvData? = self.Envs[EnvId]
  if EnvData == nil then return nil end

  local carPivot: CFrame = carData.Model:GetPivot()
  local carTrackProgress: number = CheckpointsController.GetTrackProgressState(carPivot.Position, carPivot.LookVector)

  local followingCarData: CarsController.CarData? = nil
  local followingCarProgress: number = 0
  for _, AgentId: string in ipairs(EnvData.Agents) do
    local cCarData: CarsController.CarData? = CarsController.GetCarData(AgentId)
    if cCarData == nil then continue end
    local cCarModel: Model? = cCarData.Model
    if cCarModel == nil or cCarModel == carData.Model then continue end

    local cCarPivot: CFrame = cCarModel:GetPivot()
    local cProgress: number = CheckpointsController.GetTrackProgressState(cCarPivot.Position, cCarPivot.LookVector)
    if cProgress < carTrackProgress and cProgress > followingCarProgress then
      followingCarData = cCarData
      followingCarProgress = cProgress
    end
  end

  if followingCarData == nil then return nil, nil end
  return followingCarData, followingCarProgress
end

CmdHandler.GetReward = function (self: CmdHandler, carData: CarsController.CarData): (number?, number?)
  local carModel: Model? = carData.Model
  if carModel == nil then return end

  if carData.IsTouchingWall then
    if self.WallKill then carData.TerminationPending = true end
    carData.IsTouchingWall = false
  end

  local carCFrame: CFrame = carModel:GetPivot()
  local carPosition = carCFrame.Position
  local carOrientation = carCFrame.LookVector
    
  local trackProgress: number, checkpoint: number, _: number? = CheckpointsController.GetTrackProgressState(carPosition, carOrientation)
  if checkpoint == carData.LastCheckpoint + 1 or (carData.LastCheckpoint == #CheckpointsController.Checkpoints and checkpoint == 1) then
    carData.LastCheckpoint = checkpoint
    if checkpoint == 1 then
      carData.CurrentLap += 1
      carData.LastCheckpoint = 1
    end
  end
  
  local maxLegalProgress: number = CheckpointsController.GetSizeTo(carData.LastCheckpoint + 1) / CheckpointsController.GetTrackSize()
  local currentLegalProgress: number = CheckpointsController.GetSizeTo(carData.LastCheckpoint) / CheckpointsController.GetTrackSize()
  if trackProgress > maxLegalProgress then trackProgress = currentLegalProgress end -- Return to minimal just so the robot does not learn to cheat by reversing
  
  local totalProgress: number = carData.CurrentLap + trackProgress
  if totalProgress > carData.LastBestProgress then
    if math.floor(totalProgress) > math.floor(carData.LastBestProgress) then
      local ct: number = tick()
      self.SumLoopsTime += ct - carData.LastLoopStartTime
      self.AllLoops += 1
      carData.LastLoopStartTime = ct
    end
    carData.LastBestProgress = totalProgress
    carData.LastBestProgressTime = tick()
  elseif tick() - carData.LastBestProgressTime > NoProgressTimeout and carData.FreezeStart == nil then
    carData.TruncationPending = true -- Stuck for too long
  end

  local reward: number = totalProgress - carData.LastReportedProgress
  carData.LastReportedProgress = totalProgress

  reward *= 1000
  if reward < 0 then reward *= 25 end

  -- apply reward based on the distance to the next and following cars
  if self.TerminateAll then
    local numberOfCarsCounted: number = 0
    local totalDistances: number = 0

    -- Next car
    local nextCarData: CarsController.CarData?, nextCarProgress: number? = self:GetNextCar(carData)
    if nextCarData ~= nil and nextCarProgress ~= nil then
      local distance: number = (nextCarProgress - trackProgress) * CheckpointsController.GetTrackSize()
      if carData.PrevNextDistance ~= nil then
        local distanceDiff: number = distance - carData.PrevNextDistance
        totalDistances += -distanceDiff -- Reward / penalize for getting closer to the next car
        numberOfCarsCounted += 1
      end
      carData.PrevNextDistance = distance
    else
      carData.PrevNextDistance = nil
      reward += 1 -- Bonus for maintaining the first position
    end

    -- Following car
    local followingCarData: CarsController.CarData?, followingCarProgress: number? = self:GetFollowingCar(carData)
    if followingCarData ~= nil and followingCarProgress ~= nil then
      local distance: number = (trackProgress - followingCarProgress) * CheckpointsController.GetTrackSize()
      if carData.PrevFollowingDistance ~= nil then
        local distanceDiff: number = distance - carData.PrevFollowingDistance
        totalDistances += distanceDiff -- Reward / penalize for getting farther from the following car
        numberOfCarsCounted += 1
      end
      carData.PrevFollowingDistance = distance
    else
      carData.PrevFollowingDistance = nil
      reward -= 1 -- Penalty for maintaining the last position
    end

    if numberOfCarsCounted > 0 then totalDistances /= numberOfCarsCounted end
    reward += totalDistances * 0.005
  end

  -- non-linear speed bonus (exponential scaling)
  --[[
  local carEngine = carData.CarEngine
  local velocityStatus: number = carEngine:GetAttribute('_speed') :: any
  local maxVelocity: number = carEngine:GetAttribute('forwardMaxSpeed') :: any
  local normalizedSpeed: number = math.clamp(velocityStatus / maxVelocity, 0, 1)
  local speedBonus: number = (math.exp(normalizedSpeed * 2) - 1) * 0.5 -- max ~3.2 at full speed
  reward += speedBonus
  ]]

  if carData.IsTouchingAgent then -- Touching penalty
    if self.EnableEnvCollisions then
      reward -= 60
      carData.TerminationPending = true
      self.AllCollisions += 1
    end
    carData.IsTouchingAgent = false
  end

  if carData.TerminationPending then
    self.AllAgentLoops += totalProgress
    self.AllAgents += 1
  end

  if carData.TruncationPending or carData.TerminationPending then
    return -40, totalProgress
  end

  return reward, totalProgress
end

type DataToSend = {
  obs: { number },
  reward: number,
  terminated: boolean,
  truncated: boolean
}
CmdHandler.Setup = function (self: CmdHandler): ()
  if self.HeartbeatEvent ~= nil then return end -- Already set up

  local runningPrevious: boolean = false
  local totalDt: number = 0 -- Total delta time
  local ConsecutiveUnresponsiveTicks: number = 0
  local sct: number = tick()
  local lastStatisticsPrint: number = sct
  local evaluationStart: number = sct

  self.HeartbeatEvent = RunService.Heartbeat:Connect(function (dt: number): ()
    totalDt += dt
    if totalDt < 1/self.MaxFPS then return end
    if runningPrevious then
      warn('Executed another update before previous finished')
      return
    end
    
    local ct: number = tick()
    if ct > lastStatisticsPrint + 30 then
      print('Average loop time: ', self.SumLoopsTime / math.max(self.AllLoops, 1))
      print('Average loop progress: ', self.AllAgentLoops / math.max(self.AllAgents, 1))
      print('Agents colliding %: ', math.round(self.AllCollisions / self.AllAgents * 100))
      print('Evaluation time: ', ct - evaluationStart)
      lastStatisticsPrint = ct
    end

    dt = totalDt
    totalDt = 0
    runningPrevious = true

    local dataToSend: { [string]: DataToSend? } = {}
    local isAnyCarActive: boolean = false
    local isNoneUpdated: boolean = true

    for EnvId: string, EnvData: EnvData in pairs(self.Envs) do
      local AnyAgentTerminating: boolean = false
      for _, AgentId: string in ipairs(EnvData.Agents) do
        local CarData: CarsController.CarData? = CarsController.GetCarData(AgentId)
        if CarData == nil then continue end
        if CarData.TerminationPending or CarData.TruncationPending then AnyAgentTerminating = true break end
      end
      for _, AgentId: string in ipairs(EnvData.Agents) do
        local CarData: CarsController.CarData? = CarsController.GetCarData(AgentId)
        if CarData == nil then continue end
        if CarData.FreezeStart ~= nil then continue end
        isAnyCarActive = true

        if CarData.IsStateUpdated then isNoneUpdated = false end
        CarData.IsStateUpdated = false

        -- Fetch the reward for last action
        -- It's possible to also fetch the observation before termination here but the parallel framework does not support it anyways (infos > `terminal_observation`)
        local reward: number?, totalProgress: number? = self:GetReward(CarData)
        if reward == nil then continue end
        
        local wasTerminationPending: boolean = CarData.TerminationPending
        local wasTruncationPending: boolean = CarData.TruncationPending
        local isRemoving: boolean = (self.TerminateAll and AnyAgentTerminating) or wasTerminationPending or wasTruncationPending
        if isRemoving and not AnyAgentTerminating then continue end -- Wait for the next one to remove all at once (in case of TerminateAll) to provide a more consistent experience for the learning algorithm
        if wasTerminationPending then
          print(`Car {AgentId} terminated. Total Progress: {totalProgress}`)
        elseif wasTruncationPending then
          print(`Car {AgentId} truncated due to inactivity. Total Progress: {totalProgress}`)
        elseif isRemoving then
          print(`Car {AgentId} removed due to another termination within virtual environment. Total Progress: {totalProgress}`)
        end

        if isRemoving then -- Respawn the car due to termination and/or truncation (so that we provide fresh data)
          CmdHandler['RESET_AGENTS'](self, { agents = { AgentId } } :: ResetData, EnvId)
          CarData = CarsController.GetCarData(AgentId) -- Retrieve fresh data
          if CarData == nil then continue end
        end

        local obs: CarsController.CarObservation = CarsController.RetrieveObservation(AgentId)
        dataToSend[AgentId] = {
          -- last_observation?
          obs = obs,
          -- last_action?
          reward = reward,
          terminated = wasTerminationPending,
          truncated = wasTruncationPending or (self.TerminateAll and AnyAgentTerminating and not wasTerminationPending)
        } :: DataToSend
      end
    end

    if isNoneUpdated then
      if isAnyCarActive then ConsecutiveUnresponsiveTicks += 1 end
    else
      ConsecutiveUnresponsiveTicks = 0
    end
    if ConsecutiveUnresponsiveTicks > 250 then
      warn('No cars have updated their state for a long time, closing env')
      self:Cleanup()
    end

    local success: boolean, result = pcall(function ()
      self.WSClient:Send(HttpService:JSONEncode(dataToSend))
    end)
    if not success then warn('Failed to send data to backend', result) end

    runningPrevious = false
  end)
end

CmdHandler.New = function (wsClient: WebStreamClient, carModel: string): CmdHandler
  local self: CmdHandler = setmetatable({
    Envs = {},
    WSClient = wsClient,
    HeartbeatEvent = nil,
    MaxFPS = 15,

    SumLoopsTime = 0, 
    AllLoops = 0,
    AllAgentLoops = 0,
    AllCollisions = 0,
    AllAgents = 0,

    EnableEnvCollisions = true,
    TerminateAll = false,
    WallKill = true,
    CarModel = carModel
  }, CmdHandler)
  return self
end

type EnvData = {
  Agents: { string },
  Color: Color3?
}

type CmdHandlerData = {
  Envs: { [string]: EnvData? },
  WSClient: WebStreamClient,
  CarModel: string
}

export type CmdHandler = typeof(setmetatable({} :: CmdHandlerData, CmdHandler))

return CmdHandler