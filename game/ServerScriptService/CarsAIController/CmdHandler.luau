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

local TerminationReward: number = -40
local TruncationReward: number = 0
local AgentCollisionPenalty: number = -60

-- Creates a new environment
type CreateEnvData = {}
CmdHandler['CREATE_ENV'] = function (self: CmdHandler, _data: CreateEnvData, envId: string?): ()
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
CmdHandler['UPDATE_COLLISIONS'] = function (self: CmdHandler, data: UpdateCollisionsData, _envId: string?): ()
  local enableCollisions: boolean? = data.enable_collisions
  if enableCollisions ~= nil then
    self.EnableEnvCollisions = enableCollisions
  end
  for cEnvId: string, _: EnvData? in pairs(self.Envs) do
    local CollisionGroupName: string = CarsController.GetEnvCollisionGroup(cEnvId)
    local suc: boolean, res = pcall(function()
      PhysicsService:CollisionGroupSetCollidable(CollisionGroupName, CollisionGroupName, self.EnableEnvCollisions)
    end)
    if not suc then
      warn(`Failed to update collision settings for env {cEnvId}: {res}`)
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
  local envData: EnvData? = self.Envs[envId]
  if envData == nil then return end

  local currentAgentModels: { Instance } = {}
  for _, agentId: string in ipairs(envData.Agents) do
    local carData: CarsController.CarData? = CarsController.GetCarData(agentId)
    if carData ~= nil then
      local carModel: Model? = carData.Model
      if carModel ~= nil then table.insert(currentAgentModels, carModel) end
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
CmdHandler['CLOSE'] = function (self: CmdHandler, _data: CloseData, envId: string?): ()
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
CmdHandler['CLOSE_ALL'] = function (self: CmdHandler, _data: CloseAllData, _envId: string?): ()
  for cEnvId: string in pairs(self.Envs) do
    CmdHandler['CLOSE'](self, {} :: CloseData, cEnvId)
  end
end

type ActionData = { [string]: any }
CmdHandler['ACTION'] = function (self: CmdHandler, data: ActionData, _envId: string?): ()
  local seq: number? = tonumber(data.__seq)
  for agentId: string, agentData in pairs(data) do
    if string.sub(agentId, 1, 2) == '__' then continue end -- Control fields
    local carData: CarsController.CarData? = CarsController.GetCarData(agentId)
    if carData == nil then continue end

    if typeof(agentData) ~= 'table' then continue end
    local throttle: number? = tonumber(agentData[1])
    local steering: number? = tonumber(agentData[2])
    local nitro: number? = tonumber(agentData[3])
    if throttle == nil or steering == nil or nitro == nil then
      warn(`Malformed action for agent {agentId}`)
      continue
    end

    carData.CarInputs:SetAttribute('throttleInput', throttle)
    carData.CarInputs:SetAttribute('steeringInput', steering)
    carData.CarInputs:SetAttribute('nitroInput', nitro > 0.5)
    carData.IsStateUpdated = true
    carData.LastAction = { throttle, steering, nitro }
    carData.LastActionSeq = seq

    -- Update car physics
    local lastUpdateTime: number? = carData.LastUpdateTime
    if lastUpdateTime ~= nil then
      local dt: number = math.clamp(tick() - lastUpdateTime, 0, 4 / self.MaxFPS)
      carData.CarController:update(dt)
    end
    carData.LastUpdateTime = tick()
  end
end

type FreezeData = {
  status: boolean
}
CmdHandler['FREEZE'] = function (self: CmdHandler, data: FreezeData, _envId: string?): ()
  for _: string, envData: EnvData? in pairs(self.Envs) do
    if envData == nil then continue end
    for _, agentId: string in ipairs(envData.Agents) do
      local carData: CarsController.CarData? = CarsController.GetCarData(agentId)
      if carData == nil then continue end

      -- Anchor / unanchor the car model
      local carModel: Model? = carData.Model
      if carModel ~= nil then
        if data.status then
          local saved: { [BasePart]: { Linear: Vector3, Angular: Vector3 } } = {}
          for _, part: Instance in ipairs(carModel:GetDescendants()) do
            if part:IsA('BasePart') and not part.Anchored then
              saved[part] = { Linear = part.AssemblyLinearVelocity, Angular = part.AssemblyAngularVelocity }
            end
          end
          if carData.FrozenVelocities == nil then carData.FrozenVelocities = saved end

          for _, part: Instance in ipairs(carModel:GetDescendants()) do
            if part:IsA('BasePart') then part.Anchored = true end
          end
        else
          for _, part: Instance in ipairs(carModel:GetDescendants()) do
            if part:IsA('BasePart') then part.Anchored = false end
          end

          local frozen = carData.FrozenVelocities
          if frozen ~= nil then
            for part: BasePart, velocity in pairs(frozen) do
              if part.Parent ~= nil then
                part.AssemblyLinearVelocity = velocity.Linear
                part.AssemblyAngularVelocity = velocity.Angular
              end
            end
            carData.FrozenVelocities = nil
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

type PingData = { t: number? }
CmdHandler['PING'] = function (self: CmdHandler, data: PingData, _envId: string?): ()
  local sentAt: number? = if typeof(data) == 'table' then data.t else nil
  if sentAt == nil then return end

  local success: boolean, result = pcall(function()
    self.WSClient:Send(HttpService:JSONEncode({ __pong = { t = sentAt } }))
  end)
  if not success then warn('Failed to answer ping', result) end
end

local AllowedCommands: { [string]: boolean } = {
  CREATE_ENV = true,
  UPDATE_COLLISIONS = true,
  SPAWN_AGENTS = true,
  RESET_AGENTS = true,
  REMOVE_AGENTS = true,
  CLOSE = true,
  CLOSE_ALL = true,
  ACTION = true,
  FREEZE = true,
  PING = true,
}

CmdHandler.Handle = function (self: CmdHandler, msg: BackendMessage): ()
  local command: string = msg.command
  local data: any = msg.data
  local envid: string? = msg.envid

  if typeof(command) ~= 'string' or not AllowedCommands[command] then
    warn(`Unknown command received: {command}`)
    return
  end

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
  CarsController.DisconnectWallEvents()
  if self.HeartbeatEvent ~= nil then
    self.HeartbeatEvent:Disconnect()
    self.HeartbeatEvent = nil
  end
end

CmdHandler.GetCarEnv = function (self: CmdHandler, carData: CarsController.CarData): string?
  for EnvId: string, EnvData: EnvData? in pairs(self.Envs) do
    if EnvData == nil then continue end
    for _, AgentId: string in ipairs(EnvData.Agents) do
      local cCarData: CarsController.CarData? = CarsController.GetCarData(AgentId)
      if cCarData == nil then continue end
      if cCarData == carData then return EnvId end
    end
  end
  return nil
end

-- Closest car of the same environment that is ahead of the given one on the track, or behind it
local function findNeighbourCar(self: CmdHandler, carData: CarsController.CarData, ahead: boolean): (CarsController.CarData?, number?)
  local EnvId: string? = self:GetCarEnv(carData)
  if EnvId == nil then return nil, nil end
  local EnvData: EnvData? = self.Envs[EnvId]
  if EnvData == nil then return nil, nil end

  local carModel: Model? = carData.Model
  if carModel == nil then return nil, nil end
  local carPivot: CFrame = carModel:GetPivot()
  local carTrackProgress: number = CheckpointsController.GetTrackProgressState(carPivot.Position, carPivot.LookVector)

  local neighbourCarData: CarsController.CarData? = nil
  local neighbourCarProgress: number = if ahead then math.huge else 0
  for _, AgentId: string in ipairs(EnvData.Agents) do
    local cCarData: CarsController.CarData? = CarsController.GetCarData(AgentId)
    if cCarData == nil then continue end
    local cCarModel: Model? = cCarData.Model
    if cCarModel == nil or cCarModel == carData.Model then continue end

    local cCarPivot: CFrame = cCarModel:GetPivot()
    local cProgress: number = CheckpointsController.GetTrackProgressState(cCarPivot.Position, cCarPivot.LookVector)
    local isCloser: boolean = if ahead
      then cProgress > carTrackProgress and cProgress < neighbourCarProgress
      else cProgress < carTrackProgress and cProgress > neighbourCarProgress
    if isCloser then
      neighbourCarData = cCarData
      neighbourCarProgress = cProgress
    end
  end

  if neighbourCarData == nil then return nil, nil end
  return neighbourCarData, neighbourCarProgress
end

CmdHandler.GetNextCar = function (self: CmdHandler, carData: CarsController.CarData): (CarsController.CarData?, number?)
  return findNeighbourCar(self, carData, true)
end

CmdHandler.GetFollowingCar = function (self: CmdHandler, carData: CarsController.CarData): (CarsController.CarData?, number?)
  return findNeighbourCar(self, carData, false)
end

CmdHandler.GetReward = function (self: CmdHandler, carData: CarsController.CarData): (number?, number?)
  local carModel: Model? = carData.Model
  if carModel == nil then return end

  if carData.IsTouchingWall then
    if self.WallKill then carData.TerminationPending = true end
    carData.IsTouchingWall = false
  end

  local carCFrame: CFrame = carModel:GetPivot()
  local carPosition: Vector3 = carCFrame.Position
  local carOrientation: Vector3 = carCFrame.LookVector
    
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
  elseif tick() - carData.LastBestProgressTime > self.NoProgressTimeout and carData.FreezeStart == nil then
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

  local terminalPenalty: number = 0
  if carData.IsTouchingAgent then -- Touching another agent ends the episode
    if self.EnableEnvCollisions then
      terminalPenalty += AgentCollisionPenalty
      carData.TerminationPending = true
      self.AllCollisions += 1
    end
    carData.IsTouchingAgent = false
  end

  if carData.TerminationPending then
    self.AllAgentLoops += totalProgress
    self.AllAgents += 1
  end

  if carData.TerminationPending then
    return TerminationReward + terminalPenalty, totalProgress
  end
  if carData.TruncationPending then
    return TruncationReward + terminalPenalty, totalProgress
  end

  return reward, totalProgress
end

type DataToSend = {
  obs: { number },
  last_observation: { number }?,
  reward: number,
  terminated: boolean,
  truncated: boolean,
  action_seq: number?
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
    if self.WSClient.ConnectionState == Enum.WebStreamClientState.Closed then
      warn('Simulation loop outlived its websocket; shutting it down.')
      self:Cleanup()
      return
    end

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
      print('Agents colliding %: ', math.round(self.AllCollisions / math.max(self.AllAgents, 1) * 100))
      print('Evaluation time: ', ct - evaluationStart)
      lastStatisticsPrint = ct
    end

    dt = totalDt
    totalDt = 0
    runningPrevious = true

    local tickOk: boolean, tickError: any = pcall(function(): ()
      local dataToSend: { [string]: DataToSend? } = {}
      local isAnyCarActive: boolean = false
      local isNoneUpdated: boolean = true

      for EnvId: string, EnvData: EnvData? in pairs(self.Envs) do
        if EnvData == nil then continue end
        local AnyAgentTerminating: boolean = false
        for _, AgentId: string in ipairs(EnvData.Agents) do
          local CarData: CarsController.CarData? = CarsController.GetCarData(AgentId)
          if CarData == nil then continue end
          if CarData.TerminationPending or CarData.TruncationPending then AnyAgentTerminating = true; break end
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

          local finalObservation: CarsController.CarObservation? = if isRemoving then CarData.LastObservation else nil

          if isRemoving then -- Respawn the car due to termination and/or truncation (so that we provide fresh data)
            CmdHandler['RESET_AGENTS'](self, { agents = { AgentId } } :: ResetData, EnvId)
            CarData = CarsController.GetCarData(AgentId) -- Retrieve fresh data
            if CarData == nil then continue end
          end

          local obs: CarsController.CarObservation? = CarsController.RetrieveObservation(AgentId)
          if obs == nil then continue end
          dataToSend[AgentId] = {
            obs = obs,
            last_observation = finalObservation,
            reward = reward,
            terminated = wasTerminationPending,
            truncated = wasTruncationPending or (self.TerminateAll and AnyAgentTerminating and not wasTerminationPending),
            action_seq = CarData.LastActionSeq
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
    end)
    if not tickOk then warn('Simulation tick failed:', tickError) end

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
    NoProgressTimeout = 5,
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
  CarModel: string,

  HeartbeatEvent: RBXScriptConnection?,
  MaxFPS: number,

  SumLoopsTime: number,
  AllLoops: number,
  AllAgentLoops: number,
  AllCollisions: number,
  AllAgents: number,

  EnableEnvCollisions: boolean,
  TerminateAll: boolean,
  WallKill: boolean,
  NoProgressTimeout: number
}

export type CmdHandler = typeof(setmetatable({} :: CmdHandlerData, CmdHandler))

return CmdHandler