--!strict

local Types = require(script.Parent.Parent:WaitForChild("types"))
local FinalizerCommon = require(script.Parent.Parent:WaitForChild("FinalizerCommon"))

local ServerStorage = game:GetService('ServerStorage')
local ServerScriptService = game:GetService("ServerScriptService")

local Workspace = game:GetService('Workspace')
local Terrain: Terrain = Workspace.Terrain

local TrackBase: Model = ServerStorage:WaitForChild('TrackBase') :: Model
local axis: Vector3 = Vector3.new(0, 0, 1)

local Utilities: Folder = ServerScriptService:WaitForChild("Utilities")
local Spline = require(Utilities:WaitForChild("Spline"))
local BoundingBox = require(Utilities:WaitForChild("BoundingBox"))

local Redupe = Utilities:WaitForChild("redupe")
local createGhostPreview = require(Redupe:WaitForChild("createGhostPreview"))
local bendPlacement = require(Redupe:WaitForChild("bendPlacement"))
local resizeAlignPairs = require(Redupe:WaitForChild("resizeAlignPairs"))

local probeSpacing: number = 0.5

local AsphaltMargin: Vector3 = Vector3.new(7, 27, 3)
local AirMargin: Vector3 = Vector3.new(9, 36, 5)

local MaxTileTurn: number = math.rad(9)

-- Arc length the turn rate is averaged over
local TurnWindow: number = 2

local UP: Vector3 = Vector3.new(0, 1, 0)
local function slopeLook(from: Vector3, towards: Vector3): CFrame
  local direction: Vector3 = towards - from
  if direction.Magnitude < 1e-4 then return CFrame.new(from) end
  direction = direction.Unit
  if math.abs(direction.Y) > 0.999 then return CFrame.new(from) end
  return CFrame.lookAt(from, from + direction, UP)
end

local function carveTerrain(cframe: CFrame, size: Vector3): ()
  Terrain:FillBlock(cframe * CFrame.new(0, -15, 0), size + AsphaltMargin, Enum.Material.Asphalt)
  Terrain:FillBlock(cframe * CFrame.new(0, 15, 0), size + AirMargin, Enum.Material.Air)
end

type Basis = {
  CFrame: CFrame,
  Offset: Vector3,
  Size: Vector3,
}

return function (path: { Types.Point }, pathFolder: Folder): ()
  local createCheckpoint = FinalizerCommon.checkpointWriter()

  local prev: { Instance }? = nil
  local prevBasis: Basis? = nil
  for i: number, curPoint: Types.Point in ipairs(path) do
    local nextPoint: Types.Point? = if i ~= #path then path[i + 1] else nil
    if curPoint.Next == nextPoint then -- Check if it's a skip connection (via structure)
      prev = nil
      continue
    end
    if nextPoint ~= nil then
      local span: FinalizerCommon.Span = FinalizerCommon.span(path, i, slopeLook)
      local segment: Spline.Segment = span.Segment
      local startCF: CFrame = span.StartCFrame
      local endCF: CFrame = span.EndCFrame

      local cTrackBase: Model = TrackBase:Clone()
      local TrackBaseTargets: { Instance } = { cTrackBase }
      local TrackBaseParts: { BasePart } = {}
      for _, part: Instance in ipairs(cTrackBase:GetDescendants()) do
        if part:IsA("BasePart") then table.insert(TrackBaseParts, part) end
      end

      cTrackBase.Parent = pathFolder
      cTrackBase:PivotTo(startCF)
      createCheckpoint(startCF)
      if span.EndTangent ~= nil then createCheckpoint(endCF) end
  
      local localBoundingBoxOffset: Vector3, localBoundingBoxSize: Vector3 = BoundingBox.compute(startCF, TrackBaseParts)
      local center: CFrame = startCF
      local boundsOffset: Vector3 = localBoundingBoxOffset
      local size: Vector3 = localBoundingBoxSize
      -- center *= CFrame.new(boundsOffset)
      boundsOffset = Vector3.zero

      local ghostPreview = createGhostPreview(TrackBaseTargets, startCF, localBoundingBoxSize)
      ghostPreview.hide()

      local sizeOnAxis: number = (axis * size).Magnitude

      local placements: { bendPlacement.Placement } = {} :: { bendPlacement.Placement }

      local deltaSize: Vector3 = Vector3.new() --[[ TODO: May adjust the end size ]]
      -- TODO: End Delta position (offset from the main axis may also be put here)

      local probedPoints: { Vector3 }, t1: number = Spline.sampleSegment(segment, probeSpacing, 0)

      -- Arc length from the start of the segment to each probed point.
      local travelled: { number } = Spline.arcLengths(probedPoints)
      local segmentLength: number = travelled[t1]

      local probeCursor: number = 1
      local function pointAtDistance(distance: number): Vector3
        distance = math.clamp(distance, 0, segmentLength)
        while probeCursor < t1 - 1 and travelled[probeCursor + 1] < distance do
          probeCursor += 1
        end
        while probeCursor > 1 and travelled[probeCursor] > distance do
          probeCursor -= 1
        end
        local gap: number = travelled[probeCursor + 1] - travelled[probeCursor]
        local alpha: number = if gap > 1e-4 then (distance - travelled[probeCursor]) / gap else 0
        return probedPoints[probeCursor]:Lerp(probedPoints[probeCursor + 1], math.clamp(alpha, 0, 1))
      end

      local function frameAtDistance(distance: number): CFrame
        local here: Vector3 = pointAtDistance(distance)
        local ahead: Vector3 = pointAtDistance(math.min(distance + probeSpacing, segmentLength))
        local behind: Vector3 = pointAtDistance(math.max(distance - probeSpacing, 0))
        return slopeLook(here, here + (ahead - behind))
      end

      local segmentTurn: number = 0
      local peakTurnRate: number = 0
      do
        local window: number = math.max(1, math.round(TurnWindow / probeSpacing))
        local previousDirection: Vector3? = nil
        for j = 1 + window, t1, window do
          local step: Vector3 = probedPoints[j] - probedPoints[j - window]
          if step.Magnitude > 1e-4 then
            local direction: Vector3 = step.Unit
            if previousDirection ~= nil then
              local turn: number = math.acos(math.clamp(previousDirection:Dot(direction), -1, 1))
              segmentTurn += turn
              peakTurnRate = math.max(peakTurnRate, turn / (window * probeSpacing))
            end
            previousDirection = direction
          end
        end
      end

      local byLength: number = math.round(segmentLength / math.max(sizeOnAxis, 1e-3))
      local byTurn: number = math.ceil(segmentTurn / MaxTileTurn)
      local byPeak: number = if peakTurnRate > 1e-6
        then math.ceil(segmentLength / (MaxTileTurn / peakTurnRate))
        else 0
      local byContact: number = math.max(1, math.floor(segmentLength / math.max(sizeOnAxis, 1e-3)))
      local tileCount: number = math.clamp(math.max(byLength, byTurn, byPeak, 1), 1, 512)
      tileCount = math.min(tileCount, byContact)
      local tilePitch: number = segmentLength / tileCount

      local previousCFrame: CFrame = startCF
      for slot = 1, tileCount do
        local placeCFrame: CFrame = if slot >= tileCount then endCF else frameAtDistance(slot * tilePitch)
        local placement: bendPlacement.Placement = {
          BoundsOffset = boundsOffset,
          Size = localBoundingBoxSize + (deltaSize * slot / tileCount),
          PreviousSize = if slot > 1 then placements[slot - 1].Size else localBoundingBoxSize,
          Offset = previousCFrame:Inverse() * placeCFrame,
          CFrame = placeCFrame,
        }
        table.insert(placements, placement)
        previousCFrame = placeCFrame
      end

      -- Place the copies using offsets
      local runningPosition = startCF
      local lastCopy = TrackBaseTargets
      local lastBasis = {
        CFrame = center,
        Offset = boundsOffset,
        Size = localBoundingBoxSize,
      }
      
      local resultsPerTarget = {}
      do for k = 1, #TrackBaseTargets do resultsPerTarget[k] = {} end end
      local resizeAlignPairsCache = {}

      for _, placement in placements do
        runningPosition *= placement.Offset
        runningPosition = runningPosition:Orthonormalize()
        
        local thisCopy = ghostPreview.create(false, "Extrude", runningPosition, size, axis)
        local thisInfo = {
          CFrame = runningPosition,
          Offset = boundsOffset,
          Size = size,
        }
        carveTerrain(thisInfo.CFrame, size)

        local resizeAlignResults = resizeAlignPairs(lastCopy, thisCopy, lastBasis, thisInfo, axis, resizeAlignPairsCache)
        for j, resizeAlignResult in resizeAlignResults do
          for _, resultInstance in resizeAlignResult do
            table.insert(resultsPerTarget[j], resultInstance)
          end
        end

        for j, resultInstance in thisCopy do table.insert(resultsPerTarget[j], resultInstance) end
        lastCopy = thisCopy
        lastBasis = thisInfo
      end

      if prev ~= nil and prevBasis ~= nil then
        local firstCopy = TrackBaseTargets
        local firstBasis = {
          CFrame = center,
          Offset = boundsOffset,
          Size = size,
        }
        carveTerrain(prevBasis.CFrame, size)
        local alignResults = resizeAlignPairs(prev, firstCopy, prevBasis, firstBasis, axis, resizeAlignPairsCache)

        for j, resizeAlignResult in alignResults do
          for _, resultInstance in resizeAlignResult do
            table.insert(resultsPerTarget[j], resultInstance)
          end
        end
      end
      prevBasis = lastBasis
      prev = lastCopy

      ghostPreview.trim()
    end
  end
end
