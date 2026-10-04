--!strict
-- FIN
return function (from: Vector3, to: Vector3, pos: Vector3): Vector3
  local span: Vector3 = to - from
  local length: number = span.Magnitude
  if length < 1e-4 then return from end

  local t: number = math.clamp((pos - from):Dot(span) / (length * length), 0, 1)
  return from + span * t
end
