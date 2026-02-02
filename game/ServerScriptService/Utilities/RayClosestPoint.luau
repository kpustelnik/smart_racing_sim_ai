--!strict
-- FIN
return function (from: Vector3, to: Vector3, pos: Vector3): Vector3
  local R: Ray = Ray.new(from, to - from).Unit
  local CP: Vector3 = R:ClosestPoint(pos)
  if (CP - from).Magnitude > (to - from).Magnitude then CP = to end
  return CP
end
