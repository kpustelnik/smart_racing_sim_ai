--!strict

export type Poixnt = {
  Position: Vector3,
  Characteristics: number,
  Special: string?,
  Direction: Vector3?,
  Next: Point?,
  Bonus: number?,
  
  -- Roblox only
  Part: BasePart?,
  Collider: BasePart?,
  Object: Instance?
}

return {}