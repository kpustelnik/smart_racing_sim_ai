--!strict

export type Point = {
  Position: Vector3,
  Characteristics: number,
  Special: string?,
  Direction: Vector3?,
  Next: Point?,
  Part: BasePart?,
  Bonus: number?
}

return {}