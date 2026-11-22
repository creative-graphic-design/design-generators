import Lean

/-!
Small, exact-integer surrogate of the existing bbox conversion algebra.

The Python functions operate on floating-point xywh / ltrb coordinates.
Here x, y, width, and height are integer-valued; the ltrb edges are
represented in doubled units to express half-width offsets exactly.
These theorems do not establish PyTorch or floating-point correctness.
-/

namespace BBoxPilot

structure XYWH where
  x : Int
  y : Int
  w : Int
  h : Int

structure LTRB2 where
  left : Int
  top : Int
  right : Int
  bottom : Int

def xywhToLTRB2 (b : XYWH) : LTRB2 :=
  { left := 2 * b.x - b.w
    top := 2 * b.y - b.h
    right := 2 * b.x + b.w
    bottom := 2 * b.y + b.h }

def ltrb2ToXYWH (b : LTRB2) : XYWH :=
  { x := (b.left + b.right) / 4
    y := (b.top + b.bottom) / 4
    w := (b.right - b.left) / 2
    h := (b.bottom - b.top) / 2 }

theorem xywh_roundtrip (b : XYWH) :
    ltrb2ToXYWH (xywhToLTRB2 b) = b := by
  cases b with
  | mk x y w h =>
    dsimp [xywhToLTRB2, ltrb2ToXYWH]
    congr 1 <;> omega

theorem width_of_forward (b : XYWH) :
    (xywhToLTRB2 b).right - (xywhToLTRB2 b).left = 2 * b.w := by
  dsimp [xywhToLTRB2]
  omega

theorem height_of_forward (b : XYWH) :
    (xywhToLTRB2 b).bottom - (xywhToLTRB2 b).top = 2 * b.h := by
  dsimp [xywhToLTRB2]
  omega

end BBoxPilot
