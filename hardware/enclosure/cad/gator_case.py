"""Gator enclosure v4: v2 look, centered tail with a cable channel, support-free printing.

Three printed parts, each designed to print WITHOUT supports:
  base      : floor, walls, legs, lower jaw + teeth, cheek block, centered tail.
              Prints upright. The USB-C plug drops into a pocket at the tail root
              and the cable lies in a channel along the tail between two rows of spikes.
  lid       : flat plate with scutes and eyes. Prints upright (flat side down).
  upper_jaw : snout with teeth. Prints upside down on its flat top; glues onto
              two pegs on the cheek block, angled up so the mouth gapes.
The antenna runs through the throat slot and out the open mouth.
Units mm. X = width, Y = tail -> head, Z = up.
"""
import math
import cadquery as cq

# ---------------- breadboard (edit to your measurements) ----------------
BB_W, BB_L, BB_H = 59.5, 85.5, 10.0   # measured from photo: ~59 x 85 mm (+ margin)
CLR, NUB = 0.6, 2.0
ABOVE = 12.0                   # headroom over the board for XIAO, plug, LED, wires
FRONT_GAP = 6.0                # room at the antenna end for the antenna cable to turn

WALL, FLOOR, LID_T = 2.4, 2.4, 3.5
IN_W = BB_W + 2 * CLR + NUB
IN_L = BB_L + CLR + FRONT_GAP
BB_TOP = FLOOR + BB_H
IN_TOP = BB_TOP + ABOVE        # base walls stop here; the lid sits on top
H = IN_TOP + LID_T
OUT_W = IN_W + 2 * WALL
Y0, YF = -WALL, IN_L + WALL

# tongue (on wall tops) / groove (in lid underside)
TONGUE_W, TONGUE_H = 1.2, 1.6
GROOVE_W, GROOVE_D = 1.8, 2.0

# head
HEAD_L, HINGE = 50.0, 10.0
Y_TIP = YF + HEAD_L
TILT = math.radians(9)         # upper jaw tilt (mouth gape)
JAW_T = 7.5                    # upper jaw thickness
LOW_TIP_Z = 9.0                # lower jaw top height at the tip
LOW_HINGE_Z = IN_TOP - 6.0     # lower jaw top height at the mouth corner

# USB-C / LED exit and tail
SLOT_CX, SLOT_W = 2.0, 22.0
SLOT_Z0 = BB_TOP - 1.5
CH_W = 5.5                     # cable channel width along the tail
TAIL_L = 80.0

BIG = 400


def box(x0, x1, y0, y1, z0, z1):
    return (cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False)
            .translate((x0, y0, z0)))


def side_prism(pts, x0=-BIG / 2, x1=BIG / 2):
    """Polygon in the YZ plane extruded along X from x0 to x1."""
    return (cq.Workplane("YZ").workplane(offset=x0).polyline(pts).close()
            .extrude(x1 - x0))


def snout_plan(y0, y1, z0, z1, inset=0.0):
    """Top-view taper of the head (half width 27 at YF-6 to 13 at the tip)."""
    def hw(y):
        t = (y - (YF - 6)) / (Y_TIP - (YF - 6))
        return 27 * (1 - t) + 13 * t - inset
    return (cq.Workplane("XY").workplane(offset=z0)
            .polyline([(-hw(y0), y0), (hw(y0), y0), (hw(y1), y1), (-hw(y1), y1)]).close()
            .extrude(z1 - z0).edges("|Z").fillet(5))


def snout_hw(y, inset=0.0):
    t = (y - (YF - 6)) / (Y_TIP - (YF - 6))
    return 27 * (1 - t) + 13 * t - inset


def cheek_z(y):
    """Seat plane for the upper jaw: starts at IN_TOP at YF, rises at TILT."""
    return IN_TOP + (y - YF) * math.tan(TILT)


# =====================================================================
# BASE
# =====================================================================
base = box(-OUT_W / 2, OUT_W / 2, Y0, YF, 0, IN_TOP).edges("|Z").fillet(10)

# cheek block + lower jaw (one side profile, cut to the snout plan)
y_h = YF + HINGE
lower = snout_plan(YF - 6, Y_TIP, 0, IN_TOP + 3).intersect(side_prism([
    (YF - 6, 0), (Y_TIP, 0), (Y_TIP, LOW_TIP_Z), (y_h, LOW_HINGE_Z),
    (y_h, cheek_z(y_h)), (YF, IN_TOP), (YF - 6, IN_TOP)]))
base = base.union(lower)

# lower teeth: cones pointing up along both jaw edges
def lerp(y, y0, z0, y1, z1):
    return z0 + (z1 - z0) * (y - y0) / (y1 - y0)

N_TEETH = 6
for i in range(N_TEETH):
    y = y_h + 5 + i * (Y_TIP - y_h - 10) / (N_TEETH - 1)
    zt = lerp(y, y_h, LOW_HINGE_Z, Y_TIP, LOW_TIP_Z)
    for s in (-1, 1):
        tooth = (cq.Workplane("XY").circle(1.7).workplane(offset=4.5).circle(0.35).loft()
                 .translate((s * snout_hw(y, 3.0), y, zt - 1.0)))
        base = base.union(tooth)

# pegs for the upper jaw
PEG_R, PEG_H = 1.5, 3.0
PEGS = [(-17.0, YF + 5.0), (17.0, YF + 5.0)]
for (px, py) in PEGS:
    z = cheek_z(py)
    base = base.union(cq.Workplane("XY").circle(PEG_R).extrude(PEG_H + 1.0)
                      .translate((px, py, z - 1.0)))

# legs (v2 style)
def leg(y, s):
    x_edge = s * OUT_W / 2
    thigh = box(min(x_edge - s * 1, x_edge + s * 11), max(x_edge - s * 1, x_edge + s * 11),
                y - 6, y + 6, 0, 8).edges("|Z").fillet(3)
    for dy in (-4.5, 0, 4.5):
        toe = (cq.Workplane("XY").circle(2.0).extrude(3.5)
               .translate((x_edge + s * 11, y + dy + (2 if y > IN_L / 2 else -2), 0)))
        thigh = thigh.union(toe)
    return thigh

for y in (14, IN_L - 16):
    for s in (-1, 1):
        base = base.union(leg(y, s))

# tail: straight and centered, tapering segments, cable channel down the spine
N_SEG = 10
seg_l = TAIL_L / N_SEG
tail = None
for i in range(N_SEG):
    f = i / (N_SEG - 1)
    w = 26.0 * (1 - f) + 10.0 * f
    h = 18.0 * (1 - f) + 7.0 * f
    yc = Y0 + 2.0 - (i + 0.5) * seg_l
    seg = box(-w / 2, w / 2, yc - seg_l * 0.62, yc + seg_l * 0.62, 0, h)
    seg = seg.edges("|Z").fillet(min(3.0, w / 3.5)).faces(">Z").edges().fillet(1.6)
    tail = seg if tail is None else tail.union(seg)
    # double row of spikes flanking the cable channel
    sh = 4.8 * (1 - f) + 2.2 * f
    for s in (-1, 1):
        fin = (cq.Workplane("YZ").polyline([(-seg_l * 0.38, 0), (seg_l * 0.38, 0),
                                            (-seg_l * 0.10, sh)]).close()
               .extrude(1.6).translate((s * (CH_W / 2 + 0.9) - 0.8, yc, h - 0.6)))
        tail = tail.union(fin)
base = base.union(tail)
Y_TAIL_END = Y0 + 2.0 - TAIL_L - seg_l * 0.2

# ---------------- cuts ----------------
# breadboard pocket and headroom (open top, closed by the lid)
base = base.cut(box(-IN_W / 2, IN_W / 2, 0, IN_L, FLOOR, IN_TOP + 10))
# throat slot from the cavity into the mouth (open top, closed by the upper jaw)
THROAT_W = 22.0
base = base.cut(box(-THROAT_W / 2, THROAT_W / 2, IN_L - 1, y_h + 1, BB_TOP + 0.5, IN_TOP + 20))
# USB-C + LED slot through the rear wall, and a pocket in the tail root for the plug body
base = base.cut(box(SLOT_CX - SLOT_W / 2, SLOT_CX + SLOT_W / 2, Y0 - 13, 1, SLOT_Z0, IN_TOP + 20))
# cable channel along the tail, floor sloping down to the tip
ch_floor_tip = 3.5
channel = side_prism([(Y0 - 12, SLOT_Z0), (Y0 - 12, 60), (Y_TAIL_END - 5, 60),
                      (Y_TAIL_END - 5, ch_floor_tip), (Y0 - 12, SLOT_Z0)],
                     -CH_W / 2, CH_W / 2)
# small chamfer at the slot-to-channel step so the cable bends gently
channel = channel.union(side_prism([(Y0 - 20, SLOT_Z0), (Y0 - 12, SLOT_Z0), (Y0 - 12, 60), (Y0 - 20, 60)],
                                   -CH_W / 2, CH_W / 2))
base = base.cut(channel)

# tongue on top of the walls, following the rounded outline
def band(off_out, off_in, z0, z1):
    """Ring between two inward offsets of the rounded outer outline."""
    def rr(off):
        return (cq.Workplane("XY").workplane(offset=z0)
                .center(0, (Y0 + YF) / 2).rect(OUT_W - 2 * off, (YF - Y0) - 2 * off)
                .extrude(z1 - z0).edges("|Z").fillet(10 - off))
    return rr(off_out).cut(rr(off_in))

tongue = band(WALL / 2 - TONGUE_W / 2, WALL / 2 + TONGUE_W / 2, IN_TOP - 0.5, IN_TOP + TONGUE_H)
tongue = tongue.cut(box(-IN_W / 2, IN_W / 2, 0, IN_L, 0, 60))
tongue = tongue.cut(box(-THROAT_W / 2, THROAT_W / 2, IN_L - 1, YF + 1, 0, 60))
tongue = tongue.cut(box(SLOT_CX - SLOT_W / 2, SLOT_CX + SLOT_W / 2, Y0 - 2, 1, 0, 60))
base = base.union(tongue)

# =====================================================================
# LID (prints upright, flat side down)
# =====================================================================
lid = box(-OUT_W / 2, OUT_W / 2, Y0, YF, IN_TOP, H).edges("|Z").fillet(10)
lid = lid.faces(">Z").edges().fillet(1.8)
groove = band(WALL / 2 - GROOVE_W / 2, WALL / 2 + GROOVE_W / 2, IN_TOP - 1, IN_TOP + GROOVE_D)
lid = lid.cut(groove)

# scutes (pyramids, print fine on top)
def scute(x, y, w, l, h):
    return (cq.Workplane("XY").rect(w, l).workplane(offset=h).rect(0.6, l * 0.5)
            .loft().translate((x, y, H - 0.4)))

for col, (x, w, h) in enumerate([(-20, 6, 2.2), (-7, 7, 3.2), (7, 7, 3.2), (20, 6, 2.2)]):
    y = 4.0 + (col % 2) * 4.0
    while y < IN_L - 14:
        lid = lid.union(scute(x, y, w, 5.5, h))
        y += 8.0

# eyes at the front of the lid, just behind the snout (upper hemispheres only)
for s in (-1, 1):
    ex, ey = s * 15.5, YF - 9.0
    eye = (cq.Workplane("XY").sphere(5.5).rotate((0, 0, 0), (1, 0, 0), 90).translate((ex, ey, H - 0.5))
           .intersect(box(ex - 7, ex + 7, ey - 7, ey + 7, H - 1.0, H + 7)))
    pupil = (cq.Workplane("XY").box(1.3, 2.6, 6).translate((ex + s * 1.2, ey + 3.9, H + 2.6))
             .rotate((ex, ey, H), (ex, ey, H + 1), s * 18))
    lid = lid.union(eye).cut(pupil)

# =====================================================================
# UPPER JAW (prints upside down on its flat top)
# built flat with its underside at IN_TOP, then tilted about the hinge line
# =====================================================================
jaw = snout_plan(YF + 0.3, Y_TIP, IN_TOP, IN_TOP + JAW_T)
jaw = jaw.faces(">Z").chamfer(1.0)
# upper teeth: cones hanging down from the underside (point up when printed)
for i in range(N_TEETH):
    y = y_h + 8 + i * (Y_TIP - y_h - 12) / (N_TEETH - 1)
    for s in (-1, 1):
        tooth = (cq.Workplane("XY").circle(1.8).workplane(offset=-4.8 - (1.2 if i in (1, 4) else 0))
                 .circle(0.35).loft().translate((s * snout_hw(y, 3.2), y, IN_TOP + 1.0)))
        jaw = jaw.union(tooth)
# engraved nostrils on top
for s in (-1, 1):
    jaw = jaw.cut(cq.Workplane("XY").circle(1.3).extrude(3)
                  .translate((s * 4.5, Y_TIP - 6, IN_TOP + JAW_T - 1.6)))
# brow bumps are on the lid; add a low ridge line on the snout top? kept flat for printing
jaw = jaw.rotate((0, YF, IN_TOP), (1, YF, IN_TOP), math.degrees(TILT))
# peg holes (vertical in the world frame, so they line up with the pegs)
for (px, py) in PEGS:
    jaw = jaw.cut(cq.Workplane("XY").circle(PEG_R + 0.25).extrude(PEG_H + 0.8)
                  .translate((px, py, cheek_z(py) - 0.5)))


# =====================================================================
def print_pose_jaw(j):
    """Untilt, flip upside down, drop onto the bed."""
    j = j.rotate((0, YF, IN_TOP), (1, YF, IN_TOP), -math.degrees(TILT))
    j = j.rotate((0, 0, 0), (0, 1, 0), 180)
    bb = j.val().BoundingBox()
    return j.translate((0, 0, -bb.zmin))


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    os.makedirs(out, exist_ok=True)
    parts = {"gator_base": base, "gator_lid": lid, "gator_upper_jaw": jaw}
    for n, p in parts.items():
        v = p.val()
        bb = v.BoundingBox()
        print(n, "%.1f x %.1f x %.1f" % (bb.xlen, bb.ylen, bb.zlen),
              "valid", v.isValid(), "solids", len(p.solids().vals()))
    # assembled STEP for Onshape
    asm = cq.Assembly()
    asm.add(base, name="gator_base", color=cq.Color(0.30, 0.40, 0.27))
    asm.add(lid, name="gator_lid", color=cq.Color(0.34, 0.45, 0.30))
    asm.add(jaw, name="gator_upper_jaw", color=cq.Color(0.34, 0.45, 0.30))
    asm.save(f"{out}/gator_case_v4.step")
    # print-oriented STLs
    cq.exporters.export(base, f"{out}/gator_base_PRINT.stl", tolerance=0.03)
    lid_p = lid.translate((0, 0, -IN_TOP))
    cq.exporters.export(lid_p, f"{out}/gator_lid_PRINT.stl", tolerance=0.03)
    cq.exporters.export(print_pose_jaw(jaw), f"{out}/gator_upper_jaw_PRINT.stl", tolerance=0.03)
    # assembled STLs for previews / checks
    for n, p in parts.items():
        cq.exporters.export(p, f"{out}/{n}_asm.stl", tolerance=0.05)
