# Gator enclosure

A 3D-printed alligator case for each ESP32-S3 node (one sender, two receivers).
The half-size breadboard sits inside the body, the antenna comes out of the open
mouth, and the USB-C cable runs out through a channel along the tail.

```
hardware/enclosure/
├── web/gator_case.glb     ← for the dashboard (3D model, colored, ~1.3 MB)
├── print/                 ← STL files, already oriented for printing
│   ├── gator_base.stl
│   ├── gator_lid.stl
│   └── gator_upper_jaw.stl
└── cad/
    ├── gator_case.step    ← editable CAD (Onshape, Fusion, SolidWorks…)
    └── gator_case.py      ← CadQuery script that generates everything
```

## Dashboard: showing the model

`web/gator_case.glb` is a standard glTF binary file (Y-up, metres), so any
web 3D viewer can load it. The simplest option is Google's `<model-viewer>`,
which needs no build step:

```html
<script type="module"
  src="https://cdn.jsdelivr.net/npm/@google/model-viewer/dist/model-viewer.min.js"></script>

<model-viewer src="../hardware/enclosure/web/gator_case.glb"
              alt="Gator sensor enclosure"
              camera-controls auto-rotate
              style="width: 100%; height: 320px;"></model-viewer>
```

Adjust the `src` path to wherever the page is served from. With three.js, use
`GLTFLoader`. The model has three named parts: `gator_base`, `gator_lid` and
`gator_upper_jaw`. You can recolor or highlight them individually, for example
to tint a node red when it detects a fall.

## Printing

All three parts print **without supports**. Put the STL files on the bed exactly
as they load:

| Part | Orientation | Notes |
|---|---|---|
| `gator_base.stl` | floor down | legs, tail, lower jaw and teeth; 94 × 226 × 28 mm, so place it diagonally on smaller beds |
| `gator_lid.stl` | flat side down | 68 × 97 × 9 mm; scutes and eyes face up |
| `gator_upper_jaw.stl` | flat top down | 48 × 50 × 16 mm; teeth point up while printing |

PLA, 0.2 mm layers and 15% infill work fine. The only overhangs are bridges
under 2 mm wide.

**Print one set first and check the fit.** You need three sets: one per board.

## Assembly

1. Press the breadboard into the base, with the USB-C end toward the tail and pushed
   against the tail wall. Its adhesive backing can hold it in place.
2. Plug in the USB-C cable. The plug and LED drop into the pocket at the tail
   root, and the cable lies in the channel down the tail.
3. Route the antenna through the throat slot and out of the mouth. Keep metal
   away from the antenna.
4. Push the upper jaw onto the two pegs on the head and add a drop of glue.
5. Snap the lid on. It locates on a tongue-and-groove rim and lifts off again.

## Changing the size

The pocket is **62.7 × 92.1 mm** inside, sized for our breadboard (about
59 × 85 mm with the power rails attached; up to 10 mm tall). The extra length is
room for the antenna cable at the head end, so push the board back against the
tail end. The USB-C plug body must be no wider than 22 mm. If yours differ, edit the values at the top of
`cad/gator_case.py` and regenerate:

```bash
pip install cadquery
python hardware/enclosure/cad/gator_case.py   # writes to hardware/enclosure/cad/build/
```

The Onshape copy is `gator_case_v4.step`. In Onshape it imports as three parts.
