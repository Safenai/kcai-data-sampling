# WoodScape sample: put the files here

This directory is **git-ignored**, and **optional**: the tests and the notebook
run on the comma10k sample next door (`scripts/fetch_comma10k_sample.py`). WoodScape is the real fisheye
case, for whoever has the licence.

WoodScape's *code* is MIT; its *data* is proprietary and distributed through a
Google Drive folder linked from
[`valeoai/WoodScape`](https://github.com/valeoai/WoodScape). Whoever accepts
that licence downloads the files and drops them in.

Ten images is plenty. Only `rgb_images/` is required, the loader reads
whatever else it finds and skips the rest.

```
woodscape_sample/
├── rgb_images/rgb_images/                                   *.png   1280×966 RGB
├── box_2d_annotations/box_2d_annotations/box_2d_annotations/ *.txt   name,index,left,top,right,bottom
├── instance_annotations/…/instance_annotations/             *.json  polygons per instance
├── calibration/…/calibration/                               *.json  intrinsics (pixels) + extrinsics
├── semantic_annotations/                                    *.png   optional, not fetched yet
├── motion_annotations/                                      *.png   optional, not fetched yet
└── previous_images/                                         *.png   optional, not fetched yet
```

That is how the Drive folders unzip, nested under their own name, the
annotations one level deeper than the images. The loader descends as far as a
same-named child exists, so a flat layout works too.

Keep the stems aligned: `00001_FV.png` in `rgb_images/` pairs with
`00001_FV.txt` in `box_2d_annotations/`, and so on.

`box_2d_annotations/` is **derived** from `instance_annotations/` by
`scripts/box_2d_generator.py` in the WoodScape repository; precomputed boxes
are also published in a separate Drive folder linked from its README.
