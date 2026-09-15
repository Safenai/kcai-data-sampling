# comma10k sample: ten driving frames

Ten forward-camera frames from [comma10k](https://github.com/commaai/comma10k)
with their segmentation masks, chosen for having vehicles in the scene. This
directory is git-ignored; `python scripts/fetch_comma10k_sample.py` fills it
(commit `6c205fe4c43cc53b2b1befafb1060d0606555027`, `imgs2/` and `masks2/`).

**Licence: MIT.** Copyright (c) 2020, Comma.ai, Inc.; masks labelled by the
public. Attribution: comma10k, Comma.ai, Inc. and its contributors.

```
comma10k_sample/
├── imgs/     *.png   1928×1208 RGB, forward camera
└── masks/    *.png   same name, colour-coded segmentation
```

Mask colours, from the comma10k README:

| colour | class |
| :--- | :--- |
| `#402020` | road |
| `#ff0000` | lane markings |
| `#808060` | undrivable |
| `#00ff66` | movable (vehicles, people, anything that moves) |
| `#cc00ff` | my car |
