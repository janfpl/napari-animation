# Benchmarks

Scripts used to measure interactive section rendering, HDF5 reading and
export performance. Results and write-ups are in
[`docs/dev-notes`](../docs/dev-notes).

Most of them read specific Imaris (`.ims`) and Luxendo (`.lux.h5`) datasets
from paths relative to the folder that contains this repository, so they will
not run as-is elsewhere; adjust the dataset paths at the top of each script.
Scratch output goes to `.performance-audit/` in the repository root, which is
ignored by git.

For runnable examples of the plugin itself, see [`examples`](../examples).
