# MERMAID

Python utilities for MERMAID waveform analysis, including TOMOCAT metadata
reading, synthetic arrival generation, waveform stacking, and ocean-depth
grid searches.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

The package modules live under `src/mermaid_depth`.

## Example Imports

```python
from mermaid_depth.misc.read_tomocat1 import read_tomocat1, get_mermaid_data
from mermaid_depth.depth_determination.depth_finder import load_mermaid_sac
from mermaid_depth.depth_determination.waveform_stacking import grid_search_ocean_depth
```

## Running Scripts Directly

```bash
python src/mermaid_depth/depth_determination/waveform_stacking_2d.py
```
