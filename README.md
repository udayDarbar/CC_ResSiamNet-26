# CC-ResSiamNet: Glacier Velocity Estimation from Multi-Source Satellite Imagery

A PyTorch implementation of a modified **CC-ResSiamNet** (Cross-Correlation Residual Siamese Network) for estimating glacier surface velocity from paired multi-sensor satellite imagery, using ITS_LIVE products as ground truth.

---

## Repository Structure

| Notebook | Description |
|---|---|
| `01_ITSLIVE_velocity_to_npz.ipynb` | Scan, validate, and convert NASA ITS_LIVE (NSIDC-0766 v2) velocity GeoTIFFs to `.npz` |
| `02_Sentinel_GEE_to_npz.ipynb` | Process GEE-exported Sentinel-1/2/3 paired GeoTIFFs to `.npz` |
| `03_CC_ResSiamNet_training.ipynb` | Patch extraction, model definition, training, and inference |

---

## Requirements

### Hardware
- CUDA-capable GPU is strongly recommended for Notebook 3.

### Python Packages
```bash
pip install torch torchvision numpy matplotlib tqdm rasterio scipy
```

---

## Data

### Notebook 1 — ITS_LIVE Velocity (Ground Truth)
- **Source:** [NASA ITS_LIVE](https://its-live.jpl.nasa.gov/) — NSIDC-0766 v2
- **Format:** GeoTIFF, 6/12-day Sentinel-1 SAR mosaics, 200 m resolution, EPSG:3413
- **Bands used:** `vx`, `vy`, `ex`, `ey`
- Set `ITSLIVE_DIR` in the config cell to your local NSIDC-0766 directory.

### Notebook 2 — Sentinel Imagery
- **Source:** Google Earth Engine (GEE) exports
- **Sensors:** Sentinel-1 (SAR), Sentinel-2 (optical), Sentinel-3 (OLCI)
- **Format:** `PAIR_YYYYMMDD_YYYYMMDD.tif`, 200 m, EPSG:3413, 26 bands per timestep
- Set `GEE_DIR` in the config cell to your local GEE export directory.

---

## Pipeline

```
NSIDC-0766 GeoTIFFs  -->  Notebook 1  -->  processed_velocity_npz/
GEE PAIR_*.tif       -->  Notebook 2  -->  processed_sentinel_npz/
                                    |
                                    v
                              Notebook 3  -->  patches_npy/      (training patches)
                                          -->  model_outputs/    (checkpoints)
                                          -->  figures_training/ (plots)
```

---

## Model Architecture

CC-ResSiamNet is a 6-depth Siamese U-Net with cross-connections between branches:

- **Shared Encoder:** Paired residual blocks with cross-branch feature fusion at each scale
- **Channel Attention:** SE-block style squeeze-and-excitation per branch
- **Spatial Attention:** Convolutional gating over channel-pooled feature maps
- **Decoder:** Cross-connected upsampling path with skip connections
- **Loss:** Masked Huber loss, applied only at pixels with valid ITS_LIVE observations

The architecture accepts an arbitrary number of input channels (`in_ch`), making it compatible with any subset of the 26 Sentinel bands.

---

## Large-Data Design

The pipeline is built to handle datasets exceeding 100 GB without loading data into RAM:

- **Memory-mapped arrays:** All patch `.npy` files are accessed via `numpy.load(mmap_mode='r')`
- **Streaming patchification:** Two-pass approach — count valid patches, then write directly to disk
- **Per-pair velocity matching:** Each Sentinel pair is matched to its closest ITS_LIVE velocity cycle by temporal overlap before patch extraction

---

## Output Directory Layout

```
.
├── 01_ITSLIVE_velocity_to_npz.ipynb
├── 02_Sentinel_GEE_to_npz.ipynb
├── 03_CC_ResSiamNet_training.ipynb
├── README.md
├── processed_velocity_npz/    # Notebook 1 outputs (not tracked)
├── processed_sentinel_npz/    # Notebook 2 outputs (not tracked)
├── patches_npy/               # Training patches (not tracked)
├── model_outputs/             # Model checkpoints & predictions (not tracked)
└── figures_training/          # Training plots (not tracked)
```

---

## Citation

> Citation will be added upon publication.

---

## License

> License to be added.
