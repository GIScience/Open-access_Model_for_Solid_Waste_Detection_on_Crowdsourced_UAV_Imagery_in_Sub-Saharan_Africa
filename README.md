# Open-access Model for Solid Waste Detection on Crowdsourced UAV Imagery in Sub-Saharan Africa

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Model](https://img.shields.io/badge/Model-YOLO11x--cls-orange)](https://docs.ultralytics.com/models/yolo11/)
[![DOI](https://zenodo.org/badge/1204752062.svg)](https://doi.org/10.5281/zenodo.19609437)

A reproducible deep-learning pipeline for detecting and quantifying solid waste in urban and peri-urban settlements across Sub-Saharan Africa, using crowdsourced UAV imagery from [OpenAerialMap (OAM)](https://openaerialmap.org/).

A fine-tuned **YOLO11x-cls** classifier is applied to a 5 m × 5 m tile grid over each study region. The fraction of tiles predicted as *waste* is reported as the **Relative Solid Waste Contamination Index (RSWCI)**. Each region is further characterised by street-network complexity and population density, both derived from the [MillionNeighborhoods Africa dataset](https://www.millionneighborhoods.africa/).

**Study coverage:** 29 regions across 10 Sub-Saharan African countries.

---

## Abstract

Managing solid waste in rapidly urbanizing Sub-Saharan Africa remains challenging due to dispersed informal dumping and limited high-resolution datasets for spatial monitoring. We present an open-access deep learning model for automated solid waste detection on crowdsourced UAV imagery, trained and evaluated across 29 regions in 10 countries, encompassing diverse environmental contexts. The YOLO11x-cls model, fine-tuned on manually annotated 5 m × 5 m tiles, achieved an overall F1 score of 0.928 (range: 0.783–1.000; SD: 0.054) for solid waste detection across all study regions. Predicted distributions reveal heterogeneous waste accumulation patterns, from localized hotspots — often concentrated along waterways, which can exacerbate flood and public health risks — to dispersed litter across urban landscapes. Analysis of the Relative Solid Waste Contamination Index (RSWCI) shows strong positive correlations with population density (ρ = 0.674) and block complexity (ρ = 0.700), capturing fine-scale infrastructure deficits. By contrast, the association with the coarser index of subnational human development (ρ = 0.215) is weaker, highlighting the value of high-resolution development and infrastructure metrics for understanding localized solid waste patterns. By releasing the model, this study provides a ready-to-use tool for UAV imagery collected by municipalities and local mapping communities, enabling solid waste monitoring without extensive technical expertise. This approach empowers local practitioners to convert UAV imagery into actionable insights, supporting targeted interventions and improved solid waste management across Sub-Saharan Africa.

**Keywords:** Solid Waste · OpenAerialMap · Crowdsourced UAV Imagery · Open-access · Computer Vision · Sub-Saharan Africa

---

## Table of Contents

1. [Abstract](#abstract)
2. [Repository Structure](#repository-structure)
3. [Data](#data)
4. [Installation](#installation)
5. [Workflow](#workflow)
   - [`01_data_acquisition_preprocessing/`](#01_data_acquisition_preprocessing)
     - [Query OAM Catalog](#11-query-oam-catalog)
     - [Download and Tile Imagery](#12-download-and-tile-imagery)
     - [Build YOLO Dataset](#13-build-yolo-dataset)
     - [Download Auxiliary Data](#14-download-auxiliary-data)
   - [`02_model_training/`](#02_model_training)
     - [Train Waste Classifier](#21-train-waste-classifier)
     - [Run Waste and Greenery Inference](#22-run-waste-and-greenery-inference)
   - [`03_analysis/`](#03_analysis)
     - [Calculate AOI Metrics](#31-calculate-aoi-metrics)
     - [Plot Results](#32-plot-results)
5. [Results](#results)
6. [Citation](#citation)
7. [License](#license)

---

## Repository Structure

```
waste-detection-ssa/
|
+-- 01_data_acquisition_preprocessing/
|   +-- 01_query_oam_catalog.py        <- query OAM API; output candidate CSV + GPKG
|   +-- 02_download_and_tile.py        <- download GeoTIFFs; build 5 m tile grids
|   +-- 03_create_yolo_dataset.py      <- labeled tile GPKG -> YOLO PNG dataset
|   +-- 04_download_auxiliary_data.py  <- download MillionNeighborhoods parquet (~8 GB)
|
+-- 02_model_training/
|   +-- 01_train_waste_classification.py  <- fine-tune YOLO11x-cls
|   +-- 02_predict.py                     <- add pred_class/confidence/green_pct to tile
|
+-- 03_analysis/
|   +-- 01_calculate_aoi_metrics.py    <- compute RSWCI + join all metrics -> data/AOI.gpkg
|   +-- 02_plot_results.py             <- bar charts, scatter panels
|   +-- tutorial.ipynb                 <- end-to-end interactive walkthrough
|
+-- data/                              <- runtime data (gitignored except .gitkeep stubs)
|   +-- oam_catalog.csv / .gpkg        <- OAM query output (~256 candidate scenes)
|   +-- oam_catalog_summary.csv        <- human-readable summary with location columns
|   +-- AOI.gpkg                       <- final per-AOI metrics (output of 03_analysis)
|   +-- imagery/                       <- downloaded GeoTIFFs (gitignored)
|   +-- tiles/                         <- tile grids: {oam_id}_tiles.gpkg
|   |                                     columns: tile_id, oam_id, row, col,
|   |                                              filename, label [, pred_class,
|   |                                              confidence, sam_greenery_pct, pred_class_green]
|   +-- dataset/                       <- YOLO PNG crops (gitignored)
|   +-- auxiliary/
|   |   +-- africa_geodata.parquet     <- MillionNeighborhoods Africa
|   |   +-- shdi_national.csv          <- SHDI
|   +-- predictions/
|   |   +-- green/                     <- SAM greenery polygons: {oam_id}_green.gpkg
|   +-- results/                       <- CSV summaries, figures, evaluation reports
|
+-- requirements.txt
+-- .gitignore
```

---

## Data

> Full schema, column descriptions, and a breakdown of what is committed vs generated at runtime: **[`data/README.md`](data/README.md)**

### UAV Imagery

All imagery is sourced from OpenAerialMap under contributors' respective open licenses. Two filters are applied locally after download: GSD 3.5–6.0 cm and footprint area ≥ 1 km². Overlapping scenes within the same AOI are merged into a single GeoTIFF (most recent acquisition on top). See [step 1.1](#11-query-oam-catalog) for the full selection procedure.

<details>
<summary>Full dataset — 29 study regions (click to expand)</summary>

| Country | Region | GSD (cm) | Area (km²) | OAM ID | Date | Provider |
|---|---|---|---|---|---|---|
| Cote d'Ivoire | Diepsloot | 5.00 | 13.66 | 666e4ea2f1cf8e0001fb2f64 | 2020-05-19 | IRD - MIVEGEC |
| Cote d'Ivoire | Katlehong South | 5.00 | 7.94 | 666ef3bdf1cf8e0001fb2f6e | 2020-05-08 | IRD - MIVEGEC |
| Cote d'Ivoire | Vosloorus | 5.00 | 2.61 | 666b9b63f1cf8e0001fb2f16 | 2020-05-06 | IRD - MIVEGEC |
| Ghana | Alajo (Accra) | 3.60 | 12.29 | 5be9bf8ac6c3bf0005896106 | 2018-11-12 | Makoko |
| Ghana | Old Fadama (Accra) | 5.00 | 2.08 | 66e3c6e7cd0baa0001b62114 | 2024-08-27 | OpenStreetMap Ghana |
| Kenya | Kakuma | 4.50 | 3.95 | 63b461953fb8c100063c5600 | 2022-11-22 | ESA Hub |
| Liberia | Duduza | 5.10 | 4.41 | 65c4f779499b4d000186ee78 | 2020-01-19 | HOT |
| Liberia | Geluksdal | 4.99 | 46.25 | 5dee77e79c3b1700059a3593 | 2019-10-02 | Uhurulabs |
| Liberia | KwaThema | 5.31 | 13.91 | 65c4f080499b4d000186ee76 | 2020-01-19 | HOT |
| Liberia | Langaville | 5.00 | 29.84 | 5e0ef7c515d478000501ea61 | 2019-12-09 | Uhurulabs |
| Liberia | Monrovia | 5.10 | 77.47 | 64d4eaba19cb3a000147a604 | 2023-08-09 | HOT |
| Liberia | Reiger Park | 5.08 | 59.81 | 5e557ce2642d040007b7c56f | 2020-02-23 | Uhurulabs |
| Liberia | Thokoza | 5.16 | 7.57 | 65c4f03f499b4d000186ee75 | 2020-01-19 | HOT |
| Liberia | Tsakane | 5.04 | 57.48 | 5e6a8cdb5abd57000732847c | 2020-02-23 | Uhurulabs |
| Liberia | Wattville | 5.06 | 15.75 | 65c4f27d499b4d000186ee77 | 2020-01-19 | HOT |
| Mozambique | Bairro Jardim | 5.00 | 2.60 | 5a7e18425a9ef7cb5d4efd59 | 2018-02-09 | Paolo Paron |
| Mozambique | Chamanculo | 3.55 | 3.24 | 64c3a7f13473010001ab8c4a | 2023-07-25 | #MapeandoMeuBairro |
| Mozambique | Mafala | 4.43 | 35.00 | 64a486bf64adbc00012e082e | 2023-07-24 | #MapeandoMeuBairro |
| Mozambique | Maxaquene | 4.43 | 40.00 | 66156a9fe89cf30001e0c3bb | 2024-04-08 | #MapeandoMeuBairro |
| Mozambique | Bairro Matola | 4.49 | 8.58 | 67421e365f60b100016fc070 | 2024-11-19 | #MapeandoMeuBairro |
| Mozambique | Zonkizizwe | 3.92 | 3.23 | 5dee39919c3b1700059a3586 | 2019-10-23 | INGC |
| Niger | Saga (Niamey) | 5.89 | 1.94 | 5a25ae87bac48e5b1c51946f | 2017-11-15 | Drone Africa Service |
| Sao Tome & Principe | Sao Tome | 5.92 | 5.82 | 59e62b943d6412ef7220a2a5 | 2017-02-28 | Drones Adventures |
| Sao Tome & Principe | Praia Gamboa | 4.40 | 2.86 | 59e62b943d6412ef7220a28f | 2017-03-03 | Drones Adventures |
| Sao Tome & Principe | Praia Melao | 4.53 | 1.30 | 59e62b943d6412ef7220a29d | 2017-02-28 | Drones Adventures |
| Sierra Leone | Freetown | 3.95 | 50.00 | 69075f1de47603686de24fe8 | 2025-10-22 | DroneTM |
| Tanzania | Msimbazi (Dar es Salaam) | 5.04 | 14.00 | 64f1a2b3c5d678906d3facg5 | 2025-09-23 | OpenMapDevelopment Tanzania |
| Tanzania | Bukoba City | 3.52 | 3.95 | 59e62b8a3d6412ef72209d69 | 2016-10-09 | WeRobotics |
| Uganda | Ggaba (Kampala) | 3.53 | 2.02 | 5bc0ae9fc7e1cf0008e45e1f | 2018-09-17 | GeoGecko |

</details>

### Labels

Approximately 200 tiles per region were labeled manually in QGIS: ~100 *waste* and ~100 *background*. Labels are stored in the `label` column of `data/tiles/<oam_id>_tiles.gpkg`. The dataset is split 70 / 15 / 15% (train / val / test) per region using a fixed random seed.

<!-- | Statistic | Value |
|---|---|
| Regions / Countries | 29 / 10 |
| Mean GSD | 4.8 cm/px |
| Mean coverage per region | 15.6 km^2 |
| Total labeled tiles | 5,800 (train 4,060 / val 870 / test 870) | -->

### Auxiliary Data

Urban morphology and population statistics come from a single parquet published by the MillionNeighborhoods project. Downloaded by `04_download_auxiliary_data.py` to `data/auxiliary/africa_geodata.parquet`.

---

## Installation

```bash
git clone https://github.com/<your-org>/waste-detection-ssa
cd waste-detection-ssa

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

> **PyTorch / CUDA:** `torch>=2.0` in `requirements.txt` is a CPU placeholder. Install the GPU build matching your driver from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) before running training or inference.

---

## Workflow

```
01_query_oam_catalog.py        →  data/oam_catalog_summary.csv
        |
        v  [pick OAM IDs from summary CSV]
02_download_and_tile.py        →  data/imagery/*.tif
                                   data/tiles/*_tiles.gpkg
        |
        v  [QGIS: set label=1/0 in data/tiles/*_tiles.gpkg]
03_create_yolo_dataset.py      →  data/dataset/train|val|test/
04_download_auxiliary_data.py  →  data/auxiliary/africa_geodata.parquet
        |
        v
01_train_waste_classification.py  →  data/models/<n>/weights/best.pt
02_predict.py                     →  data/tiles/*_tiles.gpkg  (+ pred_class, confidence, sam_greenery_pct, pred_class_green)
                                     data/predictions/green/*_green.gpkg
        |
        v
03_analysis/01_calculate_aoi_metrics.py  →  data/AOI.gpkg
03_analysis/02_plot_results.py           →  data/results/figures/
```

---

### `01_data_acquisition_preprocessing/`

#### 1.1 Query OAM Catalog

Pages through the OAM `/meta` API for all scenes within the Africa bounding box. Filters locally by:

1. GSD between 3.5 and 6.0 cm
2. Minimum coverage area ≥ 1 km²

```bash
python 01_data_acquisition_preprocessing/01_query_oam_catalog.py
```

> **Optional — reverse geocoding:** add `--geocode` to auto-populate the Country / City columns via the Nominatim reverse-geocoding API (~5 min extra). Without it those columns are blank — fill them in manually from `oam_catalog_summary.csv`.

> **Manual step:** Open `data/oam_catalog.gpkg` in QGIS or another GIS application. Inspect scene footprints; remove duplicates, cloud-affected, or poor-quality scenes; and dissolve overlapping footprints if needed. Save the final AOI selection as `data/oam_AOI.gpkg`. This study retained 29 AOIs from ~256 candidates.

**Outputs:** `data/oam_catalog.gpkg` · `data/oam_catalog.csv` · `data/oam_catalog_summary.csv`

---

#### 1.2 Download and Tile Imagery

Downloads all GeoTIFFs listed in the input GeoPackage with retry and resume support. With `--merge`, spatially overlapping scenes are grouped automatically using footprint geometries and merged into a single GeoTIFF per group using `rasterio.merge` (most recent acquisition takes precedence on overlap).

```bash
python 01_data_acquisition_preprocessing/02_download_and_tile.py \
    --gpkg data/oam_AOI.gpkg
```

> **Optional:** If downloaded scenes overlap, merge them in QGIS before tiling (*Raster → Miscellaneous → Merge*). Save the merged output as `<oam_id>_merged.tif` in `data/imagery/`.

> **Manual step:** Open each `data/tiles/<oam_id>_tiles.gpkg` in QGIS and set the `label` column (`1` = waste, `0` = background) for approximately 100 tiles per class per region.

| tile_id | oam_id | label | geometry |
|---|---|---|---|
| 1 | 59e62b8a… | 1 | POLYGON (…) |
| 2 | 59e62b8a… | 0 | POLYGON (…) |
| 3 | 59e62b8a… | *(blank)* | POLYGON (…) |

Set `label = 1` for tiles containing visible waste, `label = 0` for clean background. Tiles with a blank label are excluded from dataset creation.

**Outputs:** `data/imagery/<oam_id>.tif` · `data/tiles/<oam_id>_tiles.gpkg`

---

#### 1.3 Build YOLO Dataset

Reads each labeled GeoPackage from `data/tiles/`, crops the corresponding GeoTIFF tile-by-tile to 128 × 128 px PNG patches, and writes a YOLO classification directory tree under `data/dataset/`:

```
data/dataset/
+-- train/  waste/  background/
+-- val/    waste/  background/
+-- test/   waste/  background/
```

Split: 70 / 15 / 15% per region, stratified by class, seed = 0.

```bash
python 01_data_acquisition_preprocessing/03_create_yolo_dataset.py \
    --tiles-dir   data/tiles/ \
    --imagery-dir data/imagery/ \
    --outdir      data/dataset/
```

**Outputs:** `data/dataset/{train,val,test}/{waste,background}/*.png`

---

#### 1.4 Download Auxiliary Data

Downloads the MillionNeighborhoods Africa urban-morphology parquet (~8 GB) and the Sub-national Human Development Index (SHDI) from [Zenodo](https://zenodo.org/record/17467221).

```bash
python 01_data_acquisition_preprocessing/04_download_auxiliary_data.py
```

The parquet covers the entire continent. AOI-specific filtering and spatial intersection with the study regions happen in the next step (`01_calculate_aoi_metrics.py`).

> **Note:** The download is ~8 GB and may take 20–30 min depending on connection speed. Already-downloaded files are skipped automatically on subsequent runs.

**Outputs:** `data/auxiliary/africa_geodata.parquet` · `data/auxiliary/shdi_national.csv`

---

### `02_model_training/`

#### 2.1 Train Waste Classifier

Fine-tunes YOLO11x-cls on the dataset produced in step 1.3. All hyperparameters are constants in the file.

| Parameter | Value |
|---|---|
| Epochs | 150 |
| Batch size | 64 |
| Image size | 128 px |
| Optimizer | AdamW |
| Initial LR (`lr0`) | 0.0005 |
| LR schedule | Cosine decay |

```bash
python 02_model_training/01_train_waste_classification.py
```

**Outputs:** `data/models/<n>/weights/best.pt` · `data/results/test_metrics_overall.json`

| Model | Parameters | Test Accuracy | Waste F1 | Background F1 |
|---|---|---|---|---|
| YOLO11x-cls | 28.3 M | 92.87% | 92.76% | 92.99% |

---

#### 2.2 Run Waste and Greenery Inference

Runs the trained YOLO waste-detection model and the pre-trained SAM 3 model over all study scenes.

**Waste classification works at tile level** — YOLO scores each 5 × 5 m tile as *waste* or *background*. Tiles are cropped on the fly from the source GeoTIFF. SAM processes the full scene in 2048 × 2048 px windows and produces polygon outputs for greenery coverage.

```bash
# Step 1 — waste classification
python 02_model_training/02_predict.py \
    --imagery-dir data/imagery/ \
    --tiles-dir   data/tiles/ \
    --model       02_model_training/checkpoints/best.pt
```

YOLO confidence scores are used to mark each tile as waste or background. The result is written as a `pred_class` column in the corresponding `_tiles.gpkg`.

```bash
# Step 2 — greenery prediction
python 02_model_training/02_predict.py \
    --imagery-dir data/imagery/ \
    --tiles-dir   data/tiles/ \
    --model       02_model_training/checkpoints/best.pt \
    --sam         greenery
```

Meta SAM 3 segments each scene in 2048 × 2048 px windows using the text prompt `"trees, bushes, vegetation"`. The resulting polygons are saved to `data/predictions/green/<oam_id>_green.gpkg` and intersected with the tile grid: tiles where SAM polygons cover ≥ 25% of the tile area are marked green. The result is written as a `pred_class_green` column in the corresponding `_tiles.gpkg`.

**Outputs:** `data/tiles/*_tiles.gpkg` (+ `pred_class`, `pred_class_green`) · `data/predictions/green/*_green.gpkg`

---

### `03_analysis/`

#### 3.1 Calculate AOI Metrics

Reads tile GeoPackages from `data/tiles/`, computes RSWCI from the waste classifier predictions, joins green coverage from SAM polygon outputs, adds MillionNeighborhoods urban metrics via a coverage-weighted spatial join, and writes the final `data/AOI.gpkg`.

**RSWCI** — fraction of tiles classified as waste by the YOLO model:
```
waste_pct = N(pred_class == "waste") / N(tiles) × 100
```

**Green coverage** — fraction of tiles classified as green by SAM:
```
green_pct = N(pred_class_green == "green") / N(tiles) × 100
```

Both MillionNeighborhoods metrics are aggregated to AOI level as a **coverage-fraction-weighted mean**:

```
w_i            = intersection_area(block_i, AOI_extent) / block_area_i
weighted_value = sum(metric_i * w_i) / sum(w_i)
```

where `w_i` is the share of block *i*'s total area that falls within the AOI bounding extent (1.0 if fully inside, 0–1 if the block straddles the boundary). Blocks with `w_i < 0.01` (edge slivers) are excluded. Under this scheme every block that lies fully inside the AOI contributes **equally regardless of its physical size**, treating each urban unit as one observation. This is distinct from a pure area-weighted mean (where `w_i = intersection_area`) and is implemented in `03_analysis/01_calculate_aoi_metrics.py → compute_weighted_metrics()`.

SHDI is joined at country level using the latest available year per country.

| Column | Description |
|---|---|
| `waste_pct` | RSWCI (%) |
| `green_pct` | Vegetation coverage (%) |
| `k_complexity_weighted` | Coverage-weighted street-network complexity |
| `worldpop_population_un_density_hectare_weighted` | Coverage-weighted population density (persons/ha) |
| `shdi` | Sub-national HDI |

```bash
python 03_analysis/01_calculate_aoi_metrics.py
```

**Outputs:** `data/AOI.gpkg` (updated with all metrics)

---

#### 3.2 Plot Results

Produces two output figures from `data/AOI.gpkg`:

1. **RSWCI bar chart** (`waste_green_bar.png`) — regions ordered by decreasing `waste_pct`, coloured by AEZ zone (`DN` column) if present, otherwise by `country`. Vegetation coverage (`green_pct`) overlaid as a line on a secondary y-axis.
2. **Three-panel scatter** (`three_panel_analysis_highres.png`) — RSWCI vs `shdi` (left), `k_complexity_weighted` (middle), and `worldpop_population_un_density_hectare_weighted` (right). Each panel uses a log-scaled x-axis with an OLS regression line, 95% confidence band, and Spearman ρ annotation.

```bash
python 03_analysis/02_plot_results.py \
    --input  data/AOI.gpkg \
    --outdir data/results/figures/
```

**Outputs:** `data/results/figures/waste_green_bar.png` · `data/results/figures/three_panel_analysis_highres.png`

<!-- **`03_analysis/tutorial.ipynb`**

```bash
jupyter notebook 03_analysis/tutorial.ipynb
``` -->

---

## Results

The fine-tuned YOLO11x-cls model demonstrated high accuracy in classifying solid waste and background tiles across the test dataset (n = 870). Overall, the model achieved an accuracy of 92.87%, with per-class F1 scores of 92.76% for solid waste and 92.99% for background. Across individual study regions, solid waste F1 scores ranged from 0.783 to 1.000 (SD: 0.054), indicating consistently strong performance with only minor variation between heterogeneous contexts. Correct predictions were made with a mean confidence of 93.25%, while incorrect predictions exhibited lower confidence (mean 71.92%). These results indicate that the model is highly effective in distinguishing visible solid waste from surrounding urban and peri-urban backgrounds under diverse environmental conditions.

<!-- | Socio-spatial Indicator | Spearman ρ | p-value |
|---|---|---|
| SHDI (`shdi`) | 0.215 | 0.262 (not significant) |
| Street-network complexity (`k_complexity_weighted`) | 0.700 | < 0.05 |
| Population density (`worldpop_population_un_density_hectare_weighted`) | 0.674 | < 0.05 |

--- -->

---

## Citation

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
