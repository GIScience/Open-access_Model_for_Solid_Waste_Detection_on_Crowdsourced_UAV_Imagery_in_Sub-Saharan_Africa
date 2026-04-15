#!/usr/bin/env python3
"""
01_data_acquisition_preprocessing/04_download_auxiliary_data.py
================================================================
Download auxiliary data for metric enrichment:

  1. MillionNeighborhoods Africa urban-morphology parquet (~8 GB)
       Source : https://dsbprylw7ncuq.cloudfront.net/AF/africa_geodata.parquet
       Output : data/auxiliary/africa_geodata.parquet

  2. Global Data Lab Subnational HDI (SHDI) CSV from Zenodo
       Source : https://zenodo.org/records/17467221
       Output : data/auxiliary/shdi_national.csv

The parquet covers the entire continent and contains all columns needed for
downstream metric enrichment (k_complexity, block_area_km2, building statistics,
worldpop_population_un_density_hectare, etc.).  AOI-specific spatial filtering
and intersection happen in the next step (03_analysis/01_calculate_aoi_metrics.py).

Usage
-----
    python 01_data_acquisition_preprocessing/04_download_auxiliary_data.py

    # Skip individual downloads if files are already present
    python 01_data_acquisition_preprocessing/04_download_auxiliary_data.py \\
        --skip-parquet
    python 01_data_acquisition_preprocessing/04_download_auxiliary_data.py \\
        --skip-shdi

Dependencies: requests, tqdm, pandas
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent
ROOT    = HERE.parent
DATA    = ROOT / "data"
AUX_DIR = DATA / "auxiliary"

# ── MillionNeighborhoods ────────────────────────────────────────────────────
# Full Africa urban-morphology parquet (~8 GB).
# Contains: k_complexity, block_area_km2, building_count,
#           worldpop_population_un, worldpop_population_un_density_hectare, …
MN_PARQUET_URL  = "https://dsbprylw7ncuq.cloudfront.net/AF/africa_geodata.parquet"
MN_PARQUET_DEST = AUX_DIR / "africa_geodata.parquet"

# ── SHDI ───────────────────────────────────────────────────────────────────
# Global Data Lab Subnational HDI — national-level, latest year per country.
SHDI_ZENODO_RECORD_ID = 17467221
SHDI_FILE_KEY         = "Subnational HDI Data v8.3.csv"
SHDI_DEST             = AUX_DIR / "shdi_national.csv"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> bool:
    """Stream-download *url* to *dest*. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"  {dest.name}", leave=False,
        ) as pbar:
            for chunk in r.iter_content(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))
        tmp.rename(dest)
        return True
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"    \u2717 Download error: {e}")
        return False


def download_shdi(dest: Path) -> bool:
    """
    Download the Global Data Lab SHDI CSV from Zenodo, keep only the latest
    national-level value per country, and save to *dest*.
    Returns True on success.
    """
    print(f"  Fetching Zenodo record {SHDI_ZENODO_RECORD_ID} ...")
    try:
        record_url = f"https://zenodo.org/api/records/{SHDI_ZENODO_RECORD_ID}"
        with urllib.request.urlopen(record_url, timeout=60) as resp:
            record = json.loads(resp.read().decode())
    except Exception as e:
        print(f"    \u2717 Could not reach Zenodo: {e}")
        return False

    csv_url = None
    for f in record.get("files", []):
        if f.get("key") == SHDI_FILE_KEY:
            csv_url = f["links"]["self"].replace(" ", "%20")
            break
    if csv_url is None:
        print(f"    \u2717 '{SHDI_FILE_KEY}' not found in Zenodo record.")
        return False

    print("  Downloading SHDI CSV ...")
    try:
        with urllib.request.urlopen(csv_url, timeout=180) as resp:
            raw_bytes = resp.read()
    except Exception as e:
        print(f"    \u2717 Download error: {e}")
        return False

    shdi_raw = pd.read_csv(io.BytesIO(raw_bytes))
    print(f"  Downloaded {len(shdi_raw):,} rows.")

    # Keep national-level only, latest year per country
    shdi_nat = shdi_raw[shdi_raw["level"] == "National"].copy()
    shdi_nat = (
        shdi_nat.sort_values("year", ascending=False)
        .drop_duplicates(subset=["country"], keep="first")
        [["country", "shdi", "year"]]
        .rename(columns={"year": "shdi_year"})
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    shdi_nat.to_csv(dest, index=False)
    print(f"  \u2713 Saved {len(shdi_nat)} countries \u2192 {dest}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download auxiliary data: MillionNeighborhoods Africa parquet (~8 GB) "
            "and Global Data Lab Subnational HDI CSV from Zenodo."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-parquet", action="store_true",
        help="Skip parquet download (use if already present at "
             "data/auxiliary/africa_geodata.parquet).",
    )
    parser.add_argument(
        "--skip-shdi", action="store_true",
        help="Skip SHDI download (use if already present at "
             "data/auxiliary/shdi_national.csv).",
    )
    args = parser.parse_args()

    AUX_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("AUXILIARY DATA DOWNLOAD")
    print("=" * 60)

    # ── 1. MillionNeighborhoods parquet ────────────────────────────────────
    if args.skip_parquet:
        print(f"\n\u25ba MillionNeighborhoods parquet: skipped (--skip-parquet)")
        if not MN_PARQUET_DEST.exists():
            print(f"  \u26a0 File not present at {MN_PARQUET_DEST} \u2014 enrichment will fail!")
        else:
            print(f"  \u2713 Already present: {MN_PARQUET_DEST}")
    elif MN_PARQUET_DEST.exists():
        print(f"\n\u25ba MillionNeighborhoods parquet: already present")
        print(f"  {MN_PARQUET_DEST}")
    else:
        print(f"\n\u25ba MillionNeighborhoods parquet: downloading (~8 GB) \u2026")
        print(f"  Source : {MN_PARQUET_URL}")
        print(f"  Dest   : {MN_PARQUET_DEST}")
        if download_file(MN_PARQUET_URL, MN_PARQUET_DEST, chunk_size=4 << 20):
            size_gb = MN_PARQUET_DEST.stat().st_size / 1e9
            print(f"  \u2713 Saved \u2192 {MN_PARQUET_DEST}  ({size_gb:.2f} GB)")
        else:
            print("  \u2717 Download failed. Check network / URL and retry.")

    # ── 2. SHDI from Zenodo ────────────────────────────────────────────────
    if args.skip_shdi:
        print(f"\n\u25ba SHDI: skipped (--skip-shdi)")
        if not SHDI_DEST.exists():
            print(f"  \u26a0 File not present at {SHDI_DEST} \u2014 SHDI enrichment will fail!")
        else:
            print(f"  \u2713 Already present: {SHDI_DEST}")
    elif SHDI_DEST.exists():
        print(f"\n\u25ba SHDI: already present")
        print(f"  {SHDI_DEST}")
    else:
        print(f"\n\u25ba SHDI (Zenodo record {SHDI_ZENODO_RECORD_ID}): downloading \u2026")
        if not download_shdi(SHDI_DEST):
            print("  \u2717 SHDI download failed.")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"MillionNeighborhoods parquet : {'\u2713' if MN_PARQUET_DEST.exists() else '\u2717 MISSING'}")
    print(f"SHDI CSV                     : {'\u2713' if SHDI_DEST.exists() else '\u2717 MISSING'}")
    print(f"{'=' * 60}")
    print(f"\nAll auxiliary data saved to: {AUX_DIR}")
    print("Next step: python 03_analysis/01_calculate_aoi_metrics.py")


if __name__ == "__main__":
    main()

