#!/usr/bin/env python3
"""
01_query_oam_catalog.py
========================
Query the OpenAerialMap (OAM) catalog API for UAV imagery over Africa that
meets the study selection criteria, and save the results as a CSV + GeoPackage.

Selection criteria (matching the paper)
----------------------------------------
  - Bounding box : Africa  (-26 W, -35 S, 52 E, 38 N)
  - GSD          : 3.5 – 6.0 cm  (applied locally after download)
  - Area         : ≥ 1 km²       (applied locally after download)


Output
------
  data/oam_catalog.csv        — flat table of matching images
  data/oam_catalog.gpkg       — same data with WGS-84 footprint polygons
                                (expect ~150–160 rows for Africa, GSD 3.5-6 cm)

Then review the output manually (e.g. in QGIS):
  1. Inspect footprints, remove duplicates / cloud-covered / low-quality scenes.
  2. Save your final AOIs as  data/oam_AOI.gpkg    (used by 02_download_and_tile.py).

Example
-------
    python 01_data_acquisition_preprocessing/01_query_oam_catalog.py
    python 01_data_acquisition_preprocessing/01_query_oam_catalog.py \\
        --gsd-min 3.5 --gsd-max 6.0 --min-area 1.0 --outdir data/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape
from tqdm import tqdm

try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    _GEOPY_AVAILABLE = True
except ImportError:
    _GEOPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# OAM catalog API
# ---------------------------------------------------------------------------

OAM_API_BASE = "https://api.openaerialmap.org"
CATALOG_ENDPOINT = f"{OAM_API_BASE}/meta"
DEFAULT_PAGE_SIZE = 100

# Africa bounding box: minLon, minLat, maxLon, maxLat
AFRICA_BBOX = "-26,-35,52,38"

# GSD filter applied locally (server-side gsd= is an exact-match, not range)
GSD_MIN_CM = 3.5
GSD_MAX_CM = 6.0


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "waste-detection-ssa/1.0 (+https://github.com)",
        "Accept":     "application/json",
    })
    return session


def query_catalog_page(
    session: requests.Session,
    page: int,
    limit: int,
    bbox: str = AFRICA_BBOX,
) -> dict:
    """Fetch one page from the OAM /meta endpoint, filtered to a bounding box.

    NOTE: The ``gsd`` API parameter is an exact-match filter, not a max-GSD
    filter.  We therefore omit it here and apply GSD filtering locally after
    downloading the full catalog for the bounding box.
    """
    params = {
        "limit":   limit,
        "page":    page,
        "bbox":    bbox,          # restrict to Africa to keep the result set manageable
        "orderBy": "gsd",
        "order":   "asc",
    }
    resp = session.get(CATALOG_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_result(r: dict) -> dict | None:
    """Extract fields from a single OAM catalog result."""
    try:
        oam_id   = r.get("_id", "")    # OAM's own scene ID  (e.g. 59e62b8a3d6412ef72209d69)
        title    = r.get("title", "")
        provider = r.get("provider", "")
        acquired = r.get("acquisition_start", "")[:10] if r.get("acquisition_start") else ""

        download = r.get("uuid", "") or r.get("download") or r.get("download_path") or ""

        gsd_m    = r.get("gsd")          # metres
        if gsd_m is None:
            return None
        gsd_cm   = round(float(gsd_m) * 100, 3)

        # File-level UUID = stem of the download URL path
        # e.g. https://.../0/5821c0e3b0eae7f3b143a8ef.tif → 5821c0e3b0eae7f3b143a8ef
        try:
            uuid = Path(urlparse(download).path).stem if download else ""
        except Exception:
            uuid = ""

        # Footprint geometry (GeoJSON)
        geom_raw = r.get("geojson") or r.get("bbox")
        if geom_raw is None:
            return None
        try:
            geom = shape(geom_raw)
        except Exception:
            return None

        # Area in km²  (project to a cylindrical equal-area for rough estimate)
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        area_m2 = abs(geod.geometry_area_perimeter(geom)[0])
        area_km2 = round(area_m2 / 1e6, 4)

        return {
            "oam_id":    oam_id,
            "uuid":      uuid,
            "title":     title,
            "provider":  provider,
            "acquired":  acquired,
            "gsd_cm":    gsd_cm,
            "area_km2":  area_km2,
            "download":  download,
            "geometry":  geom,
        }
    except Exception:
        return None


def query_oam(
    bbox: str = AFRICA_BBOX,
    gsd_min_cm: float = GSD_MIN_CM,
    gsd_max_cm: float = GSD_MAX_CM,
    min_area_km2: float = 1.0,
    page_size: int = DEFAULT_PAGE_SIZE,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Page through the OAM catalog (Africa bbox) and return a GeoDataFrame.

    GSD and area filters are applied locally — the /meta API ``gsd`` param
    is an exact-match and cannot be used as a range filter.
    """
    session = build_session()
    records = []
    page    = 1

    if verbose:
        print(f"Querying OAM catalog  (bbox={bbox}, "
              f"GSD {gsd_min_cm}–{gsd_max_cm} cm, area ≥ {min_area_km2} km²) …")

    # First request to get total count within bbox
    first = query_catalog_page(session, page=1, limit=1, bbox=bbox)
    total = first.get("meta", {}).get("found", 0)
    if verbose:
        print(f"  API reports {total} scenes within the bounding box")
        print(f"  Will filter locally for GSD {gsd_min_cm}–{gsd_max_cm} cm "
              f"and area ≥ {min_area_km2} km²")

    n_pages = max(1, (total + page_size - 1) // page_size)
    pbar    = tqdm(total=n_pages, desc="Fetching pages", unit="page")

    while True:
        try:
            data = query_catalog_page(session, page=page,
                                      limit=page_size, bbox=bbox)
        except requests.HTTPError as e:
            print(f"\n  HTTP error on page {page}: {e}", file=sys.stderr)
            break

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            parsed = parse_result(r)
            if parsed is None:
                continue
            # Local GSD range filter (3.5 – 6.0 cm)
            if not (gsd_min_cm <= parsed["gsd_cm"] <= gsd_max_cm):
                continue
            # Local area filter
            if parsed["area_km2"] < min_area_km2:
                continue
            records.append(parsed)

        pbar.update(1)

        meta = data.get("meta", {})
        if page * page_size >= meta.get("found", 0):
            break
        page += 1

    pbar.close()

    if not records:
        print("  No matching scenes found.", file=sys.stderr)
        return gpd.GeoDataFrame(columns=["oam_id", "uuid", "title", "provider", "acquired",
                                          "gsd_cm", "area_km2", "download",
                                          "geometry"],
                                crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf = gdf.sort_values("gsd_cm").reset_index(drop=True)
    return gdf


# ---------------------------------------------------------------------------
# Reverse geocoding
# ---------------------------------------------------------------------------

def reverse_geocode_gdf(
    gdf: gpd.GeoDataFrame,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Add four reverse-geocoded columns to gdf by querying Nominatim
    (OpenStreetMap) for each row's footprint centroid:

        country      — e.g. "Ghana"
        state_region — e.g. "Greater Accra Region"
        city         — e.g. "Accra"
        region_name  — e.g. "Ayawaso West Municipal District"

    Rate-limited to ~1 request / second to respect the Nominatim ToS.
    For 256 scenes this takes ~5 minutes.
    Requires:  pip install geopy
    """
    empty_cols = {"country": "", "state_region": "", "city": "", "region_name": ""}

    if not _GEOPY_AVAILABLE:
        print(
            "  ⚠ geopy not installed — skipping geocoding.\n"
            "    pip install geopy   then re-run with --geocode",
            file=sys.stderr,
        )
        return gdf.assign(**empty_cols)

    geocoder = Nominatim(user_agent="waste-detection-ssa/1.0")
    reverse  = RateLimiter(
        geocoder.reverse,
        min_delay_seconds=1.1,
        error_wait_seconds=5.0,
        max_retries=2,
    )

    countries, states, cities, regions = [], [], [], []

    for _, row in tqdm(
        gdf.iterrows(), total=len(gdf),
        desc="Reverse geocoding", disable=not verbose,
    ):
        try:
            centroid = row.geometry.centroid
            loc = reverse(
                f"{centroid.y:.6f}, {centroid.x:.6f}",
                language="en",
                exactly_one=True,
            )
            if loc is None:
                raise ValueError("no result")
            addr = loc.raw.get("address", {})
            country = addr.get("country", "")
            state   = addr.get("state",
                       addr.get("province",
                       addr.get("region", "")))
            city    = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("municipality", "")
            )
            region  = (
                addr.get("county")
                or addr.get("suburb")
                or addr.get("district", "")
            )
        except Exception:
            country = state = city = region = ""

        countries.append(country)
        states.append(state)
        cities.append(city)
        regions.append(region)

    gdf = gdf.copy()
    gdf["country"]      = countries
    gdf["state_region"] = states
    gdf["city"]         = cities
    gdf["region_name"]  = regions
    return gdf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Query OAM catalog (Africa bbox) for UAV scenes meeting GSD/area "
            "criteria.  GSD and area filters are applied locally."
        )
    )
    parser.add_argument("--bbox", default=AFRICA_BBOX,
                        help=(
                            "Bounding box for server-side spatial filter: "
                            "'minLon,minLat,maxLon,maxLat' "
                            f"(default: Africa = {AFRICA_BBOX})"
                        ))
    parser.add_argument("--gsd-min", type=float, default=GSD_MIN_CM,
                        help=f"Minimum GSD in cm, applied locally (default: {GSD_MIN_CM})")
    parser.add_argument("--gsd-max", type=float, default=GSD_MAX_CM,
                        help=f"Maximum GSD in cm, applied locally (default: {GSD_MAX_CM})")
    parser.add_argument("--min-area", type=float, default=1.0,
                        help="Minimum footprint area in km², applied locally (default: 1.0)")
    parser.add_argument("--outdir",   default="data",
                        help="Output directory (default: data/)")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE,
                        help=f"API page size (default: {DEFAULT_PAGE_SIZE})")
    parser.add_argument(
        "--geocode", action="store_true",
        help=(
            "Reverse-geocode each AOI centroid via Nominatim (OSM). "
            "Adds country / state_region / city / region_name columns. "
            "Requires: pip install geopy.  Takes ~5 min for 256 scenes."
        ),
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gdf = query_oam(
        bbox=args.bbox,
        gsd_min_cm=args.gsd_min,
        gsd_max_cm=args.gsd_max,
        min_area_km2=args.min_area,
        page_size=args.page_size,
    )

    if gdf.empty:
        print("Nothing to save.")
        return

    # ── Format date column (MM-DD-YYYY) ───────────────────────────────────
    gdf["date"] = (
        pd.to_datetime(gdf["acquired"], errors="coerce")
        .dt.strftime("%m-%d-%Y")
        .fillna("")
    )

    # ── Reverse geocode (optional) ─────────────────────────────────────────
    if args.geocode:
        print("\n► Reverse geocoding AOI centroids …")
        gdf = reverse_geocode_gdf(gdf)
    else:
        for col in ("country", "state_region", "city", "region_name"):
            if col not in gdf.columns:
                gdf[col] = ""

    # ── Save full GeoPackage (with footprint polygons) ─────────────────────
    gpkg_path = outdir / "oam_catalog.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="oam_catalog")
    print(f"✓ GPKG saved: {gpkg_path}")

    # ── Save full CSV (all columns, no geometry) ───────────────────────────
    csv_path = outdir / "oam_catalog.csv"
    gdf.drop(columns="geometry").to_csv(csv_path, index=False)
    print(f"✓ CSV saved : {csv_path}  ({len(gdf)} scenes)")

    # ── Save 9-column summary CSV  ─────────────────────────
    summary_cols = [
        "country", "state_region", "city", "region_name",
        "gsd_cm", "area_km2", "oam_id", "date", "provider",
    ]
    summary = gdf[[c for c in summary_cols if c in gdf.columns]].copy()
    summary.columns = [
        "Country", "State/Region", "City", "Region Name",
        "GSD (cm)", "Coverage (km²)", "OAM ID", "Date", "Provider/Owner",
    ][:len(summary.columns)]
    summary_path = outdir / "oam_catalog_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"✓ Summary  : {summary_path}")

    print(f"\nGSD range  : {gdf['gsd_cm'].min():.2f} – {gdf['gsd_cm'].max():.2f} cm")
    print(f"Area range : {gdf['area_km2'].min():.2f} – {gdf['area_km2'].max():.2f} km²")
    print(f"\nNext step (MANUAL):")
    print(f"  Open {gpkg_path} in QGIS.")
    print(f"  Inspect footprints, remove duplicates/cloud-covered scenes,")
    print(f"  merge overlapping AOIs, and save final AOIs as:")
    print(f"  data/oam_AOI.gpkg  (used by 02_download_and_tile.py)")
    if not args.geocode:
        print(f"\nTo add country/city columns, re-run with --geocode:")
        print(f"  python 01_data_acquisition_preprocessing/01_query_oam_catalog.py --geocode")


if __name__ == "__main__":
    main()
