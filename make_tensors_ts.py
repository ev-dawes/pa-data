from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio as rio
from pyproj import Transformer
from pystac_client import Client
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

ENDPOINT = (
    "https://earth-search.aws.element84.com/v1"  # Any STAC API endpoint will work here
)
PROD_COLLECTION = "sentinel-2-l2a"  # If we do change endpoint, we will need verify what collections are available
QA_PERCENT = 5  # Later on we will be looking at the SCL (quality) band, this is the threshold used for rejection
TARGET_DIR = Path(
    "datasets/yields"
)  # Targets in this context refers to the yield data, as they form the target for the model
OUTPUT_DIR = Path(
    "datasets/sentinel_ts"
)  # Directory where the processed Sentinel-2 data will be stored
# We want the 10m bands
# These correspond to the naming under 'assets' in the STAC items
# Equivalent to B04, B03, B02, B08, which will show on the filenames
BANDS = ["red", "green", "blue", "nir"]
# We assume the growing season starts April 1st
SEASON_START = {"month": 4, "day": 1}
# We assume the growing season ends September 30th
SEASON_END = {"month": 9, "day": 30}
SCENES = 5  # Number of scenes to fetch


def get_crop(filename):
    return filename.split("_")[0]


def get_yeardate(filename):
    return filename.split("_")[2]


def get_year(filename):
    return get_yeardate(filename)[0:4]


def get_date(filename):
    return get_yeardate(filename)[4:]


def get_bbox(raster):
    with rio.open(raster) as src:
        return src.bounds


def get_crs(raster):
    with rio.open(raster) as src:
        return src.crs


def get_first_candidate_below_threshold(candidates, bbox, crs, threshold):
    """
    Get the first candidate that has a cloud percentage below the threshold (percent).
    Candidates are STAC items.

    Args:
        candidates: List of STAC items.
        bbox: Bounding box.
        crs: CRS of the yield raster.
        threshold: Maximum cloud percentage.

    Returns:
        STAC item corresponding to the first candidate below the threshold if found, None otherwise.
    """
    for candidate in candidates:
        scl = candidate.assets["scl"]
        scl_href = scl.href

        with rio.open(scl_href) as src:
            # Check crs against yield raster
            assert src.crs == crs

            # Get the window for the bbox
            window = from_bounds(*bbox, src.transform)
            scl_data = src.read(1, window=window)

            # Sum 0, 1, 3, 8, 9, 10 pixels
            # 0: No data
            # 1: Saturated or defective
            # 3: Cloud shadows
            # 8: Cloud medium probability
            # 9: Cloud high probability
            # 10: Thin cirrus
            cloud_pixels = np.sum(np.isin(scl_data, [0, 1, 3, 8, 9, 10]))
            cloud_prc = cloud_pixels / scl_data.size * 100

            if cloud_prc < threshold:
                return candidate

    return None


def get_bands(candidate, bands, bbox, crs, to_float=False):
    """
    Get the bands for the given candidate.

    Args:
        candidate: Candidate STAC item.
        bands: List of bands to fetch.
        bbox: Bounding box.
        crs: CRS of the yield raster.
        to_float: Whether to return the bands as float32.

    Returns:
        Memfile with the band data.
    """
    memfile = rio.io.MemoryFile()

    # Get the first band to set up the profile
    band = bands[0]
    asset = candidate.assets[band]
    band_href = asset.href

    with rio.open(band_href) as src:
        # Check crs against yield raster
        # Might want to relax this in the future
        assert src.crs == crs

        # Get the window for the bbox
        # and update the profile
        window = from_bounds(*bbox, src.transform)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            count=len(bands),
            height=window.height,
            width=window.width,
            transform=transform,
        )

        if to_float:
            profile.update(dtype="float32")

        # Save the bands to the memfile
        with memfile.open(**profile) as dst:
            for i, band in enumerate(bands, start=1):
                asset = candidate.assets[band]
                band_href = asset.href

                with rio.open(band_href) as src:
                    band_data = src.read(1, window=window)
                    if to_float:
                        band_data = band_data.astype("float32")
                    dst.write(band_data, i)

    return memfile


def match_raster_and_mask(
    target, mask, output_path=None, resampling_method=Resampling.bilinear
):
    """
    Take our bands and reporject, resample, crop, and mask them to match the mask (target/yield) raster.

    Args:
        target: Path to the target raster.
        mask: Path to the mask/target/yield raster.
        output_path: Path to save the output raster. If none is given, an in-memory file is used.
        resampling_method: Resampling method to use when reprojecting the bands.

    Returns:
        Path to the output raster if output_path is given, otherwise a MemoryFile.
    """
    with rio.open(mask) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width, ref_height = ref.width, ref.height
        mask_data = ref.read(1)
        mask_nodata = ref.nodata if ref.nodata is not None else 0

    with rio.open(target) as src:
        # Update metadata
        new_meta = src.meta.copy()
        new_meta.update(
            {
                "crs": ref_crs,
                "transform": ref_transform,
                "width": ref_width,
                "height": ref_height,
                # Use 0 as nodata for integer types
                # and the mask nodata value for float32
                "nodata": mask_nodata if src.dtypes[0] == "float32" else 0,
            }
        )

        # Use in-memory file if no output path is given
        if output_path is None:
            memfile = MemoryFile()
            output = memfile.open(**new_meta)
        else:
            memfile = None
            output = rio.open(output_path, "w", **new_meta)

        with output as dst:
            for i in range(1, src.count + 1):
                # Reproject band
                band_data = np.zeros((ref_height, ref_width), dtype=src.dtypes[i - 1])
                reproject(
                    source=rio.band(src, i),
                    destination=band_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling_method,
                )

                # Apply mask
                # This can easily be made toggleable
                band_data = np.where(
                    np.isclose(mask_data, mask_nodata), output.nodata, band_data
                )

                # Write to output
                dst.write(band_data, i)

        return memfile if memfile else output_path


if __name__ == "__main__":
    # Check for existance of TARGET_DIR
    if not TARGET_DIR.exists():
        raise FileNotFoundError(f"{TARGET_DIR} does not exist")

    # Create OUTPUT_DIR if it does not exist
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # We can refactor to process all crops at once, or a list thereof if desired
    # IIRC naming doesn't even depend on this, and checks the target raster
    crop = "SOYBEANS"
    # Get a list of the yield rasters for the given crop
    yields = list(TARGET_DIR.glob("*.tif"))
    yields = [y for y in yields if crop == get_crop(y.name)]

    client = Client.open(ENDPOINT)

    # Bonus points to anyone who can guess why this isn't just named 'yield'
    for yield_raster in yields:
        year = get_year(yield_raster.stem)
        day = get_date(yield_raster.stem)

        print(f"Processing {yield_raster.stem} from {year} {day}")

        # Convert the julian date and year to a date object
        harvest_date = datetime.strptime(f"{year} {day}", "%Y %j")
        begin_search_date = datetime(
            harvest_date.year, SEASON_START["month"], SEASON_START["day"]
        )
        # Looks like some harvest dates might be off, to be safe, we end the search
        # a few days before the harvest date.
        end_search_date = harvest_date - timedelta(days=2)

        # The STAC client requires the bounding box in lat/lon
        crs = get_crs(yield_raster)
        transformer = Transformer.from_crs(crs, "epsg:4326")
        bbox = get_bbox(yield_raster)
        # WARNING: transform takes x, y but returns y, x as we go from epsg:326XX to epsg:4326
        # Check the related WKTs to see the order of the coordinates
        min_lat, min_lon = transformer.transform(bbox.left, bbox.bottom)
        max_lat, max_lon = transformer.transform(bbox.right, bbox.top)
        # The bbox order for search is [minx, miny, maxx, maxy]
        bbox_latlon = [min_lon, min_lat, max_lon, max_lat]

        search = client.search(
            collections=[PROD_COLLECTION],
            datetime=f"{begin_search_date.date()}/{end_search_date.date()}",
            bbox=bbox_latlon,
            query={"proj:epsg": {"eq": crs.to_epsg()}},
            sortby=[
                {
                    "field": "properties.datetime",
                    "direction": "desc",
                },  # Most recent first
                {"field": "properties.s2:sequence", "direction": "desc"},
            ],  # Items may have multiple sequences, we take the highest
            max_items=100,
        )

        candidates = list(search.items())

        # Post-process to keep only the highest s2:sequence for each unique date
        # WARNING: datetime can vary a bit between scenes of the same date, so we take only the date
        unique_scenes = {}
        for candidate in candidates:
            scene_date = datetime.strptime(
                candidate.properties["datetime"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).date()
            sequence = candidate.properties["s2:sequence"]
            if (
                scene_date not in unique_scenes
                or sequence > unique_scenes[scene_date].properties["s2:sequence"]
            ):
                unique_scenes[scene_date] = candidate

        # Convert the dictionary back to a list of candidates
        candidates = list(unique_scenes.values())

        # TODO filter by quality

        print(f"Found {len(candidates)} candidates")

        for candidate in candidates:
            print(f"Processing candidate {candidate.id}")

            # Get the bands
            stack = get_bands(candidate, BANDS, bbox, crs, to_float=True)

            # Crop, mask, and reproject stack
            date = datetime.strptime(
                candidate.properties["datetime"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).date()
            field_output_dir = OUTPUT_DIR / yield_raster.stem
            field_output_dir.mkdir(exist_ok=True, parents=True)
            output = field_output_dir / f"{date}.tif"
            # If the output already exists, we skip it
            if output.exists():
                print(f"{output} already exists, skipping")
                continue

            match_raster_and_mask(stack, yield_raster, output)

            print(f"Output saved to {output}")
