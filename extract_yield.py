from pathlib import Path

import rasterio as rio

RAW_DATA_DIR = Path("raw_data/jd")
PROCESSED_DATA_DIR = Path("datasets/yields")


# For each harvest raster we extract the yield and save it as a new .tif in PROCESSED_DATA_DIR
# The new tifs will be named with the crop name instead of harv to make it easier to identify them
if __name__ == "__main__":
    # Check for the existence of the directories
    if not RAW_DATA_DIR.exists():
        raise Exception(f"Directory {RAW_DATA_DIR} does not exist")
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    # Get years from the raw data directory
    years = [f.name for f in RAW_DATA_DIR.iterdir() if f.is_dir()]
    print(years)

    # Find .tifs that start with harv in RAW_DATA_DIR/{year}/rasters
    rasters = [
        tif
        for year in years
        for tif in RAW_DATA_DIR.joinpath(year, "rasters").glob("harv*.tif")
    ]

    for raster in rasters:
        with rio.open(raster) as src:
            # Get index of vryieldvol band and read it
            band_index = (
                src.tags()["bands_order"].split(",").index("vryieldvol")
            )  # also could target 'wetmass' I guess
            vryieldvol = src.read(band_index + 1)

            crop_name = src.tags()["crop"]

            # Replace harv in filename with crop name
            filename = raster.name.replace("harv", crop_name)

            # Update metadata to reflect that there is only one band
            profile = src.profile
            profile.update(count=1)

            # Save vryieldvol band as a new .tif in PROCESSED_DATA_DIR
            with rio.open(PROCESSED_DATA_DIR.joinpath(filename), "w", **profile) as dst:
                dst.write(vryieldvol, 1)
                dst.update_tags(**src.tags())
