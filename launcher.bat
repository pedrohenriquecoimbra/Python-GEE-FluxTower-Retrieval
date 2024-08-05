:: Launch Earth Engine Download

@echo off

call conda activate satsite

call python ee_google.py -n FR-Gri -lat 48.84422 -lon 1.95191 -collection COPERNICUS/S2_SR_HARMONIZED --download_folder "Grignon/Satellites/raw/" --download_to_cloud 1 --download_rclone_path C:/Users/%USERNAME%/Documents/rclone-v1.66.0-windows-amd64

call python ee_google.py -n FR-Gri -lat 48.84422 -lon 1.95191 -collection COPERNICUS/S2_SR_HARMONIZED --new_band_names NDVI EVI LSWI LAI --new_band_formulas "(NIR - RED) / (NIR + RED)" "2.5 * (NIR -RED) / ((NIR + 6 * RED - 7.5 * BLUE) + 1 * 10**4)" "(B8 - B11) / (B8 + B11)" "0.57 * 2.71828 ** (2.33 * ((NIR - RED) / (NIR + RED)) / 10000)" --download_selected_bands NDVI EVI LSWI LAI --download_folder "Grignon/Satellites/raw/" --download_to_cloud 1 --download_rclone_path C:/Users/%USERNAME%/Documents/rclone-v1.66.0-windows-amd64

call python ee_google.py -n FR-Gri -lat 48.84422 -lon 1.95191 -collection ECMWF/ERA5_LAND/HOURLY --download_folder "Grignon/Satellites/raw/" --download_to_cloud 1 --download_rclone_path C:/Users/%USERNAME%/Documents/rclone-v1.66.0-windows-amd64
