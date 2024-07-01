:: Launch Earth Engine Download

@echo off

call conda activate satsite

call python ee_google.py -n FR-Gri -lat 48.84422 -lon 1.95191 --download_folder "Grignon/Satellites/raw/" --download_to_cloud 1 --download_rclone_path C:/Users/%USERNAME%/Documents/rclone-v1.66.0-windows-amd64
