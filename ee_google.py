
"""
        # Trigger the authentication flow.
        # ee.Authenticate()

        # Initialize the library.
        ee.Initialize()
"""
import os
import datetime
import pandas as pd
import ee
import geemap
import requests
import threading
import subprocess
import logging

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()


def are_we_in_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

we_are_in_jupyter = False
if are_we_in_jupyter():
    jupyter = True
else:
    import curses
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()


class eengine():
    def __init__(self, **kwargs):
        self.collection = {}

        default = {
            "SITE_NAME": "SITE_NAME",
            "longitude": 1.95191,
            "latitude": 48.84422,
            "START_DATE": '2020-06-01',
            "END_DATE": '2020-06-02',
            "CLD_PRB_THRESH": 50,
            "NIR_DRK_THRESH": 0.15,
            "CLD_PRJ_DIST": 10,
            "BUFFER": 10**3,
            "CLD_BUFFER": 50,
        }
        self.set_value(**default)
        # update based on user's input
        self.set_value(**kwargs)
        
        self.eepoint = ee.Geometry.Point(self.longitude, self.latitude)
        self.set_value(**{"AOI": self.eepoint.buffer(self.BUFFER).bounds()})
        return


    def set_value(self, **kwargs):
        self.__dict__.update(**kwargs)


    def get_collection(self, collection):
        self.collection[collection] = (ee.ImageCollection(collection)
                           .filterBounds(self.AOI)
                           .filterDate(self.START_DATE, self.END_DATE))
        return self


    def describe(self):
        #echo given parameter + describes topography, surface temperature using satelite data
        self.elevation = ee.Image('USGS/SRTMGL1_003').sample(
            self.eepoint, 30).first().get('elevation').getInfo()
        return
    

    def add_bands(self, newbands={'COPERNICUS/S2_SR_HARMONIZED': {'NDVI': '(NIR - RED) / (NIR + RED)'}}):
        for c, b in newbands.items():
            for k, v in b.items():
                self.collection[c] = self.collection[c].map(lambda x: add_index(x, k, v))
        return self
    

    def s2_cloud_free(self, collection='COPERNICUS/S2_SR_HARMONIZED', **kwargs):
        assert collection in self.collection.keys()
        # assert required bands

        CLD_BUFFER, CLD_PRB_THRESH, NIR_DRK_THRESH, CLD_PRJ_DIST = [kwargs.get(k, vars(self)[k]) for k in ['CLD_BUFFER', 'CLD_PRB_THRESH', 'NIR_DRK_THRESH', 'CLD_PRJ_DIST']]
        self.collection[collection+'_cloudless'] = (
            self.collection[collection]
            #.map(lambda img: img.updateMask(img.select('MSK_CLDPRB').lt(CLD_PRB_THRESH)))
            .map(lambda img: add_cld_shdw_mask(img, CLD_BUFFER, CLD_PRB_THRESH, NIR_DRK_THRESH, CLD_PRJ_DIST))
            .map(lambda img: img.updateMask(img.select('cloudmask').Not()))
            )
        return self 
    

    def show(self, collection=None, image_viz_params=None, zoom=12, **kwargs):
        # Define the visualization parameters.
        if image_viz_params is None:
            image_viz_params = {
                'false color composite': {
                    'bands': ['B5', 'B4', 'B3'],
                    'min': 0 * 10**4,
                    'max': 0.5 * 10**4,
                    'gamma': [0.95, 1.1, 1],
                    },
                'true color composite': {'bands': ['TCI_R', 'TCI_G', 'TCI_B'],},
                    }
        
        if collection is None:
            collection = list(self.collection.keys())[0]
        
        img_shw = self.collection[collection].mosaic().clip(self.AOI)
        
        # Define a map centered on San Francisco Bay.
        map_shw = geemap.Map(center=[self.latitude, self.longitude], zoom=zoom)

        # Add the image layer to the map and display it.
        for k, v in image_viz_params.items():
            map_shw.add_layer(img_shw, v, k)
        display(map_shw)

    def download(self, folder="Grignon/Satellites/", to_cloud=True, rclone_path = r'C:\Users\phherigcoimb\Documents\rclone-v1.66.0-windows-amd64',
                 selected_bands=None):
        assert self.collection, 'DOWNLOAD NOT POSSIBLE. The self.collection element is empty.'

        for collection_name, collection in self.collection.items():
            collection_name = collection_name.replace('/', '_').replace('\\', '_')
            N = collection.size().getInfo()
            collection_lst = collection.toList(N)

            d0, d1 = [pd.to_datetime(datetime.datetime.fromtimestamp(
                ee.Image(collection_lst.get(i)).get('system:time_start').getInfo()//1000)) for i in [0, N-1]]

            for i in range(N):
                image_date = ee.Image(collection_lst.get(i))

                ymdhm = pd.to_datetime(datetime.datetime.fromtimestamp(
                    ee.Image(image_date).get('system:time_start').getInfo()//1000))
                
                if we_are_in_jupyter:
                    print(f'{i+1} / {N} current: {ymdhm} ( {d0} - {d1} ){" "*10}')#, end='\r')
                else:
                    stdscr.addstr(20, 0, f'{i+1} / {N} current: {ymdhm} ( {d0} - {d1} ){" "*10}')
                    stdscr.refresh()
                
                date = str(ymdhm.date()).replace('-', '')

                #run it by band
                for b in image_date.bandNames().getInfo():
                    if selected_bands is None or b in selected_bands:
                        output_name = os.path.join(folder, f'{self.SITE_NAME}_{collection_name}_{date}.{b}.tif')
                        url = image_date.getDownloadUrl({
                            'bands': [b],
                            #'scale': 10, 
                            'region': self.AOI,
                            'format': 'GEO_TIFF',
                        })
                        if to_cloud and rclone_path:
                            #urls3 = url.replace('https://earthengine.googleapis.com/', '--http-url https://earthengine.googleapis.com/ :http:')
                            #print("url: ", url, '\n', urls3, '\n', os.path.basename(url))
                            command = f'{os.path.join(rclone_path, "rclone.exe")} --config {os.path.join(rclone_path, "config-rclone.txt")} copyurl {url} ICOS:bdd-icos-raw-data/{output_name}'
                            #print(command)
                            result = subprocess.run(command, capture_output=True)#, shell=True)
                            if result.stdout or result.stderr:
                                logging.warn(i+1, '/', N, 'current:', ymdhm, '(', d0, '-', d1, 'stdout:', result.stdout.decode("utf-8"), 'stderr:', result.stderr.decode("utf-8"), ')', end='\n')
                        else:
                            response = requests.get(url)
                            with open(output_name, 'wb') as fd:
                                fd.write(response.content)
                                
                        if we_are_in_jupyter:
                            print(output_name, 'was written.', ' '*10, end='\r')
                        else:
                            stdscr.addstr(21, 0, f'{output_name} was written.{" "*10}')
                            stdscr.refresh()
                #print('\x1B[1A', end='\r')
                #curses.echo()
                #curses.nocbreak()
                #curses.endwin()
    
    def taskstatus(self):
        self.tasktoexport.status()


def add_cloud_bands(img, CLD_PRB_THRESH):
    # Get s2cloudless image, subset the probability band.
    #cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    cld_prb = img.select('MSK_CLDPRB')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img, NIR_DRK_THRESH, CLD_PRJ_DIST):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(
        NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
    
    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img, CLD_BUFFER, CLD_PRB_THRESH, NIR_DRK_THRESH, CLD_PRJ_DIST):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img, CLD_PRB_THRESH)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud, NIR_DRK_THRESH, CLD_PRJ_DIST)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(
        img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(CLD_BUFFER*2/20)
                   .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                   .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def add_index(img, NAME, FORMULA):
    # Normalized difference vegetation index (NDVI)
    band_available = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    band_codinames = {'NIR': 'B8', 'RED': 'B4', 'GREEN': 'B3', 'BLUE': 'B2',}
    all_bands_with_aka = {b: img.select(b) for b in band_available}
    all_bands_with_aka.update({n: all_bands_with_aka[b] for n, b in band_codinames.items()})
    idx = img.expression(FORMULA, all_bands_with_aka).rename(NAME).multiply(10**4).toShort()
    img = img.addBands(idx)
    return img

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--SITE_NAME',   type=str)
    parser.add_argument('-lat', '--latitude',   type=float)
    parser.add_argument('-lon', '--longitude',   type=float)
    parser.add_argument('-START_DATE',  type=str)
    parser.add_argument('-END_DATE', type=str)
    parser.add_argument('-BUFFER', type=int, default=10**3)
    parser.add_argument('-collection', type=str, default='COPERNICUS/S2_SR_HARMONIZED')
    parser.add_argument('--CLD_PRB_THRESH', type=int, default=50)
    parser.add_argument('--NIR_DRK_THRESH', type=int, default=0.15)
    parser.add_argument('--CLD_PRJ_DIST', type=int, default=1)
    parser.add_argument('--CLD_BUFFER', type=int, default=50)
    parser.add_argument('--new_band_names', type=str, nargs='+', default="")
    parser.add_argument('--new_band_formulas', type=str, nargs='+', default="")
    parser.add_argument('-bands', '--download_selected_bands', type=str, nargs='+')
    parser.add_argument('-d2folder', '--download_folder', type=str, default="data/tmp/Grignon/")
    parser.add_argument('-d2c', '--download_to_cloud', type=int, default=0)
    parser.add_argument('-rclone', '--download_rclone_path', type=str)
    parser.add_argument('-log', '--log_level', type=int, default=0)
    args = parser.parse_args()
    args = vars(args)

    #print('Start run w/')
    if we_are_in_jupyter:
        print('Start run w/')
    else:
        stdscr.addstr(0, 0, 'Start run w/')
    # replace os.get_cwd() for '' if str
    
    if we_are_in_jupyter:
        print('\n'.join([f'{k}:\t{v[:5] + "~" + v[-25:] if isinstance(v, str) and len(v) > 30 else v}' for k, v in args.items()]), end='\n\n')
    else:
        for i, (k, v) in enumerate(args.items()):
            stdscr.addstr(i+1, 0, f'{k}:')
            stdscr.addstr(i+1, 30, f'{v[:5] + "~" + v[-25:] if isinstance(v, str) and len(v) > 30 else v}')
        stdscr.refresh()
        
    log_level = args.pop('log_level')
    logname = f"log/current_{datetime.datetime.now().strftime('%y%m%dT%H%M%S')}.log"
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=log_level, 
                        force=True)

    collection = args.pop('collection')
    g = eengine(**args).get_collection(collection)
    
    newbands_keys = [k for k in args.keys() if k.startswith('new_band_')]
    newbands_args = {k.replace('new_band_', ''): args.pop(k) for k in newbands_keys}
    assert len(newbands_args['names']) == len(newbands_args['formulas']), f"Length of new bands names ({len(newbands_args['names'])}) and formulas ({len(newbands_args['formulas'])}) must be equal."
    newbands_args = dict(zip(newbands_args['names'], newbands_args['formulas']))
    g = g.add_bands({collection: newbands_args})

    download_keys = [k for k in args.keys() if k.startswith('download_')]
    download_args = {k.replace('download_', ''): args.pop(k) for k in download_keys}
    assert (download_args['to_cloud'] == 0) or (download_args['to_cloud'] and download_args['rclone_path'] is not None), "If send to cloud, path to rclone folder ('-rclone', '--download_rclone_path') must be given."

    # Thread it to not freeze python for long run
    # /!\ caution, multiple runs WILL run multiple times in google's servers
    #              resulting in duplicates on your drive
    threading.Thread(target=g.download, kwargs=download_args).start()
