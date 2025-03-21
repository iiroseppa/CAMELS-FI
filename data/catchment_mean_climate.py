import xarray as xr
import rioxarray as rioxr

import geopandas as geopd
import pandas as pd

import numpy as np

import os
from multiprocessing import Pool

def mean_weather(args):
    """ Calculates the daily mean of the given watersheds for one variable, over the time period of one year (and one file)
    Does not return anything, instead writes the result to a csv file of shape (timesteps, catchments)
    """
    src_path,  dst_path, watersheds = args

    weather = pd.DataFrame(index=pd.to_datetime([]), columns=watersheds.Paikka_Id)
    weather.index.name = 'date'

    with rioxr.open_rasterio(src_path, mask_and_scale=True) as data_array:
        # iterating over the days in the file 
        for time_step in data_array.Time:
            time = time_step.item()
            one_day_data = data_array.sel({'Time':time})

            row = []
            # looks a bit unpythonic but DataFrame.iterrows returns series, not dataframes and it's a hassle to change back
            for i in range(len(watersheds)): 
                watershed = watersheds.iloc[[i]]
                place_id = watershed.Paikka_Id[i]
                
                # Calculating the average of the attribute for the whole catchment
                clipped = one_day_data.rio.clip(watershed.geometry.values, crs=watershed.crs)
                average = clipped.mean().item()
                average = round(average, 1)
                # Failsafe for catchments smaller than the pixel size
                if average is np.nan:
                    clipped = data_array.rio.clip(watershed.geometry.values, crs=watershed.crs, all_touched=True)
                    average = clipped.mean().item()

                row.append(average)
            weather.loc[str(time)] = row
            
    weather.to_csv(dst_path)

root = os.getcwd()

dst_root = os.path.join(root, "raw_time_series")

src_root = os.path.join(root, "fmi_grid")

vector_watershed_path = "CAMELS-FI_catchments.gpkg"
watersheds = geopd.read_file(vector_watershed_path, layer='v1')



dirs = ['RRday', 'ET0_FAO', 'Tday',
        'Tgmin', 'Tmin', 'Tmax',
        'Rh', 'Globrad', 'Snow']
# Converting from FMI namespace to CAMELS
attributes = {'Rh' : 'humidity' ,'ET0_FAO': 'pet', 'Tday': 'temperature_mean',
              'Tmin': 'temperature_min', 'Tgmin': 'temperature_gmin', 'Tmax': 'temperature_max',
              'RRday': 'precipitation', 'Globrad': 'radiation_global', 'Snow': 'snow_depth'}


args = []
years = (1961, 2023)

for current_dir in dirs:
    dst_dir =  os.path.join(dst_root, attributes[current_dir])
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for year in range(years[0], years[1] + 1):
        # tgmin is not available for 2023
        if year == 2023 and current_dir == 'Tgmin':
            continue
        
        # Evapotranspiration has differing scheme to all other sources
        if current_dir == 'ET0_FAO':
            #Pet is only available from 1981
            if year < 1981:
                continue
            # Different time range than other variables
            src_file_name = f"{current_dir}_{year}_months_4_to_9.nc"
        else:
            src_file_name = f"{current_dir.lower()}_{year}.nc"
        

        src_path = os.path.join(src_root, current_dir, src_file_name)
        
        if not os.path.exists(src_path):
            #print(f"path {src_path} doesn't exists")
            with open(os.path.join(dst_root, "error_log.txt"), 'a') as error_log:
                error_log.write(f"Path does not exist:{src_path}\n")
            continue
            
        dst_file_name = f"{attributes[current_dir]}_{year}.csv"
        dst_path = os.path.join(dst_dir, dst_file_name)
        args.append((src_path,  dst_path, watersheds))

with Pool() as p:
    p.map(mean_weather, args)