import xarray as xr
import rioxarray as rioxr

import geopandas as geopd
import pandas as pd

import numpy as np

import os
import time

from dask_jobqueue import SLURMCluster
from dask.distributed import Client, print #Dask print enables seeing worker printouts from main script
from dask import delayed
from dask import compute

# This script calculates the daily mean climate variables for catchments using Dask for parallel processing.
# It reads climate data from NetCDF files, clips the data to the catchment geometries, and calculates the mean values for each catchment.
# The results are saved as CSV files.

# Derived from https://github.com/csc-training/geocomputing/blob/master/python/puhti/06_parallel_dask/multi_node/dask_multinode.py

# CSC project name for SLURMCluster, given in the command line
project_name = "project_2013241"

root = f"/scratch/{project_name}/csc_weather/"
dst_root = os.path.join(root, "raw_time_series")
src_root = os.path.join(root, "fmi_grid")
watershed_path =  os.path.join(root, "CAMELS-FI_catchments.gpkg")

def createSLURMCluster():
    # The number of SLURM jobs
    # Practically, how many nodes you want to use
    number_of_jobs = 181

    # Next, limits and settings for ONE SLURM job. 
    
    # Number of cores per SLURM job. 
    # In bigger analysis this has to fit to one HPC node, so in Puhti max 40 cores.  
    no_of_cores = 1
    
    # Here no_of_cores is also used as number of workers (processes) per SLURM job, but number of workers could also be smaller, but not bigger.
    
    # The memory per SLURM job, so all workers of one SLURM job together. Count with at least 7 GB per worker, possibly more.
     
    # Pay attention to the time option here, especially if you have more delayed functions (=files here) than workers.
    # The worker lifetime should be long enough to handle all delayed functions.
    
    # For futher details see: https://jobqueue.dask.org/en/latest/configuration-setup.html
    
    cluster = SLURMCluster(
        queue="small",
        account=project_name,
        cores=no_of_cores,
        processes=no_of_cores,
        memory=f"{7 * no_of_cores}G",
        walltime="3:30:00",
        interface="ib0"
    )

    cluster.scale(number_of_jobs)
    client = Client(cluster)
    print(cluster.job_script())
    print(client)


def mean_weather(args):
    """ Calculates the daily mean of the given watersheds for one variable, over the time period of one year (and one file)
    Does not return anything, instead writes the result to a csv file of shape (timesteps, catchments)
    """
    src_path,  dst_path, watershed_path = args
    watersheds = geopd.read_file(watershed_path, layer='v1')
    
    weather = pd.DataFrame(index=pd.to_datetime([]), columns=watersheds.Paikka_Id)
    weather.index.name = 'date'

    with rioxr.open_rasterio(src_path, mask_and_scale=True) as data_array:
        # Force load all data to memory, important for the supercomputer
        data_array = data_array.load()
        
        # Iterating over the days in the file 
        for time_step in data_array.Time:
            day = time_step.item()
            one_day_data = data_array.sel({'Time':day})

            row = []
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
                
            weather.loc[str(day)] = row

    weather.to_csv(dst_path)

def main():
    createSLURMCluster()

    ## This list hosts the delayed functions which are then ran with compute()
    list_of_delayed_functions = []

    dirs = ['RRday', 'ET0_FAO', 'Tday',
        'Tgmin', 'Tmin', 'Tmax',
        'Rh', 'Globrad', 'Snow']
    
    attributes = {'Rh' : 'humidity','ET0_FAO': 'pet', 'Tday': 'temperature_mean',
              'Tmin': 'temperature_min', 'Tgmin': 'temperature_gmin', 'Tmax': 'temperature_max',
              'RRday': 'precipitation', 'Globrad': 'radiation_global', 'Snow': 'snow_depth'}
    
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
                with open(os.path.join(dst_root, "error_log.txt"), 'a') as error_log:
                    error_log.write(f"Path does not exist:{src_path}\n")
                continue
                
            dst_file_name = f"{attributes[current_dir]}_{year}.csv"
            dst_path = os.path.join(dst_dir, dst_file_name)
            arg = ((src_path,  dst_path, watershed_path))

            if not os.path.exists(dst_path):
                list_of_delayed_functions.append((delayed(mean_weather)(arg)))

    # After constructing the Dask graph of delayed functions, run them with the resources available
    compute(list_of_delayed_functions)

if __name__ == "__main__":
    start = time.time()
    print("-" * 20)
    print("Starting to create jobs")
    print("-" * 20)
    main()
    end = time.time()
    print("Script completed in " + str(round(end - start, 1)) + " seconds")
