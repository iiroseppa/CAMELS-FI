

import requests
import os

from tqdm import tqdm
from time import sleep

def download_large_file(url, destination):
    """
    Convenience function for downloading a large file
    Originally from https://www.geeksforgeeks.org/how-to-download-large-file-in-python-with-requests/
    :param url: url of the file to be downloaded
    :param destination: destination filepath
    :return: downloads file to the given location
    """
    chunk_size=8192
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(destination, 'wb') as f:

                chunk_pbar = tqdm(response.iter_content(chunk_size=chunk_size))
                byte_count = 0
                for chunk in chunk_pbar:
                    byte_count += chunk_size
                    gigabytes = round(byte_count / 1_000_000_000, 2)
                    chunk_pbar.set_description(f"{gigabytes} GB / 1.0 GB")
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print("Error downloading the file:", e)
        with open("error_log.txt", 'a') as error_log:
            error_log.write(f"{destination}")


url_root = "http://fmi-gridded-obs-daily-1km.s3-website-eu-west-1.amazonaws.com/Netcdf"
# ET0_FA0 is available only from 1981 onwards,
variables = ['ET0_FAO', 'Globrad', 'Psea',
             'RRday', 'Rh', 'Snow',
             'Tday', 'Tgmin', 'Tmax',
             'Tmin'
             ]
start_year = 1961
end_year = 2023

# In the next level the used names are in lower case, except for the ET0_FAO, which also requires some additional formatting later
lower_case_variables = ['ET0_FA0'] + [variable.lower() for variable in variables[1:]]

variable_pbar = tqdm(enumerate(variables))
for i, variable in variable_pbar:
    variable_pbar.set_description(f"{variable}")
    # if variable folder does not exists, it is created
    if os.path.isdir(variable):
        pass
    else:
        os.mkdir(variable)
    year_pbar = tqdm(range(start_year, end_year + 1))

    for year in year_pbar:
        year_pbar.set_description(f"{variable} {year}")
        # ET0 is only available from 1980 onwards
        if variable == 'ET0_FAO' and year < 1981:
            continue
        # Tgmin is missing for 2022
        if variable == 'Tgmin' and year > 2022:
            continue
        # The name of ET0_FA0 is a bit different from other data
        if variable == 'ET0_FAO':
            filename = f"{variable}_{year}_months_4_to_9.nc"
        else:
            filename = f"{lower_case_variables[i]}_{year}.nc"

        download_url = f"{url_root}/{variable}/{filename}"

        destination = os.path.join(variable, filename)


        download_large_file(download_url, destination)
        




