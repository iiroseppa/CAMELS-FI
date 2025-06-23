#!/usr/bin/env python3
import cdsapi
 
c = cdsapi.Client()
 
first_year = 1950
last_year = 2024

dataset = "reanalysis-era5-land-monthly-means"
variables = ['2m_dewpoint_temperature', '2m_temperature', 'snow_depth_water_equivalent', 
                            'snowmelt', 'skin_reservoir_content', 'volumetric_soil_water_layer_1',
                             'volumetric_soil_water_layer_2', 'total_evaporation', 'total_precipitation']

for year in range(first_year, last_year + 1):
    for month in range(1, 13):
        for variable in variables:
            print("=========================================================")
            print(f"Downloading {year}-{month:02d}-{variable}")
            request = {
                        'product_type': ['monthly_averaged_reanalysis_by_hour_of_day'],
                        'variable': variable,
                        'year': str(year),
                        'month': "{month:02d}".format(month=month),
                        'day': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                            '13', '14', '15',
                            '16', '17', '18',
                            '19', '20', '21',
                            '22', '23', '24',
                            '25', '26', '27',
                            '28', '29', '30',
                            '31',
                        ],
                        'time': [
                            '00:00', '01:00', '02:00',
                            '03:00', '04:00', '05:00',
                            '06:00', '07:00', '08:00',
                            '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00',
                            '15:00', '16:00', '17:00',
                            '18:00', '19:00', '20:00',
                            '21:00', '22:00', '23:00',
                        ],
                        'area': [
                            70, 20, 59, 32,
                        ],
                        'data_format': 'netcdf',
                        'download_format': 'unarchived'
                    }            
            try:
                c.retrieve(
                    dataset,
                    request,
                    "data/{year}-{month:02d}-{variable}.nc".format(year=year, month=month, variable=variable))
            except Exception as ex:
                print(f"An exception occured while downloading {year}-{month}-{variable}")
                print(f"The Exception was {ex}")
                with open("error_log.txt", 'a') as file:
                    file.write(f"{year}-{month}-{variable}" + '\n')
                
            
