import geopandas as geopd
import pandas as pd
from shapely.geometry import box

import rasterio
from rasterio.plot import show
from rasterio.features import rasterize
from rasterio.windows import from_bounds

import numpy as np

try:# yoga
    from whitebox.WBT.whitebox_tools import WhiteboxTools
except:# fossa
    from whitebox.whitebox_tools import WhiteboxTools

import os

from shutil import copyfile

from tqdm.notebook import tqdm

def breach_catchment(catchments, root, src_vrt_path, buffer_d):
    """
    Conveniece wrapper for automating breaching
    """
    operation_name = "breach"
    
    tmp_dir = "/tmp/stream_processing"
    if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
    
    
    src_tmp_path = os.path.join(tmp_dir, "source.tif") 
    dst_dir = os.path.join(root, operation_name) 
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    
    # Text file that contains path of all the processed files
    dst_txt_path = os.path.join(root, f"{operation_name}_10m.txt")
    # The resulting vrt_path
    #dst_vrt_path = os.path.join(root, f"{operation_name}_10m.vrt")
    
    
    
    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    
    
    dst_file_paths = []
    
    with tqdm(catchments.iterrows(), total=len(catchments)) as pbar:
        for i, catchment in pbar:
                
            pbar.set_description(f" Doing {operation_name} for catchment {i}")
            
            # Changing the catchment from series to dataframe 
            catchment = geopd.GeoDataFrame(
                    dict(zip(list(catchment.index), list(catchment.values))),
                    crs=catchments.crs, geometry='geometry', index=[0])
    
            catchment_minx = catchment.bounds.at[0, 'minx']
            catchment_miny = catchment.bounds.at[0, 'miny']
            catchment_maxx = catchment.bounds.at[0, 'maxx']
            catchment_maxy = catchment.bounds.at[0, 'maxy']
            
            # Buffering so that edge effects can be mitigated
            buffer = catchment.buffer(buffer_d, join_style='mitre')
            bounds = buffer.bounds
    
            minx = bounds.at[0, 'minx']
            miny = bounds.at[0, 'miny']
            maxx = bounds.at[0, 'maxx']
            maxy = bounds.at[0, 'maxy']
    
            """ reading only the portion of the raster that falls on the buffered area.
            The catchment and dataset need to be in the same projection.
            If the buffer falls outside of the bounds of vrt, the portion is filled with nodata
            """
            with rasterio.open(src_vrt_path) as src:
                profile = src.profile
                values = src.read(
                    1, window=from_bounds(minx, miny, maxx, maxy, src.transform),
                    boundless=True, fill_value=profile['nodata'])
                
                #mask = src.read_masks(window=from_bounds(minx, miny, maxx, maxy, src.transform))
            """For some reason, the dataset contains completely empty DEMs (for example X5142).
            These are skipped and not added anywhere. The indexes are then added to a list and removed from catchments
            """
            unique = np.unique(values)
            if unique.max() < 0:
                continue
            
            # Updating the profile and recalculating the transform to match the values
            profile['transform'] = rasterio.transform.from_bounds(minx, miny, maxx, maxy, values.shape[1], values.shape[0])
            profile['width'] = values.shape[1]
            profile['height'] = values.shape[0]
            profile['driver'] = 'GTiff'
            
            # Saving the opened raster as a temporary file
            with rasterio.open(src_tmp_path, 'w', **profile) as dst:
                dst.write(values, 1)

            dst_path = os.path.join(dst_dir, f"{operation_name}_{i}.tif")

            # The actual processing, depending on the processing option
            
            status = wbt.breach_depressions_least_cost(src_tmp_path, dst_path, dist=20, max_cost=100)

            # The raster is not clipped so that edge effects can be avoided in future prosessing round
            # clipping the raster and moving it to the correct location
            
    
            dst_file_paths.append(dst_path)

            # Writing the files to a file list
            with open(dst_txt_path, 'w') as fp:
                for name in dst_file_paths:
                    fp.write(f"{name}\n")
            
            # Creating the vrt
            #subprocess.run(["gdalbuildvrt", "-input_file_list", dst_txt_path, dst_vrt_path])
            return dst_txt_path

def process_catchments(operation_name, catchments, root, source_txt_path, buffer_d, network_path=None, raise_path=None, verbose=False):
    """ Convenience wrapper for processing the catchment tiffs
    """
    sources = pd.read_csv(source_txt_path, header=None, names=["path"])
    """
    tmp_dir = "/tmp/stream_processing"
    if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
    """
    
    #src_tmp_path = os.path.join(tmp_dir, "source.tif") 
    dst_dir = os.path.join(root, operation_name) 
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    
    # Text file that contains path of all the processed files
    dst_txt_path = os.path.join(root, f"{operation_name}_10m.txt")
    # The resulting vrt_path
    #dst_vrt_path = os.path.join(root, f"{operation_name}_10m.vrt")
    
    wbt = WhiteboxTools()
    wbt.set_verbose_mode(verbose)
    
    dst_file_paths = []
    # a separate counter is needed
    k = 0
    with tqdm(catchments.iterrows(), total=len(catchments)) as pbar:
        for j, catchment in pbar:
            pbar.set_description(f" Doing {operation_name} for catchment {j}")
            
            # Changing the catchment from series to dataframe 
            catchment = geopd.GeoDataFrame(
                    dict(zip(list(catchment.index), list(catchment.values))),
                    crs=catchments.crs, geometry='geometry', index=[0])
    
            catchment_minx = catchment.bounds.at[0, 'minx']
            catchment_miny = catchment.bounds.at[0, 'miny']
            catchment_maxx = catchment.bounds.at[0, 'maxx']
            catchment_maxy = catchment.bounds.at[0, 'maxy']
            
            # Buffering so that edge effects can be mitigated
            buffer = catchment.buffer(buffer_d, join_style='mitre')
            bounds = buffer.bounds
    
            minx = bounds.at[0, 'minx']
            miny = bounds.at[0, 'miny']
            maxx = bounds.at[0, 'maxx']
            maxy = bounds.at[0, 'maxy']
    
            # Fetching the filename
            src_path = sources.at[k, 'path']
            # If the buffer falls outside of the bounds of vrt, the portion is filled with nodata
            """
            with rasterio.open(src_path) as src:
                profile = src.profile
                values = src.read(
                    1, window=from_bounds(minx, miny, maxx, maxy, src.transform),
                    boundless=True, fill_value=profile['nodata'])
            """    
                #mask = src.read_masks(window=from_bounds(minx, miny, maxx, maxy, src.transform))
            """For some reason, the dataset contains completely empty DEMs (for example X5142).
            These are skipped and not added anywhere. The indexes are then added to a list and removed from catchments
            """
            """
            unique = np.unique(values)
            if unique.max() < 0:
                continue
            
            # Updating the profile and recalculating the transform to match the values
            profile['transform'] = rasterio.transform.from_bounds(
                minx, miny, maxx, maxy, values.shape[1], values.shape[0])
            profile['width'] = values.shape[1]
            profile['height'] = values.shape[0]
            profile['driver'] = 'GTiff'
            
            # Saving the opened raster as a temporary file
            with rasterio.open(src_tmp_path, 'w', **profile) as dst:
                dst.write(values, 1)
            """
            dst_path = os.path.join(dst_dir, f"{operation_name}_{j}.tif")
            dst_file_paths.append(dst_path)

            if operation_name == "raise":
                # Raise is either performed with the boundary of the watershed or a custom file, if it is provided
                if raise_path != None:
                    boundary = geopd.read_file(raise_path, bbox=(minx, miny, maxx, maxy))
                        
                else: # default, use the watershed boundary
                    boundary = catchment.boundary.explode().reset_index()
                    boundary['length'] = boundary.length
                    boundary = boundary.loc[boundary['length'] > 160]
                    boundary = boundary.rename({0:'geometry'}, axis=1)

                # There are no changes for this watershed, we can just copy the existing file
                if len(boundary) == 0:
                    # if there are no changes and the paths stay the same, there is no need to copy
                    if src_path != dst_path:
                        copyfile(src_path, dst_path)
                
                else:
                    with rasterio.open(src_path) as src:
                        values = src.read(1)
                        profile = src.profile
                        mask = src.read_masks(1)
    
                    # Rasterizing the catchment boundary.
                    rasterized = rasterize(
                        boundary['geometry'], (profile['height'], profile['width']),
                        dtype=profile['dtype'], transform=profile['transform'])
                
                    # raising the catchment boundaries as walls
                    values = np.where(rasterized==1, values + 100, values)
    
                    with rasterio.open(dst_path, 'w', **profile) as dst:
                            dst.write(values, 1)
            elif operation_name == "dig_intersection":
                """ Walls that are raised around the catchments need to be breached on the pour point of the catchment
                """
                # This operation requires additional keyword argument to function
                assert network_path is not None, "operation dig_intersection requires the keyword argument network_path"
                boundary = catchment.boundary.explode().reset_index()
                boundary['length'] = boundary.length
                boundary = boundary.loc[boundary['length'] > 160]
                boundary = boundary.rename({0:'geometry'}, axis=1)
                boundary = boundary.set_geometry('geometry', crs=catchment.crs)
                
                network = geopd.read_file(network_path, bbox=(minx, miny, maxx, maxy))
                intersection = geopd.overlay(boundary, network, keep_geom_type=False)

                if len(intersection) == 0:
                    print(f"{operation_name} failed for catchment {j} because there were no intersections")
                    return boundary, network, intersection, catchment # for debugging
                    #continue
                intersection['geometry'] = intersection.buffer(10, cap_style='square')
                with rasterio.open(src_path) as src:
                    values = src.read(1)
                    profile = src.profile
                    mask = src.read_masks(1)

                rasterized = rasterize(
                    intersection['geometry'], (profile['height'], profile['width']),
                    dtype=profile['dtype'], transform=profile['transform'])
                values = np.where(rasterized==1, values - 105, values)

                with rasterio.open(dst_path, 'w', **profile) as dst:
                        dst.write(values, 1)
                
            elif operation_name == "breach":
                status = wbt.breach_depressions_least_cost(src_path, dst_path, dist=10, max_cost=50)

                
            elif operation_name == "d8_pointer":
                status = wbt.d8_pointer(src_path, dst_path)

            elif operation_name == "d8_flow_accumulation":
                status = wbt.d8_flow_accumulation(src_path, dst_path, pntr=True)

            elif operation_name == "stream_thresholding":
                with rasterio.open(src_path) as src:
                    values = src.read(1)
                    profile = src.profile
                    mask = src.read_masks(1)
                # value clipping, 40_000 is required for one of pesiös locations. 55_000 is closer to SYKES flow channel estimate
                values = np.where(values > 40_000, 1, 0).astype('uint8')
                profile['dtype'] = 'uint8'
                # update nodata to 255
                values = np.where(mask == 0, 255, values)
                profile['nodata'] = 255
                
                with rasterio.open(dst_path, 'w', **profile) as dst:
                        dst.write(values, 1)
            

            else:
                raise Exception(f"unregocnized operation name {operation_name}")

            # The raster is not clipped so that edge effects can be avoided in future prosessing round
            # clipping the raster and moving it to the correct location
            """
            with rasterio.open(dst_path) as src:
                profile = src.profile
                values = src.read(
                    1, window=from_bounds(
                        catchment_minx, catchment_miny, catchment_maxx,
                        catchment_maxy, src.transform),
                    boundless=True, fill_value=profile['nodata'])
            
            profile['transform'] = rasterio.transform.from_bounds(
                catchment_minx, catchment_miny, catchment_maxx,
                catchment_maxy, values.shape[1], values.shape[0])
            profile['width'] = values.shape[1]
            profile['height'] = values.shape[0]
            """
    
            k +=1

            
    # Writing the files to a file list
    with open(dst_txt_path, 'w') as fp:
        for name in dst_file_paths:
            fp.write(f"{name}\n")
    
    # Creating the vrt
    #subprocess.run(["gdalbuildvrt", "-input_file_list", dst_txt_path, dst_vrt_path])
    return dst_txt_path

def burn_river_graph(catchments, river_network_path, root, src_vrt_path, buffer_d):
    """
    Artificially deepen the river channels in a given river network to dem
    """
    operation_name = "burn"
    """
    tmp_dir = "/tmp/stream_processing"
    if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
    """
    
    #src_tmp_path = os.path.join(tmp_dir, "source.tif") 
    dst_dir = os.path.join(root, operation_name) 
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    
    # Text file that contains path of all the processed files
    dst_txt_path = os.path.join(root, f"{operation_name}_10m.txt")
    # The resulting vrt_path
    #dst_vrt_path = os.path.join(root, f"{operation_name}_10m.vrt")
    
    
    
    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    
    
    dst_file_paths = []
    
    with tqdm(catchments.iterrows(), total=len(catchments)) as pbar:
        for i, catchment in pbar:
                
            pbar.set_description(f" Doing {operation_name} for catchment {i}")
            
            # Changing the catchment from series to dataframe 
            catchment = geopd.GeoDataFrame(
                    dict(zip(list(catchment.index), list(catchment.values))),
                    crs=catchments.crs, geometry='geometry', index=[0])
    
            catchment_minx = catchment.bounds.at[0, 'minx']
            catchment_miny = catchment.bounds.at[0, 'miny']
            catchment_maxx = catchment.bounds.at[0, 'maxx']
            catchment_maxy = catchment.bounds.at[0, 'maxy']
            
            # Buffering so that edge effects can be mitigated
            buffer = catchment.buffer(buffer_d, join_style='mitre')
            bounds = buffer.bounds
    
            minx = bounds.at[0, 'minx']
            miny = bounds.at[0, 'miny']
            maxx = bounds.at[0, 'maxx']
            maxy = bounds.at[0, 'maxy']

            # Reading the portion of teh river network that is needed
            network = geopd.read_file(river_network_path, bbox=(minx, miny, maxx, maxy))
            
            """ reading only the portion of the raster that falls on the buffered area.
            The catchment and dataset need to be in the same projection.
            If the buffer falls outside of the bounds of vrt, the portion is filled with nodata
            """
            with rasterio.open(src_vrt_path) as src:
                profile = src.profile
                values = src.read(
                    1, window=from_bounds(minx, miny, maxx, maxy, src.transform),
                    boundless=True, fill_value=profile['nodata'])
                
                #mask = src.read_masks(window=from_bounds(minx, miny, maxx, maxy, src.transform))
            """For some reason, the dataset contains completely empty DEMs (for example X5142).
            These are skipped and not added anywhere. The indexes are then added to a list and removed from catchments
            """
            unique = np.unique(values)
            if unique.max() < 0: # TODO check if nodata value is the only value
                continue
            
            # Updating the profile and recalculating the transform to match the values
            profile['transform'] = rasterio.transform.from_bounds(minx, miny, maxx, maxy, values.shape[1], values.shape[0])
            profile['width'] = values.shape[1]
            profile['height'] = values.shape[0]
            profile['driver'] = 'GTiff'

            # Rasterizing the network.
            rasterized = rasterize(
                network['geometry'], (profile['height'], profile['width']),
                dtype=profile['dtype'], transform=profile['transform'])
            
            # burning the river channels 1 meter deep
            values = np.where(rasterized==1, values - 1, values)
            
            dst_path = os.path.join(dst_dir, f"{operation_name}_{i}.tif")
            # Saving the opened raster as a temporary file
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(values, 1)

            # The raster is not clipped so that edge effects can be avoided in future prosessing round
            dst_file_paths.append(dst_path)

    # Writing the files to a file list
    with open(dst_txt_path, 'w') as fp:
        for name in dst_file_paths:
            fp.write(f"{name}\n")
    
    # Creating the vrt
    #subprocess.run(["gdalbuildvrt", "-input_file_list", dst_txt_path, dst_vrt_path])
    return dst_txt_path
