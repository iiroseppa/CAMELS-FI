import geopandas as geopd
import pandas as pd
from shapely.geometry import box

import rasterio
from rasterio.plot import show
from rasterio.features import rasterize
from rasterio.windows import from_bounds

import numpy as np

try:
    from whitebox.WBT.whitebox_tools import WhiteboxTools
except:
    from whitebox.whitebox_tools import WhiteboxTools

import os
from shutil import copyfile
from tqdm.notebook import tqdm
from parallelbar import progress_map

def breach_catchment(catchments, root, src_vrt_path, buffer_d):
    """
    Convenience wrapper for automating breaching.

    Parameters:
    catchments (GeoDataFrame): The GeoDataFrame containing catchment geometries.
    root (str): The root directory for DEM data.
    src_vrt_path (str): Path to the source VRT file.
    buffer_d (int): The buffer distance to apply.

    Returns:
    str: The path to the text file containing the processed file paths.
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
    
            """ Reading only the portion of the raster that falls on the buffered area.
            The catchment and dataset need to be in the same projection.
            If the buffer falls outside of the bounds of vrt, the portion is filled with nodata
            """
            with rasterio.open(src_vrt_path) as src:
                profile = src.profile
                values = src.read(
                    1, window=from_bounds(minx, miny, maxx, maxy, src.transform),
                    boundless=True, fill_value=profile['nodata'])
                
            unique = np.unique(values)
            if unique.max() < 0:
                continue
            
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
            
            dst_file_paths.append(dst_path)

            # Writing the files to a file list
            with open(dst_txt_path, 'w') as fp:
                for name in dst_file_paths:
                    fp.write(f"{name}\n")
            
            return dst_txt_path

def main_parallel_processing(args):
     """
    Main function for parallel processing of catchments.

    Parameters:
    args (tuple): A tuple containing the arguments for processing.

    Returns:
    str: The path to the processed file.
    """
    operation_name, k, catchment, source_txt_path, buffer_d, network_path, raise_path, crs, src_path, dst_dir = args

    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    
    # Changing the catchment from series to dataframe 
    catchment = geopd.GeoDataFrame(
            dict(zip(list(catchment.index), list(catchment.values))),
            crs=crs, geometry='geometry', index=[0])

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

    dst_path = os.path.join(dst_dir, f"{operation_name}_{k}.tif")
    #dst_file_paths.append(dst_path)

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
        boundary = boundary.set_geometry('geometry', crs=crs)
        
        network = geopd.read_file(network_path, bbox=(minx, miny, maxx, maxy))
        intersection = geopd.overlay(boundary, network, keep_geom_type=False)

        if len(intersection) == 0:
            print(f"{operation_name} failed for catchment {k} because there were no intersections")
            return boundary, network, intersection, catchment
       
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
        
        values = np.where(values > 40_000, 1, 0).astype('uint8')
        profile['dtype'] = 'uint8'
        # update nodata to 255
        values = np.where(mask == 0, 255, values)
        profile['nodata'] = 255
        
        with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(values, 1)
    
    else:
        raise Exception(f"unregocnized operation name {operation_name}")

    return dst_path

def paralell_process_catchments(operation_name, catchments, root, source_txt_path, buffer_d, network_path=None, raise_path=None, verbose=False, n_cpu=os.cpu_count()):
    """
    Convenience wrapper for processing the catchment tiffs in parallel.

    Parameters:
    operation_name (str): The name of the operation to perform.
    catchments (GeoDataFrame): The GeoDataFrame containing catchment geometries.
    root (str): The root directory for DEM data.
    source_txt_path (str): Path to the source text file.
    buffer_d (int): The buffer distance to apply.
    network_path (str, optional): Path to the network data.
    raise_path (str, optional): Path to the raise vector data.
    verbose (bool, optional): Whether to enable verbose mode.
    n_cpu (int, optional): Number of CPUs to use for parallel processing.

    Returns:
    str: The path to the text file containing the processed file paths.
    """
    
    sources = pd.read_csv(source_txt_path, header=None, names=["path"])
    
    dst_dir = os.path.join(root, operation_name) 
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    
    # Text file that contains path of all the processed files
    dst_txt_path = os.path.join(root, f"{operation_name}_10m.txt")
      
    # Generating argument list
    args_list = []
    for k, catchment in catchments.iterrows():
        # Fetching the filename
        src_path = sources.at[k, 'path']
        args_list.append((operation_name, k, catchment,
                source_txt_path, buffer_d, network_path,
                raise_path, catchments.crs, src_path,
                dst_dir))
    
    dst_file_paths = progress_map(main_parallel_processing, args_list, n_cpu=n_cpu)

    # Writing the files to a file list
    with open(dst_txt_path, 'w') as fp:
        for name in dst_file_paths:
            fp.write(f"{name}\n")
    
    return dst_txt_path
        
def process_catchments(operation_name, catchments, root, source_txt_path, buffer_d, network_path=None, raise_path=None, verbose=False):
    """     
    Convenience wrapper for processing the catchment tiffs.

    Parameters:
    operation_name (str): The name of the operation to perform.
    catchments (GeoDataFrame): The GeoDataFrame containing catchment geometries.
    root (str): The root directory for DEM data.
    source_txt_path (str): Path to the source text file.
    buffer_d (int): The buffer distance to apply.
    network_path (str, optional): Path to the network data.
    raise_path (str, optional): Path to the raise vector data.
    verbose (bool, optional): Whether to enable verbose mode.

    Returns:
    str: The path to the text file containing the processed file paths.
    """
    sources = pd.read_csv(source_txt_path, header=None, names=["path"])
 
    dst_dir = os.path.join(root, operation_name) 
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    
    # Text file that contains path of all the processed files
    dst_txt_path = os.path.join(root, f"{operation_name}_10m.txt")

    wbt = WhiteboxTools()
    wbt.set_verbose_mode(verbose)
    
    dst_file_paths = []
    k = 0 #counter
    
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

                values = np.where(values > 40_000, 1, 0).astype('uint8')
                profile['dtype'] = 'uint8'
                # update nodata to 255
                values = np.where(mask == 0, 255, values)
                profile['nodata'] = 255
                
                with rasterio.open(dst_path, 'w', **profile) as dst:
                        dst.write(values, 1)
            

            else:
                raise Exception(f"unregocnized operation name {operation_name}")
    
            k +=1

    # Writing the files to a file list
    with open(dst_txt_path, 'w') as fp:
        for name in dst_file_paths:
            fp.write(f"{name}\n")
    
    return dst_txt_path

def burn_river_graph(catchments, river_network_path, root, src_vrt_path, buffer_d):
    """
    Artificially deepen the river channels in a given river network to DEM
    
    Parameters:
    catchments (GeoDataFrame): The GeoDataFrame containing catchment geometries.
    river_network_path (str): Path to the river network data.
    root (str): The root directory for DEM data.
    src_vrt_path (str): Path to the source VRT file.
    buffer_d (int): The buffer distance to apply.

    Returns:
    str: The path to the text file containing the processed file paths.
    """
    operation_name = "burn"
    
    #src_tmp_path = os.path.join(tmp_dir, "source.tif") 
    dst_dir = os.path.join(root, operation_name) 
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    
    # Text file that contains path of all the processed files
    dst_txt_path = os.path.join(root, f"{operation_name}_10m.txt")
    
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

            # Reading the portion of the river network that is needed
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

            unique = np.unique(values)
            if unique.max() < 0: 
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

            dst_file_paths.append(dst_path)

    # Writing the files to a file list
    with open(dst_txt_path, 'w') as fp:
        for name in dst_file_paths:
            fp.write(f"{name}\n")
    
    return dst_txt_path
