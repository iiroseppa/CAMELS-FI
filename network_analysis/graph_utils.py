import pandas as pd
import geopandas as geopd
import numpy as np

import shapely
from shapely import Point, LineString
from shapely.ops import split, snap

    
def add_connections(nodes, pour, tolerance=5):
    """ Add connections to nodes.
    Nodes: 
    """
    nodes=nodes.copy()
    for i, row in nodes.iterrows():
        # End points are processed and merged later
        if i % 2 == 1:
            continue
    
        nodes.at[i, 'next'] = i+1
    # The next point from pour point is the pour point itself (or should it remain nan?)
    nodes.at[pour.at[0, 'id'], 'next'] = pour.at[0, 'id']
    
    s_index = nodes.sindex
    # Fetching overlapping points
    nearest = s_index.nearest(nodes.geometry, max_distance=tolerance)
    
    # Removing cases where nearest node is the node itself
    zipped = np.array(list(zip(nearest[0], nearest[1])))
    nearest = zipped[~(nearest[0]== nearest[1])]
    
    for i, row in nodes.iterrows():
        # Start points were processed earlier
        if i % 2 == 0:
            continue
        # Pour point is skipped
        if i == pour.at[0, 'id']:
            continue
        
        # Fetching the node indexes that intersect with the current node
        idxs = np.where(nearest[:, 0] == i)[0]
        # Fetching the dataframe indexes
        df_idxs = []
        for j in idxs:
            df_idxs.append(nearest[j, 1])
        
        # The next node is already defined for even nodes, and every odd node except pour point should intersect with exactly one even node
        next_node = None
        for j in df_idxs:
            if j % 2 == 1:
                continue
            next_node = j
        #assert next_node is not None, f"Something is wrong with graph structure on node {i}"
        
        nodes.at[i, 'next'] = next_node
    
    # Making shortcuts 
    for i, row in nodes.iterrows():
        if i % 2 == 1:
            continue
        # These are floats becauuse of nan
        next_node = nodes.at[i, 'next']
        nodes.at[i, 'next'] = nodes.at[next_node, 'next']
        
    #nodes['next'] = nodes['next']
    # Odd nodes except pour are not needed
    nodes = nodes.loc[(nodes.id % 2 == 0) | (nodes.id == pour.at[0, 'id'])]

    return nodes



def check_network_topology(lines):
    """ Checks whether all parts of the network are connected to each other.
    """
    result = {}
    # check for self intersections and self tangency
    parts = lines.dissolve()
    result['is_simple'] = parts.at[0, 'geometry'].is_simple

    # There should only be one pour point
    nodes = get_start_and_end_nodes(lines)
    pour = get_pour(nodes)
    result['has_one_pour'] = len(pour) == 1 
    
    # TODO add ring detection? 

    # Are all the nodes reachable
    
    # Requires going trough the network from pour to source and checking if some place is met more than once
    return result

def check_node_network_topology(nodes):
    """ Checks whether all parts of the network are connected to each other.
    """
    result = {}
    # check for duplicate geometries
    duplicated = nodes.duplicated("geometry")
    # When summing booleans, pandas considers True = 1 and False = 0
    result['no_duplicates'] = duplicated.sum() == 0

    pour = nodes[nodes['pour']]
    result['one_pour'] = len(pour) == 1
    pour_id = pour.iloc[0].id

    # Check for loops
    result['no_loops'] = not is_loopy(nodes, pour_id)

    # Check that all nodes are connected
    result['all_reachable'] = len(traverse_graph(nodes, pour_id)) == len(nodes)
    return result

def correct_dam_slivers(nodes, max_removal_length=10):
    """
    Corrects the dam nodes in a network by merging consecutive dam nodes with
    and updating the lengths of the previous node.

    Parameters:
    nodes (pd.DataFrame): A DataFrame containing the network nodes with columns 'id', 'dam', 'next', and 'pituus_m'.

    Returns:
    pd.DataFrame: A DataFrame with the corrected nodes.
    
    """
    nodes = nodes.copy()
    dams = nodes[nodes['dam']]
    dam_to_dam = dams[dams['next'].isin(dams['id'])]
    # only nodes that are too short are removed
    dam_to_dam = dam_to_dam[dam_to_dam["pituus_m"] < max_removal_length]
    nodes_before_removal = nodes[nodes['next'].isin(dam_to_dam['id'])]

    # test if there are dams slivers connected to each other 
    has_multi_sliver_connections = len(dam_to_dam[dam_to_dam['next'].isin(dam_to_dam['id'])]) > 0

    if has_multi_sliver_connections:
        first_sliver_nodes = dam_to_dam[dam_to_dam['next'].isin(dam_to_dam['id'])]
        # There are more than two slivers in a row
        if len(first_sliver_nodes) > 1:
            raise NotImplementedError
            # Add code to handle case with multiple (>= 3) consecutive slivers
        else: # two slivers
            first_sliver_id = first_sliver_nodes.iloc[0].id
            last_sliver_id = dam_to_dam.at[first_sliver_id, 'next']
            length_sum = first_sliver_nodes.pituus_m.sum() + nodes.at[last_sliver_id, 'pituus_m']
           
            src_node_ids = list(nodes[nodes['next'] == first_sliver_id].id)
            dst_node_id = nodes.at[last_sliver_id, 'next']
            nodes.loc[src_node_ids, 'pituus_m'] += length_sum
            nodes.loc[src_node_ids, 'next'] = dst_node_id
            # delete the sliver nodes that have been gone trough
            nodes = nodes.drop([first_sliver_id, last_sliver_id])
    
    # Done again to start removing from a clean slate
    dams = nodes[nodes['dam']]
    dam_to_dam = dams[dams['next'].isin(dams['id'])]
    # only nodes that are too short are removed
    dam_to_dam = dam_to_dam[dam_to_dam["pituus_m"] < max_removal_length]
    nodes_before_removal = nodes[nodes['next'].isin(dam_to_dam['id'])]      
        
    
    for i, row in nodes_before_removal.iterrows():
        
        nodes.loc[i, 'pituus_m'] = nodes.loc[i, 'pituus_m'] + nodes.at[nodes.at[i, 'next'] , 'pituus_m']
        nodes.loc[i, 'next'] = nodes.loc[nodes.at[i, 'next'] , 'next']
    
    nodes = nodes.drop(dam_to_dam.index)
    return nodes

def count_downstream_occurences(nodes, column):
    """
    Counts the amount of occurences in a column of the nodes, downstream from the node
    
    Parameters:
        nodes(GeoDataFrame): a node representations of the river network
        column: (name of column of type boolean)
    """
    assert nodes[column].dtype == 'bool', "This function only supports boolean columns"
    
    flow_paths = get_flow_paths(nodes)
    counts = {}
    
    
    for flow_path in flow_paths:
        count = 0
        for node_id in reversed(flow_path):
            # is the current node the type of feature we are looking for?
            if nodes.at[node_id, column]:
                count += 1
            
            # if node is already in the distances-dict, it doesn't need to be reassigned, because bifurcations are not allowed in the network
            if node_id in counts:
                continue
            
            # Otherwise it is assigned
            else:
                counts[node_id] = count
    nodes[f'ds_{column}_count'] = nodes.index.map(counts)
    return nodes

def drop_z_coordinates(line):
    """
    Removes the z-coordinates from a shapely LineString.

    Parameters:
        line (LineString): A shapely LineString with z-coordinates.
    
    Returns:
        LineString: A new LineString with only x and y coordinates.
    """
    if not isinstance(line, LineString):
        raise ValueError("Input must be a shapely LineString.")

    # Extract only x and y coordinates from the LineString
    xy_coords = [(x, y) for x, y, z in line.coords]
    
    # Return a new 2D LineString
    return LineString(xy_coords)

def d_to_pour(nodes):
    """
    Calculates the distance to the pour point for all nodes based on the pituus_m (distance_m) attribute
    """
    flow_paths = get_flow_paths(nodes)
    
    distances = {}
    for flow_path in flow_paths:
        distance = 0
        # implementation is simpler if the list is reversed (pour to source)
        for node_id in reversed(flow_path): 
            # the pituus_m of current node is added distance
            distance += nodes.at[node_id, 'pituus_m']
            # if node is already in the distances-dict, it doesn't need to be reassigned, because bifurcations are not allowed in the network
            if node_id in distances:
                continue
            
            # Otherwise it is assigned
            else:
                distances[node_id] = distance
    # Setting the values as a new column
    nodes['d_to_pour'] = nodes.index.map(distances)
    return nodes

def get_pour(nodes, tolerance=5):
    """ Fetches the pour point(s) from a river network node representation
    """
    nodes = nodes.copy()
    start_nodes = nodes.loc[nodes.id % 2 == 0]
    end_nodes = nodes.loc[nodes.id % 2 == 1]
    # Avoiding floating point imprecisions with buffering in a small area around point
    buffered_starts = start_nodes.buffer(tolerance)
    # This converts buffered_starts from GeoSeries to GeoDataFrame
    buffered_starts = buffered_starts.reset_index()
    pour = end_nodes.overlay(buffered_starts, keep_geom_type=True, how='difference')
    return pour

def get_flow_paths(nodes):
    """ Searches trough the route that water follows from all of the source points
    and returns the ids of all the routes as list of lists of indexes
    """
    nodes = nodes.copy()
    source_ids = get_source_idxs(nodes)
    paths = []
    pour_id = nodes[nodes['pour']].iloc[0].id
    
    for source in source_ids:
        path = []
        current_node = source
        counter = 0
        # Pour point is self looping, so execution should be stopped when it is reached
        while current_node != pour_id:
            path.append(current_node)
            current_node = nodes.at[current_node, 'next']
            
            counter += 1
            if counter > 1000:
                raise AssertionError("loop limit exceeded")
        # Adding the pour point to the source
        path.append(current_node)
        paths.append(path)
    
    return paths

def get_start_and_end_nodes(lines):
    """ Gets start and end points from dataframe containing linestrings.
    """
    # start points in even rows, end nodes at odd nodes
    nodes = pd.DataFrame()
    j = 0
    for i, row in lines.iterrows():
        coords = list(row['geometry'].coords)
         
        #TODO round to 5 m accuracy
        new_coords = []
        for point_coord in coords:
            new_point_coord = []
            for num in point_coord:
                
                new_point_coord.append(num)
    
            new_coords.append(new_point_coord)
        coords = new_coords
        #start_nodes
        nodes.at[j, 'geometry'] = Point(coords[0])
        nodes.at[j, 'id'] = j
        nodes.at[j, 'pituus_m'] = row['pituus_m']
        j += 1
        #end_nodes.at[i, 'end_point'] = Point(coords[-1])
        nodes.at[j, 'geometry'] = Point(coords[-1])
        nodes.at[j, 'id'] = j
        j += 1
    nodes = geopd.GeoDataFrame(nodes, crs=lines.crs, geometry='geometry')
    nodes['id'] = nodes['id'].astype(int)
    #nodes = add_connections(nodes)
    return nodes

def get_source_idxs(nodes):

    if "in_connect" in nodes.columns:
        pass
    else:
        nodes = in_connectedness(nodes)
    # No incoming connections
    source = nodes[nodes['in_connect'] == 0]
    source_ids = list(source.index)
    return source_ids

def in_connectedness(nodes):
    """ Calculates in-connectedness of each node for a given dataframe, and adds it as a new column
    
    """
    nodes = nodes.copy()
    queue = nodes.copy()
    
    pour_id = nodes[nodes.pour].iloc[0].id
    pour_idx = nodes[nodes.pour].index[0]
    search_buffer  = list(nodes[nodes.next == pour_id].id)

    nodes.at[pour_idx, 'in_connect'] = len(search_buffer)
    
    
    search_buffer.remove(pour_id)
    
    while len(search_buffer) > 0:
        
        new_search_buffer = []
        for i in search_buffer:
            idx = nodes[nodes['id'] == i].index[0]
            incoming_nodes = list(nodes[nodes.next==i].id)
            new_search_buffer += incoming_nodes
            nodes.at[idx, 'in_connect'] = len(incoming_nodes)
            
        search_buffer = new_search_buffer
    nodes['in_connect'] = nodes['in_connect'].astype(int)
    return nodes


def is_loopy(nodes, pour_id):
    """ Traverses trough connections of a graph and 
    checks if any of the encountered nodes has been met previously
    """

    current_node = pour_id
    visited = [pour_id]

    # Getting the next node (and removing already visited locations)
    next_nodes = nodes.loc[nodes['next'] == current_node]
    next_nodes = next_nodes[~next_nodes['id'].isin(visited)]
    search_buffer = list(next_nodes["id"])
    # TODO the inverse of previous line could be saved to check for loops

    duplicates = []
    
    while len(search_buffer) > 0:
        current_node = search_buffer.pop(-1)
        visited.append(current_node)
        
        next_nodes = nodes.loc[nodes['next'] == current_node]
        next_nodes = next_nodes[~next_nodes['id'].isin(visited)]
        search_buffer += list(next_nodes["id"])

        # Getting the locations that have already been visited
        already_visited = next_nodes[next_nodes['id'].isin(visited)]
        duplicates += list(already_visited.id)
        
    return len(duplicates) != 0

def is_source(nodes):
    """
    Adds a column to the node network, telling if it is a source or not
    """
    if "in_connect" in nodes.columns:
        pass
    else:
        nodes = in_connectedness(nodes)
    nodes['source'] = nodes["in_connect"] == 0
    return nodes

def hops_to_pour(nodes):
    """ Calculates number of hops from all nodes to the pour point for the given preprocessed dataframe
    Input: Preprocessed dataframe
    Output: Returns same dataframe but with additional column containing the distance to pour point in hops
    """
    nodes = nodes.copy()
    pour_idx = nodes.index[nodes.pour][0]
    distance = 0
    
    nodes.at[pour_idx, 'hops_to_pour'] =  distance
    
    # Pour point is self looping, so it must be removed before search buffer
    queue = nodes.copy()
    queue_idx = list(queue.index)
    queue_idx.remove(pour_idx)
    queue = queue.loc[queue_idx]
    search_buffer = list(queue[queue['next']==pour_idx].id)
    
    
    # Loop until all the nodes have a distance
    while len(search_buffer) > 0:
        distance += 1
        
        # stuff happens
        for i in search_buffer:
            nodes.at[i, 'hops_to_pour'] =  distance
        
        new_search_buffer = []
        for i in search_buffer:
            new_search_buffer += list(queue[queue['next']==i].id)
            
        search_buffer = new_search_buffer
    #print(nodes[nodes['hops_to_pour'].isna()])
    nodes['hops_to_pour'] = nodes['hops_to_pour'].astype(int)
    return nodes

def merge_short_segments(nodes, max_seg_length):
    """
    Merge graph segments that are shorter than a specified maximum length with their neighboring segments.

    Parameters:
    nodes (GeoDataFrame): A GeoDataFrame representing the directed graph structure. Each node must have an 'id' attribute and a 'next' attribute indicating the next node in the graph.
    max_seg_length (float): The maximum allowable segment length. Segments shorter than half of this length will be merged with their neighbors.

    Returns:
    GeoDataFrame: A new GeoDataFrame with the short segments merged into their neighboring segments.

    Notes:
    - The function creates a copy of the input GeoDataFrame to avoid modifying the original data.
    - Removes nodes that are too short downstream and merges their information upstream of the removed node
    - Pour points are not removed
    - Dams (segments with the 'dam' attribute set to True) are not removed during the merging process.
    """
    nodes = nodes.copy()
    
    too_short = nodes[nodes.pituus_m < max_seg_length / 2]
    
    deleted = []
    
    for j, row in too_short.iterrows():
        
        previous_nodes = nodes[nodes['next'] == row.id]
        # Get the smallest length of the previous segment(s)
        previous_l = previous_nodes.pituus_m.min()
    
        # Get the length of the next segment
        next_node = nodes[nodes.id == row.next]
        # min is a useful way to get the value out
        next_l = next_node.pituus_m.min()
    
        # Check if either one of the above is empty, and branch from there
        if previous_l is np.nan:
            has_previous = False
        else:
            has_previous = True
    
        if next_l is np.nan:
            has_following = False
        else:
            has_following = True

        #print(has_following, has_previous)
        
        # Comparing lengths
        if has_previous:
            if has_following:
                
                # Dams are not removed, this directs to a branch that handles it
                if row.dam:
                    previous_is_shorter = False
                
                # If the following is pour, it cannot be deleted. However, in that case short segment might be merged with upstream
                elif nodes.at[row.next, 'pour']:
                    previous_is_shorter = True

                elif nodes.at[row.next, 'dam']:
                    previous_is_shorter = True
                
                else: 
                    previous_is_shorter = previous_l < next_l

                
            else:
                previous_is_shorter = True
        else:
            previous_is_shorter = False
    
       
        if previous_is_shorter:
            """
            There is a chance that already removed node appears again, if a intersection node was deleted. 
            This can be ignored, because the node was updated previously
            """
            
            if j in deleted:
                continue

            # Pour point is never deleted
            if nodes.at[j, 'pour']:
                continue
            
            # Join happens by deleting the short node and adding it's length to upstream node(s). Skip connections are made
            # nodes that lead to the will-be-removed node
            for i, _ in previous_nodes.iterrows():
                nodes.loc[i, 'pituus_m'] = nodes.loc[i, 'pituus_m'] + nodes.at[nodes.at[i, 'next'] , 'pituus_m']
                nodes.loc[i, 'next'] = nodes.loc[nodes.at[i, 'next'] , 'next']

            nodes = nodes.drop(j)
            deleted.append(j)
        # if the upstream is shorter, If there are multiple upstream nodes, use the minimum length
        # also enter this branch if next node is the pour, because that shall not be deleted
            #the join happens by deleting the short node and adding it's length to upstream node(s)
        
        # the join happens by deleting the downstream node and adding it's length to the current node
        else: 
            """
            There is a chance that already removed node appears again, if a intersection node was deleted. 
            This can be ignored, because the node was updated previously
            """
            
            if row.next in deleted:
                continue
            
            # Pour point is never deleted
            if nodes.at[row.next, 'pour']:
                continue
                
            if nodes.loc[row.next, 'dam']: # Dams are not removed
                #print(f"Node {row.next} is a dam, so it was not removed")
                continue

            
            
            updated_nodes = nodes[nodes['next'] == row.next]
            
            for i, _ in updated_nodes.iterrows():
                nodes.at[i, 'pituus_m'] = nodes.at[i, 'pituus_m'] + nodes.at[row.next, 'pituus_m']
                nodes.at[i, 'next'] = nodes.at[nodes.at[i, 'next'] , 'next']
                
            nodes = nodes.drop(row.next)
            deleted.append(row.next)
    return nodes


def remove_duplicate_geom_nodes(nodes):
    """ Removes duplicate geometries, ensuring that no important information is lost.
    If either one of the duplicates is a lake, dam or pour point, the new node has all of the corresponding features.
    """
    nodes = nodes.copy()
    
    duplicates = nodes[nodes.geometry.duplicated(keep=False)]
    
    if len(duplicates) == 0:
        return nodes
    # Grouping by geometry results in pairs that share the geometry
    grouped = duplicates.groupby("geometry") 
    
    for point, group in grouped:
        
        if len(group) > 2:
            raise NotImplementedError("currently the function only supports pairs of duplicates")
        # Getting the important values. True = 1, False = 0 when summing. 
        dam = group['dam'].sum() > 0
        pour = group['pour'].sum() > 0
        lake = group['lake'].sum() > 0

        # If the snapped nodes are not one after another, but rather lead to the same node
        if group.iloc[0].next == group.iloc[1].next:
            all_previous = nodes[nodes['next'].isin(group.id)]
            all_previous_ids = list(all_previous.id)

            remaining_idx = group.index[0]

            # Setting the connections to the previous
            nodes.loc[all_previous.index, 'next'] = group.at[remaining_idx, 'id']

            nodes.at[remaining_idx, 'pour'] = pour
            nodes.at[remaining_idx, 'dam'] = dam
            nodes.at[remaining_idx, 'lake'] = lake
            
            # Pour point is self looping, so it needs to be handled
            if pour:
                nodes.at[remaining_idx, 'next'] = remaining_idx
            
            nodes.at[remaining_idx, 'pituus_m'] = group.pituus_m.mean(skipna=False)
            nodes = nodes.drop(group.index[1])
            
        else: # The nodes are in sequence
            # getting the id of the first node of the duplicate cluster
            all_previous = nodes[nodes['next'].isin(group.id)]
            #previous = all_previous[~all_previous['id'].isin(group.id)]
            #previous_ids = list(previous.id)
            upstream_node = all_previous[all_previous['id'].isin(group.id)]
            #  TODO this might cause bugs if the length of group > 2
            
            remaining_idx = upstream_node.index[0]
    
            all_following = nodes[nodes['id'].isin(group.next)]
            following_id = all_following[~all_following['id'].isin(group.id)].iloc[0].id
            downstream_node = all_following[all_following['id'].isin(group.id)]
            
            # Updating the new information to upstream node
            nodes.at[remaining_idx, 'pour'] = pour
            nodes.at[remaining_idx, 'dam'] = dam
            nodes.at[remaining_idx, 'lake'] = lake
            if not pour:
                nodes.at[remaining_idx, 'pituus_m'] += downstream_node.iloc[0].pituus_m
            # Pour point is self looping, so it needs to be handled
            if pour:
                nodes.at[remaining_idx, 'next'] = remaining_idx
            else:
                nodes.at[remaining_idx, 'next'] = downstream_node.iloc[0].next
            
            # Deleting the downstream
            nodes = nodes.drop(downstream_node.index[0])

    # The inconnectedness needs to be recalculated
    nodes = in_connectedness(nodes)    
    return nodes

def reset_graph_index(nodes, sort=True):
    """
    TODO: buggy, has to be fixed
    Resets the index in a way that id and next are preserved. Optionally also reorders the graph based on distance to pour
    """
    if 'hops_to_pour' in nodes.columns:
        diameter = nodes['hops_to_pour'].max()
    else:
        nodes = hops_to_pour(nodes)
        diameter = nodes['hops_to_pour'].max()
    
    if sort:
        nodes = nodes.sort_values(['hops_to_pour'])
    
    result = nodes.reset_index(drop=True)
    
    replacement_dict = {}
    for i in range(len(nodes)):
        previous_id = result.at[i, 'id']
        replacement_dict[previous_id] = i 
    
    result = result.replace({'next': replacement_dict}) 
    result = result.replace({'id': replacement_dict}) 
    
    return result

def shreve(nodes):
    """ Calculates the shreve river order for the given preprocessed dataframe
    Input: Preprocessed dataframe
    Output: Returns same dataframe but with additional column containing the shreve order
    """
    if "source" in nodes.columns:
        pass
    else:
        nodes = is_source(nodes)

    flow_paths = get_flow_paths(nodes)
    
    shreve = {} 
    for flow_path in flow_paths:
        for node_id in flow_path:
            # The node has already been visited, and thus can just be incremented
            if node_id in shreve:
                shreve[node_id] += 1
            else:
                shreve[node_id] = 1
    nodes['shreve'] = nodes.index.map(shreve)
    return nodes   

def slice_linestrings(base_gdf, cutter_gdf):
    """
    Slices the LineString geometries in base_gdf at the intersection points with the geometries in cutter_gdf.
    
    Parameters:
        base_gdf (geopd.GeoDataFrame): GeoDataFrame containing LineString geometries to be sliced.
        cutter_gdf (geopd.GeoDataFrame): GeoDataFrame containing geometries used for slicing.
    
    Returns:
        geopd.GeoDataFrame: A new GeoDataFrame containing the sliced LineString segments.
    
    Created with chatgpt, then modified from that, prompt:
    Help me create a function in python that slices a geodataframe that has LineString geometry into segments,
    given another geodataframe that has LineString geometry.
    The first geodataframe should be sliced at the intersection points of the two geodataframes

    """
    # If there is nothing to cut, we can return immediatly
    if len(cutter_gdf) == 0:
        return base_gdf

    sliced_segments = []
    
    # Ensure both GeoDataFrames are in the same CRS
    if base_gdf.crs != cutter_gdf.crs:
        cutter_gdf = cutter_gdf.to_crs(base_gdf.crs)

    
    cutter = cutter_gdf['geometry'].union_all()
    for line in base_gdf.geometry:
        #if not isinstance(line, LineString) or not isinstance(line, MultiLineString):
            #raise ValueError("All geometries in base_gdf must be LineStrings.")
    
        splitted = split(line, cutter)
        split_result = list(splitted.geoms)
                
        sliced_segments.extend(split_result)
        
    # Create a GeoSeries from the sliced segments
    sliced = geopd.GeoDataFrame({"geometry":sliced_segments}, geometry="geometry", crs=base_gdf.crs)

    sliced = sliced.drop_duplicates('geometry')
    return sliced

def subdivide_lines(lines, max_seg_length):
    """ Takes in a dataframe and divides it to segments,
    where most of the segments are usually less than the max_seg_length
    Segments at the end of lines can be slightly longer, because is the last point is alone, it is appended to the last segment
    """
    lines['pituus_m'] = lines['geometry'].length
    
    too_big = []
    small_enough = []
    for i, row in lines.iterrows():
        
        if row.pituus_m < max_seg_length:
            small_enough.append(i)
            continue
    
        too_big.append(i)
    
    too_big_sections = lines.loc[too_big, :]
    subdivided_branches = lines.loc[small_enough, :]
    #subdivided_branches = lines.loc[~lines.index.isin(too_big_sections.index)].reset_index(drop=True)
    too_big_sections = too_big_sections.reset_index(drop=True)
    
    
    
    for i, row in too_big_sections.iterrows():
        # changing the row to df so it can be appended later to the subdivided branches
        row_base = geopd.GeoDataFrame(
            dict(zip(list(row.index), list(row.values))),
            crs=too_big_sections.crs, geometry='geometry', index=[0]) 
        line_string = row.geometry
        coords = list(line_string.coords)
        points = [Point(coord) for coord in coords]
        section_points = []
        distance_sum = 0
        segments = []
        # loop iterates trough all but the last point, which is added at the end anyway.
        start_points = points[:-1]
        for j, point in enumerate(start_points):
            distance_sum += shapely.distance(point, points[j + 1])
            section_points.append(point)
            if distance_sum >= max_seg_length:
                segments.append(LineString(section_points))
                # TODO test if this works
                section_points = [point]
                distance_sum = 0
                
    
        section_points.append(points[-1])
        #if len(section_points) > 1: 
        segments.append(LineString(section_points))
        # if the last point is alone, it is appended to the last segment
        #else:
            #segments[-1] = append_to_linestring(segments[-1], section_points[0])
            
        for segment in segments:
            row_base.at[0, 'geometry'] = segment
            subdivided_branches = pd.concat([subdivided_branches, row_base])
    
    # Segment length needs to be  recalculated
    subdivided_branches['pituus_m'] = subdivided_branches['geometry'].length
    subdivided_branches = subdivided_branches.reset_index(drop=True)
    return subdivided_branches

def traverse_graph(nodes, pour_id):
    """ Traverses trough connections of a graph and 
    collects all the encountered nodes in a list
    """

    current_node = pour_id
    visited = [pour_id]

    # Getting the next node (and removing already visited locations)
    next_nodes = nodes.loc[nodes['next'] == current_node]
    next_nodes = next_nodes[~next_nodes['id'].isin(visited)]
    search_buffer = list(next_nodes["id"])
    # TODO the inverse of previous line could be saved to check for loops
    
    while len(search_buffer) > 0:
        current_node = search_buffer.pop(-1)
        visited.append(current_node)
        
        next_nodes = nodes.loc[nodes['next'] == current_node]
        next_nodes = next_nodes[~next_nodes['id'].isin(visited)]
        search_buffer += list(next_nodes["id"])
        
    return visited

