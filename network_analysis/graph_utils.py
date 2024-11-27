import pandas as pd
import geopandas as geopd
import numpy as np

from shapely import Point


    
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

def correct_dam_slivers(nodes, max_removal_length=100):
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
    nodes_to_removal = nodes[nodes['next'].isin(dam_to_dam['id'])]
    

    for i, row in nodes_to_removal.iterrows():
        nodes.loc[i, 'pituus_m'] = nodes.loc[i, 'pituus_m'] + nodes.at[nodes.at[i, 'next'] , 'pituus_m']
        nodes.loc[i, 'next'] = nodes.loc[nodes.at[i, 'next'] , 'next']
        
    nodes = nodes.drop(dam_to_dam.index)
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



def in_connectedness(nodes):
    """ Calculates in-connectedness of each node for a given dataframe, and adds it as a new column
    
    """
    nodes = nodes.copy()
    queue = nodes.copy()
    
    pour_idx = nodes.index[nodes.pour][0]
    
    search_buffer  = list(nodes[nodes.next == pour_idx].id)

    nodes.at[pour_idx, 'in_connectedness'] = len(search_buffer)

    search_buffer.remove(pour_idx)

    while len(search_buffer) > 0:
        
        new_search_buffer = []
        for i in search_buffer:
            incoming_nodes = list(nodes[nodes.next==i].id)
            new_search_buffer += incoming_nodes
            nodes.at[i, 'in_connectedness'] = len(incoming_nodes)
            
        search_buffer = new_search_buffer
    nodes['in_connectedness'] = nodes['in_connectedness'].astype(int)
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
    
        
        # Comparing lengths
        if has_previous:
            if has_following:
                # Dams are not removed
                if row.dam:
                    previous_is_shorter = False
                previous_is_shorter = previous_l - next_l
            else:
                previous_is_shorter = True
        else:
            previous_is_shorter = False
    
        # Join happens by deleting the short node and adding it's length to upstream node(s). Skip connections are made
        if previous_is_shorter: 
            # nodes that lead to the will-be-removed node
            for i, _ in previous_nodes.iterrows():
                nodes.loc[i, 'pituus_m'] = nodes.loc[i, 'pituus_m'] + nodes.at[nodes.at[i, 'next'] , 'pituus_m']
                nodes.loc[i, 'next'] = nodes.loc[nodes.at[i, 'next'] , 'next']
            
            nodes = nodes.drop(j)
        # if the upstream is shorter, If there are multiple upstream nodes, use the minimum length
        # also enter this branch if next node is the pour, because that shall not be deleted
            #the join happens by deleting the short node and adding it's length to upstream node(s)
        
        # the join happens by deleting the downstream node and adding it's length to the current node
        else: 
            if nodes.at[row.next, 'dam']: # Dams are not removed
                print(f"Node {row.next} is a dam, so it was not removed")
                continue
            updated_nodes = nodes[nodes['next'] == row.next]
            
            for i, _ in updated_nodes.iterrows():
                nodes.at[i, 'pituus_m'] = nodes.at[i, 'pituus_m'] + nodes.at[row.next, 'pituus_m']
                nodes.at[i, 'next'] = nodes.at[nodes.at[i, 'next'] , 'next']
                
            nodes= nodes.drop(row.next)
    return nodes

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

