# Dubai Cycling Paths Explorer (Single File Version)
# This standalone application visualizes Dubai cycling paths from KML files
# with automatic English labels and route suggestions

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap, MeasureControl, Fullscreen
import gradio as gr
import matplotlib.pyplot as plt
import io
import base64
import shapely
from shapely.geometry import Point, LineString, Polygon
import random
import os
import requests
import time
import shutil
from functools import lru_cache
from pathlib import Path
import numpy as np

# ============ FILE MANAGEMENT FUNCTIONS ============

def setup_default_kml():
    """
    Sets up the default KML file to use with the application.
    Creates a data directory if it doesn't exist.
    Returns the path to the default KML file.
    """
    # Create a data directory to store KML files
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Default KML file path
    default_kml_path = data_dir / "default_cycling_paths.kml"
    
    # Check if we need to create a placeholder
    if not default_kml_path.exists():
        print(f"No default KML file found at {default_kml_path}")
        print(f"Please upload a KML file through the interface.")
    
    return default_kml_path

def get_available_kml_files():
    """
    Returns a list of all available KML files in the data directory
    """
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
    
    # Get all KML files in the data directory
    kml_files = list(data_dir.glob("*.kml"))
    return kml_files

def replace_default_kml(new_file_path):
    """
    Replaces the default KML file with a new one
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    default_kml_path = data_dir / "default_cycling_paths.kml"
    
    # First backup the old default
    if default_kml_path.exists():
        backup_path = data_dir / f"backup_{default_kml_path.name}"
        shutil.copy(default_kml_path, backup_path)
        print(f"Backed up previous default KML to {backup_path}")
    
    # Copy the new file as default
    shutil.copy(new_file_path, default_kml_path)
    print(f"New default KML set to {default_kml_path}")
    
    return default_kml_path

# ============ DATA PROCESSING FUNCTIONS ============

def load_kml_data(kml_file):
    """
    Load KML file into a GeoDataFrame with English column names
    """
    try:
        # First attempt - standard loading
        try:
            gdf = gpd.read_file(kml_file, driver='KML')
        except Exception as e:
            print(f"Standard KML loading failed: {e}, trying with fiona...")
            # Second attempt - using fiona driver for more flexibility
            import fiona
            fiona.drvsupport.supported_drivers['KML'] = 'rw'  # Enable KML driver
            gdf = gpd.read_file(kml_file, driver='KML')
        
        # Ensure column names are in English
        # Check for common Arabic or non-English column names
        column_map = {
            'اسم': 'Name',
            'وصف': 'Description',
            'نوع': 'Type',
            'طول': 'Length',
            'المسار': 'Path'
        }
        
        # Rename columns if they exist
        for arabic, english in column_map.items():
            if arabic in gdf.columns:
                gdf = gdf.rename(columns={arabic: english})
        
        # Ensure the CRS is set correctly for Dubai
        if gdf.crs is None:
            gdf.crs = "EPSG:4326"  # WGS84
            
        # Check if we need to explode collections
        collections = gdf[gdf.geometry.type.isin(['GeometryCollection'])]
        if len(collections) > 0:
            print(f"Found {len(collections)} geometry collections, exploding them...")
            exploded = collections.explode(index_parts=True)
            gdf = pd.concat([gdf[~gdf.geometry.type.isin(['GeometryCollection'])], exploded])
        
        # Print column names to help with debugging
        print(f"Available columns: {list(gdf.columns)}")
        
        return gdf, f"Data loaded successfully! Found {len(gdf)} features with geometry types: {gdf.geometry.type.value_counts().to_dict()}"
    except Exception as e:
        print(f"KML loading error: {str(e)}")
        return None, f"Error loading KML file: {str(e)}"

def get_english_name(row):
    """Helper function to extract English name from possible mixed-language fields"""
    # Check for common english name fields
    for field in ['Name', 'name', 'NAME', 'english_name', 'title_en', 'TITLE_EN', 'Name_En']:
        if field in row and pd.notna(row[field]):
            return row[field]
    
    # If no English name found but there's a Description field, use first part
    if 'Description' in row and pd.notna(row['Description']):
        desc = str(row['Description'])
        # Return first 30 chars of description or up to first period
        return desc.split('.')[0][:30] + ('...' if len(desc) > 30 else '')
    
    # Default fallback
    return "Cycling Path"

def extract_points_from_geom(geom):
    """Extract points from any geometry type"""
    points = []
    try:
        if geom.geom_type == 'Point':
            return [geom]
        elif geom.geom_type == 'MultiPoint':
            return list(geom.geoms)
        elif geom.geom_type == 'LineString':
            # Sample points along the line
            return [shapely.geometry.Point(p) for p in list(geom.coords)]
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                points.extend([shapely.geometry.Point(p) for p in list(line.coords)])
            return points
        elif geom.geom_type == 'Polygon':
            # Just use the exterior ring points
            return [shapely.geometry.Point(p) for p in list(geom.exterior.coords)]
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                points.extend([shapely.geometry.Point(p) for p in list(poly.exterior.coords)])
            return points
        else:
            print(f"Unsupported geometry type: {geom.geom_type}")
            return []
    except Exception as e:
        print(f"Error extracting points: {e}")
        return []

# ============ MAP VISUALIZATION FUNCTIONS ============

@lru_cache(maxsize=128)
def reverse_geocode(lat, lon):
    """
    Use OpenStreetMap Nominatim API to get location name from coordinates
    Returns location name in English
    """
    try:
        # Skip the delay for testing purposes
        # In production you should respect Nominatim usage policy
        # time.sleep(1.1)  # Nominatim has a usage limit of 1 request per second
        
        # Set up the API request with English language preference
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=en&zoom=16"
        headers = {
            "User-Agent": "DubaiCyclingPathsExplorer/1.0",  # Identify your application per Nominatim ToS
            "Accept-Language": "en"  # Request English results
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            
            # Try different address components in order of specificity
            if 'address' in data:
                addr = data['address']
                
                # Prefer English name over road address
                if 'name:en' in addr:
                    return addr['name:en']
                
                # Try to get a good descriptive name - neighborhood is often ideal
                for key in ['road', 'neighbourhood', 'suburb', 'quarter', 'district']:
                    if key in addr and addr[key]:
                        return addr[key]
                        
                # Fall back to other components
                for key in ['city_district', 'city', 'town', 'village']:
                    if key in addr and addr[key]:
                        return f"{addr[key]} Area"
            
            # If no good address components, use the display name
            if 'display_name' in data:
                # Just use the first part before the first comma
                display_parts = data['display_name'].split(',')
                if display_parts and display_parts[0]:
                    return display_parts[0].strip()
                    
        return "Cycling Path"  # Default fallback
    except Exception as e:
        print(f"Reverse geocoding error: {e}")
        return "Cycling Path"  # Default fallback

def add_dubai_area_labels(m):
    """
    Add predefined English labels for key areas in Dubai to the map
    """
    # Define key areas in Dubai with their coordinates and names
    key_areas = [
        {"name": "Downtown Dubai", "coords": [25.1972, 55.2744]},
        {"name": "Dubai Marina", "coords": [25.0765, 55.1403]},
        {"name": "Jumeirah Beach", "coords": [25.2048, 55.2708]},
        {"name": "Al Qudra Cycling Track", "coords": [24.9983, 55.3012]},
        {"name": "Nad Al Sheba Cycle Park", "coords": [25.1649, 55.3240]},
        {"name": "Dubai Sports City", "coords": [25.0387, 55.2269]},
        {"name": "Business Bay", "coords": [25.1864, 55.2769]},
        {"name": "Dubai Creek", "coords": [25.2454, 55.3273]},
        {"name": "Jumeirah Lake Towers", "coords": [25.0699, 55.1407]},
        {"name": "Palm Jumeirah", "coords": [25.1124, 55.1390]},
        {"name": "Dubai International City", "coords": [25.1539, 55.4074]},
        {"name": "Al Barsha", "coords": [25.1107, 55.2054]},
        {"name": "Meydan", "coords": [25.1511, 55.2998]},
        {"name": "Dubai Silicon Oasis", "coords": [25.1188, 55.3889]},
        {"name": "Hatta Mountain Bike Trail", "coords": [24.8175, 56.1335]},
        {"name": "Dubai Festival City", "coords": [25.2262, 55.3492]},
        {"name": "Dubai Knowledge Park", "coords": [25.0990, 55.1658]},
        {"name": "Dubai Internet City", "coords": [25.0909, 55.1535]},
        {"name": "Deira", "coords": [25.2697, 55.3070]},
        {"name": "Al Karama", "coords": [25.2480, 55.3020]},
    ]
    
    # Add large, clear labels for each area
    for area in key_areas:
        folium.Marker(
            location=area["coords"],
            icon=folium.DivIcon(
                icon_size=(100, 20),
                icon_anchor=(70, 10),
                html=f'''
                <div style="
                    font-size: 8pt; 
                    font-weight: bold; 
                    color: black; 
                    background-color: rgba(255, 255, 255, 0.9); 
                    padding: 5px 8px; 
                    border-radius: 3px; 
                    border: 3px solid #FF5733;
                    box-shadow: 0 0 15px rgba(0,0,0,0.5); 
                    font-family: Arial, sans-serif; 
                    text-align: center;
                    z-index: 10000;
                ">{area["name"]}</div>
                '''
            )
        ).add_to(m)
    
    return m

def create_map(gdf, map_type="basic", center_lat=25.2048, center_lng=55.2708, zoom_level=11, map_style="positron"):
    """
    Create a Folium map based on the provided GeoDataFrame with explicit English labels
    """
    if gdf is None:
        return None, "No data available. Please load a KML file first."
        
    # Print useful debugging information
    print(f"GeoDataFrame info: {len(gdf)} features")
    print(f"Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")
    print(f"CRS: {gdf.crs}")
    
    # Check for invalid geometries
    try:
        invalid_geoms = gdf[~gdf.geometry.is_valid]
        if len(invalid_geoms) > 0:
            print(f"Found {len(invalid_geoms)} invalid geometries. Attempting to fix...")
            gdf.geometry = gdf.geometry.buffer(0)  # This can fix some invalid geometries
    except Exception as e:
        print(f"Error checking geometry validity: {e}")
    
    # Select map tiles based on style - use options that provide English labels
    if map_style == "positron":
        tiles = "CartoDB positron"
    elif map_style == "voyager":
        tiles = "CartoDB voyager"
    elif map_style == "osm":
        tiles = "OpenStreetMap"
    elif map_style == "satellite":
        # For satellite, add a label layer with English names
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    else:
        tiles = "CartoDB positron"
    
    # Create base map centered on Dubai
    m = folium.Map(location=[center_lat, center_lng], 
                   zoom_start=zoom_level,
                   tiles=tiles,
                   attr='Map tiles by Esri' if map_style == "satellite" else None,
                   prefer_canvas=True)  # prefer_canvas can improve performance
    
    # If using satellite imagery, add a transparent overlay with labels
    if map_style == "satellite":
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='English Labels',
            overlay=True,
            control=True
        ).add_to(m)
    
    # Add title with English text
    title_html = '''
             <h3 align="center" style="font-size:8px"><b>Dubai RTA Cycling Paths</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add different map visualizations based on selected type
    if map_type == "basic":
        # Determine which fields to show in tooltip
        tooltip_fields = []
        tooltip_aliases = []
        
        if 'Name_EN' in gdf.columns:
            tooltip_fields.append('Name_EN')
            tooltip_aliases.append('Name (English):')
        elif 'Name' in gdf.columns:
            tooltip_fields.append('Name')
            tooltip_aliases.append('Name:')
        if 'Description' in gdf.columns:
            tooltip_fields.append('Description')
            tooltip_aliases.append('Description:')
        
        # Add cycling paths to map with clear styling
        folium.GeoJson(
            gdf,
            name='Cycling Paths',
            style_function=lambda x: {
                'color': 'green',
                'weight': 3,
                'fillOpacity': 0.1
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=False,  # Set to False to ensure English formatting
                style='background-color: white; color: black; font-family: Arial, sans-serif; border: 1px solid black; border-radius: 3px; box-shadow: 3px'
            )
        ).add_to(m)
        
        # First check what columns are available that might contain place names
        potential_name_columns = [
            'Name_EN', 'name_en', 'TITLE_EN', 'title_en', 'english_name', 
            'Location_EN', 'location_en', 'Place_EN', 'place_en', 
            'Name', 'name', 'TITLE', 'title', 'Location', 'location'
        ]
        
        # Find available name columns in the dataframe
        available_name_columns = [col for col in potential_name_columns if col in gdf.columns]
        print(f"Available name columns: {available_name_columns}")
        
        # Always use reverse geocoding to ensure English labels
        use_reverse_geocoding = True
        print(f"Using reverse geocoding: {use_reverse_geocoding}")
        
        if not available_name_columns and not use_reverse_geocoding:
            print("No name columns found. Adding a Name_EN column for display.")
            # If no name columns exist, try to create one from Description or other fields
            try:
                # Create temporary English name field if none exists
                gdf['Name_EN'] = gdf.apply(
                    lambda row: get_english_name(row) if callable(globals().get('get_english_name')) else 
                    (row.get('Description', '').split('.')[0][:30] if pd.notna(row.get('Description', '')) else 
                     f"Location {0}"), 
                    axis=1
                )
                available_name_columns = ['Name_EN']
            except Exception as e:
                print(f"Error creating name field: {e}")
        
        # Ensure we create enough labels on the map
        # Label all paths for better visibility of English names
        label_indices = []
        if len(gdf) > 40:
            # Increase the number of labeled paths to 40 for better coverage
            sampling_rate = max(1, len(gdf) // 40)
            label_indices = list(range(0, len(gdf), sampling_rate))
        else:
            # Label all paths if less than 40
            label_indices = list(range(len(gdf)))
            
        # Add markers with location names
        for idx in label_indices:
            if idx >= len(gdf):
                continue
                
            row = gdf.iloc[idx]
            if hasattr(row.geometry, 'centroid'):
                try:
                    # Get location name from first available name column
                    location_name = None
                    for col in available_name_columns:
                        if pd.notna(row.get(col, None)):
                            location_name = row[col]
                            break
                    
                    # If no name found or using reverse geocoding, use the API
                    if not location_name or use_reverse_geocoding:
                        # Determine a good point to use for reverse geocoding
                        geocode_point = None
                        
                        if row.geometry.geom_type == 'LineString':
                            # Use midpoint for LineString for better geocoding results
                            coords = list(row.geometry.coords)
                            if len(coords) > 0:
                                midpoint_idx = len(coords) // 2
                                geocode_point = coords[midpoint_idx]
                                # Add logging to help debug
                                print(f"Geocoding LineString at point: {geocode_point}")
                        elif row.geometry.geom_type == 'MultiLineString':
                            # Use midpoint of longest segment for MultiLineString
                            longest_line = max(row.geometry.geoms, key=lambda line: line.length)
                            midpoint_idx = len(list(longest_line.coords)) // 2
                            geocode_point = list(longest_line.coords)[midpoint_idx]
                        else:
                            # Use centroid for other geometry types
                            geocode_point = (row.geometry.centroid.x, row.geometry.centroid.y)
                        
                        if geocode_point:
                            # Perform reverse geocoding using the Nominatim API
                            # Note: lat=y, lon=x in geographic coordinates
                            location_name = reverse_geocode(geocode_point[1], geocode_point[0])
                        else:
                            # Fallback if we couldn't get a proper point
                            location_name = f"Dubai Cycling Path"
                    
                    # If we still have no name, try to extract from other fields
                    if not location_name:
                        if 'Description' in gdf.columns and pd.notna(row['Description']):
                            location_name = row['Description'].split('.')[0][:30]
                        else:
                            location_name = f"Location {idx+1}"
                    
                    # For cycling paths, calculate a good position for the label
                    if row.geometry.geom_type in ['LineString', 'MultiLineString']:
                        # Try to get midpoint rather than centroid for better label placement
                        if row.geometry.geom_type == 'LineString':
                            midpoint_idx = len(list(row.geometry.coords)) // 2
                            label_point = list(row.geometry.coords)[midpoint_idx]
                            location = [label_point[1], label_point[0]]  # y, x (lat, lon)
                        else:
                            # For multilinestring, use the longest segment's midpoint
                            longest_line = max(row.geometry.geoms, key=lambda line: line.length)
                            midpoint_idx = len(list(longest_line.coords)) // 2
                            label_point = list(longest_line.coords)[midpoint_idx]
                            location = [label_point[1], label_point[0]]
                    else:
                        # For other geometry types, use centroid
                        location = [row.geometry.centroid.y, row.geometry.centroid.x]
                    
                    # Ensure more labels are visible - make length threshold smaller
                    # Only skip extremely short paths
                    skip_label = False
                    if row.geometry.geom_type in ['LineString', 'MultiLineString'] and hasattr(row.geometry, 'length'):
                        if row.geometry.length < 0.005:  # Only skip extremely short paths (reduced threshold)
                            skip_label = True
                    
                    if not skip_label:
                        # Add the label with more prominent styling
                        folium.Marker(
                            location=location,
                            icon=folium.DivIcon(
                                icon_size=(120, 25),
                                icon_anchor=(60, 12),
                                html=f'''
                                <div style="
                                    font-size: 4pt; 
                                    font-weight: bold; 
                                    color: black; 
                                    background-color: rgba(255, 255, 255, 0.9); 
                                    padding: 3px 6px; 
                                    border-radius: 3px; 
                                    border: 2px solid #4CAF50;
                                    box-shadow: 0 0 10px rgba(0,0,0,0.3); 
                                    font-family: Arial, sans-serif; 
                                    max-width: 110px; 
                                    text-align: center;
                                    overflow: hidden; 
                                    text-overflow: ellipsis; 
                                    white-space: nowrap;
                                    z-index: 9999;
                                ">{location_name}</div>
                                '''
                            )
                        ).add_to(m)
                except Exception as e:
                    print(f"Error adding location label for {idx}: {e}")
                    continue
        
        # Add predefined Dubai area labels for key locations
        add_dubai_area_labels(m)
                    
    elif map_type == "cluster":
        # Extract points from paths for clustering
        points = []
        for _, row in gdf.iterrows():
            try:
                if row.geometry.geom_type == 'LineString':
                    for point in row.geometry.coords:
                        points.append([point[1], point[0]])  # Folium expects [lat, lng]
                elif row.geometry.geom_type == 'MultiLineString':
                    for line in row.geometry:
                        for point in line.coords:
                            points.append([point[1], point[0]])
                elif row.geometry.geom_type == 'Point':
                    points.append([row.geometry.y, row.geometry.x])
                elif row.geometry.geom_type == 'MultiPoint':
                    for point in row.geometry.geoms:
                        points.append([point.y, point.x])
                elif row.geometry.geom_type == 'Polygon':
                    for point in row.geometry.exterior.coords:
                        points.append([point[1], point[0]])
                elif row.geometry.geom_type == 'MultiPolygon':
                    for polygon in row.geometry.geoms:
                        for point in polygon.exterior.coords:
                            points.append([point[1], point[0]])
            except Exception as e:
                print(f"Skipping geometry due to error: {e}")
                continue
        
        # Add marker cluster if points are available
        if points:
            marker_cluster = MarkerCluster(
                name="Path Points",
                overlay=True,
                control=True,
                icon_create_function='''
                function(cluster) {
                    return L.divIcon({
                        html: '<div style="background-color: rgba(0, 128, 0, 0.7); color: white; border-radius: 25%; width: 15px; height: 15px; line-height: 15px; text-align: center; font-family: Arial, sans-serif;">' + cluster.getChildCount() + '</div>',
                        className: 'marker-cluster',
                        iconSize: L.point(12, 12)
                    });
                }
                '''
            ).add_to(m)
            
            # Sample points to reduce density - take every 10th point or at least 100 points
            sampling_rate = max(1, len(points) // 100)
            for point in points[::sampling_rate]:
                try:
                    folium.CircleMarker(
                        location=point,
                        radius=2,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        popup='Cycling Path Point',  # English label
                    ).add_to(marker_cluster)
                except Exception as e:
                    print(f"Error adding marker: {e}")
                    continue
        
        # Also add the paths in a lighter color
        try:
            folium.GeoJson(
                gdf,
                name='Cycling Paths',
                style_function=lambda x: {
                    'color': 'green',
                    'weight': 2,
                    'opacity': 0.5
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['Name_EN'] if 'Name_EN' in gdf.columns else ['Name'] if 'Name' in gdf.columns else [],
                    aliases=['Name (English):'] if 'Name_EN' in gdf.columns else ['Name:'] if 'Name' in gdf.columns else [],
                    localize=False,  # English formatting
                    style='font-family: Arial, sans-serif;'
                )
            ).add_to(m)
        except Exception as e:
            print(f"Error adding GeoJson layer: {e}")
            # Fallback to simpler representation
            for _, row in gdf.iterrows():
                try:
                    folium.GeoJson(
                        row.geometry.__geo_interface__,
                        style_function=lambda x: {
                            'color': 'green',
                            'weight': 2,
                            'opacity': 0.5
                        }
                    ).add_to(m)
                except:
                    continue
    
    elif map_type == "heatmap":
        print("Attempting to create heatmap visualization...")
        
        # Create a simpler alternative heatmap implementation
        try:
            # Process each geometry and collect points
            all_points = []
            for idx, row in gdf.iterrows():
                try:
                    points = extract_points_from_geom(row.geometry)
                    all_points.extend(points)
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
            
            print(f"Extracted {len(all_points)} points for heatmap")
            
            # Sample points if there are too many
            max_points = 3000  # Reduced max points for better performance
            if len(all_points) > max_points:
                all_points = random.sample(all_points, max_points)
                print(f"Sampled down to {len(all_points)} points")
            
            # Create heat data in the format needed by folium
            heat_data = [[point.y, point.x, 1] for point in all_points
                          if -90 <= point.y <= 90 and -180 <= point.x <= 180]
            
            if heat_data:
                print(f"Final heatmap data points: {len(heat_data)}")
                HeatMap(
                    heat_data, 
                    min_opacity=0.5,
                    radius=10, 
                    blur=15,
                    name="Cycling Density"  # English name
                ).add_to(m)
                print("Heatmap successfully added to map")
            else:
                print("No valid heat data points generated")
                return m._repr_html_(), "Could not generate heatmap. No valid points extracted from geometry."
                
        except Exception as e:
            print(f"Overall heatmap error: {e}")
            return m._repr_html_(), f"Error creating heatmap: {str(e)}. Try another visualization type."
    
    # Add English language legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 10px; right: 10px; width: 40px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; font-size:8px;
                font-family: Arial, sans-serif;
                padding: 5px">
        <span style="color: green;"><b>&#9473;</b></span> Cycling Path<br>
        <span style="color: blue;"><b>&#9679;</b></span> Path Point<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
   # Add English language layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add scale bar with English units
    MeasureControl(
        position='bottomleft', 
        primary_length_unit='kilometers',
        secondary_length_unit='miles'
    ).add_to(m)
    
    # Add a note about the reverse geocoding source
    attribution_html = '''
    <div style="position: fixed; 
                bottom: 5px; left: 5px; 
                background-color: rgba(255, 255, 255, 0.7); 
                font-size:10px; font-family: Arial, sans-serif;
                padding: 2px 5px; border-radius: 3px; z-index:9998;">
        Location names from OpenStreetMap Nominatim
    </div>
    '''
    m.get_root().html.add_child(folium.Element(attribution_html))
    
    # Save map to HTML string
    map_html = m._repr_html_()
    
    return map_html, "Map created successfully with English location names!"

# ============ ANALYSIS FUNCTIONS ============

def get_stats(gdf):
    """
    Generate statistics about the cycling paths
    """
    if gdf is None:
        return "No data available. Please load a KML file first."
    
    stats = []
    
    # Count total number of paths
    total_paths = len(gdf)
    stats.append(f"Total number of cycling paths: {total_paths}")
    
    # Calculate total length of paths (in km)
    # First ensure the data is in a projected CRS for accurate measurements
    if not gdf.crs or gdf.crs.is_geographic:
        # Convert to UTM zone 43N (appropriate for Dubai)
        gdf_proj = gdf.to_crs(epsg=32643)
    else:
        gdf_proj = gdf
    
    total_length = gdf_proj.geometry.length.sum() / 1000  # Convert to km
    stats.append(f"Total length of cycling paths: {total_length:.2f} km")
    
    # Check if there are specific attributes we can analyze
    if 'Name' in gdf.columns:
        named_paths = gdf['Name'].notna().sum()
        stats.append(f"Number of named paths: {named_paths}")
    
    if 'Description' in gdf.columns:
        with_description = gdf['Description'].notna().sum()
        stats.append(f"Number of paths with descriptions: {with_description}")
    
    return "\n".join(stats)

def plot_path_distribution(gdf):
    """
    Create a visualization of path distribution
    """
    if gdf is None:
        return None, "No data available. Please load a KML file first."
    
    plt.figure(figsize=(10, 6))
    
    # Project data for visualization
    gdf_proj = gdf.to_crs(epsg=32643)
    
    # Calculate length of each path in km
    gdf_proj['length_km'] = gdf_proj.geometry.length / 1000
    
    # Create histogram of path lengths
    plt.hist(gdf_proj['length_km'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Cycling Path Lengths in Dubai')
    plt.xlabel('Path Length (km)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Create data URI for image
    data_uri = base64.b64encode(buf.read()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{data_uri}"/>'
    
    return img_tag, "Distribution plot created successfully!"

# ============ CYCLING ROUTE SUGGESTION FUNCTIONS ============

def find_cycling_route(gdf, start_area, target_distance_km, tolerance_km=2.0, use_better_connectivity=False):
    """
    Find a cycling route based on desired starting area and approximate distance
    
    Parameters:
    - gdf: GeoDataFrame containing cycling paths
    - start_area: String name of the starting area (e.g., "Dubai Marina", "Downtown Dubai")
    - target_distance_km: Desired cycling distance in kilometers
    - tolerance_km: Acceptable deviation from target distance in kilometers
    - use_better_connectivity: Whether to use enhanced connectivity algorithm
    
    Returns:
    - route_gdf: GeoDataFrame containing the suggested route
    - stats: Dictionary with route statistics
    """
    import numpy as np
    from shapely.geometry import Point
    import geopandas as gpd
    import folium
    
    if gdf is None or len(gdf) == 0:
        return None, {"error": "No cycling path data available"}
    
    # Ensure the GeoDataFrame has a CRS set
    if gdf.crs is None:
        print("Warning: Input GeoDataFrame has no CRS. Setting to WGS84 (EPSG:4326)")
        gdf = gdf.set_crs("EPSG:4326")
    
    # Define known Dubai locations
    known_locations = {
        "downtown dubai": (25.1972, 55.2744),
        "dubai marina": (25.0765, 55.1403),
        "jumeirah beach": (25.2048, 55.2708),
        "al qudra": (24.9983, 55.3012),
        "nad al sheba": (25.1649, 55.3240),
        "sports city": (25.0387, 55.2269),
        "business bay": (25.1864, 55.2769),
        "dubai creek": (25.2454, 55.3273),
        "jumeirah lake towers": (25.0699, 55.1407),
        "jlt": (25.0699, 55.1407),
        "palm jumeirah": (25.1124, 55.1390),
        "international city": (25.1539, 55.4074),
        "al barsha": (25.1107, 55.2054),
        "meydan": (25.1511, 55.2998),
        "silicon oasis": (25.1188, 55.3889),
        "hatta": (24.8175, 56.1335),
        "festival city": (25.2262, 55.3492),
        "knowledge park": (25.0990, 55.1658),
        "internet city": (25.0909, 55.1535),
        "deira": (25.2697, 55.3070),
        "al karama": (25.2480, 55.3020)
    }
    
    # Normalize input
    start_area_lower = start_area.lower().strip()
    
    # Find the starting point coordinates
    start_coords = None
    for area, coords in known_locations.items():
        if area in start_area_lower or start_area_lower in area:
            start_coords = coords
            break
    
    if start_coords is None:
        return None, {"error": f"Could not find location: {start_area}. Try one of: {', '.join(known_locations.keys())}"}
    
    # Create a copy of the GeoDataFrame
    gdf_copy = gdf.copy()
    
    # Ensure the GeoDataFrame is properly projected for distance calculations
    if gdf_copy.crs.is_geographic:
        gdf_proj = gdf_copy.to_crs(epsg=32643)  # UTM zone 43N for Dubai
    else:
        gdf_proj = gdf_copy
    
    # Calculate length of each path in km
    gdf_proj['length_km'] = gdf_proj.geometry.length / 1000
    
    # Create a point representing the starting location
    start_point = Point(start_coords[1], start_coords[0])  # lon, lat
    
    # Create a GeoDataFrame for the start point with explicit CRS
    start_gdf = gpd.GeoDataFrame(geometry=[start_point], crs="EPSG:4326")
    
    # Convert starting point to the same projection
    start_point_proj = start_gdf.to_crs(gdf_proj.crs).geometry[0]
    
    # Calculate distance from each path to the starting point
    gdf_proj['distance_to_start'] = gdf_proj.geometry.distance(start_point_proj) / 1000  # Convert to km
    
    # Sort paths by proximity to starting point
    gdf_proj = gdf_proj.sort_values('distance_to_start')
    
    # If using better connectivity, try that first
    if use_better_connectivity:
        try:
            route_gdf = find_better_connected_paths(gdf_proj, start_point_proj, target_distance_km, tolerance_km)
            if route_gdf is not None and len(route_gdf) > 0:
                # Calculate statistics
                total_distance = route_gdf['length_km'].sum()
                num_segments = len(route_gdf)
                
                stats = {
                    "total_distance_km": round(total_distance, 2),
                    "number_of_segments": num_segments,
                    "start_area": start_area,
                    "target_distance_km": target_distance_km,
                    "route_type": "Connected paths",
                    "avg_segment_length": round(total_distance / num_segments, 2),
                    "longest_segment": round(route_gdf['length_km'].max(), 2),
                    "estimated_time": f"{round(total_distance / 15, 1)} - {round(total_distance / 20, 1)} hours"
                }
                
                # Convert back to original CRS
                route_gdf = route_gdf.to_crs(gdf.crs)
                
                return route_gdf, stats
        except Exception as e:
            print(f"Error using better connectivity method: {e}")
            # Fall back to simpler method
    
    # Use the simpler approach as a fallback
    # Just select paths near the starting point up to the target distance
    nearby_paths = gdf_proj[gdf_proj['distance_to_start'] < 5]  # Within 5km
    
    if len(nearby_paths) == 0:
        return None, {"error": f"No cycling paths found near {start_area}"}
    
    # Sort by distance to starting point
    nearby_paths = nearby_paths.sort_values('distance_to_start')
    
    # Select paths up to target distance
    selected_paths = []
    current_distance = 0
    
    for _, path in nearby_paths.iterrows():
        if current_distance >= target_distance_km - tolerance_km:
            break
        
        selected_paths.append(path)
        current_distance += path['length_km']
    
    if not selected_paths:
        return None, {"error": f"Could not find enough paths to reach {target_distance_km}km near {start_area}"}
    
    # Create route GeoDataFrame
    route_gdf = gpd.GeoDataFrame(selected_paths)
    
    # Make sure it has the same CRS as the original
    route_gdf = route_gdf.set_crs(gdf_proj.crs)
    
    # Convert back to original CRS
    route_gdf = route_gdf.to_crs(gdf.crs)
    
    # Calculate route statistics
    total_distance = route_gdf['length_km'].sum()
    num_segments = len(route_gdf)
    
    stats = {
        "total_distance_km": round(total_distance, 2),
        "number_of_segments": num_segments,
        "start_area": start_area,
        "target_distance_km": target_distance_km,
        "route_type": "Nearby paths",
        "avg_segment_length": round(total_distance / num_segments, 2),
        "longest_segment": round(route_gdf['length_km'].max(), 2),
        "estimated_time": f"{round(total_distance / 15, 1)} - {round(total_distance / 20, 1)} hours",
        "note": "This route consists of separate path segments near your starting location."
    }
    
    return route_gdf, stats
    # Add this to the stats dictionary in find_cycling_route
    stats = {
        "total_distance_km": round(total_distance, 2),
        "number_of_segments": num_segments,
        "start_area": start_area,
        "target_distance_km": target_distance_km,
        "route_type": route_type,
        "connectivity": f"{connected_segments}/{num_segments} segments connected",
        "avg_segment_length": round(total_distance / num_segments, 2),
        "longest_segment": round(route_gdf['length_km'].max(), 2),
        "estimated_time": f"{round(total_distance / 15, 1)} - {round(total_distance / 20, 1)} hours"  # Assuming 15-20 km/h
    }

def find_better_connected_paths(gdf_proj, start_point_proj, target_distance_km, tolerance_km=2.0):
    """Enhanced function to find better connected cycling routes"""
    
    # Find the nearest path to start point
    nearest_idx = gdf_proj.distance(start_point_proj).idxmin()
    nearest_path = gdf_proj.loc[nearest_idx]
    
    # Start the route
    route_paths = [nearest_path]
    current_distance = nearest_path['length_km']
    
    # Make working copy without the path we've already selected
    remaining_paths = gdf_proj.drop(nearest_idx).copy()
    
    # Keep track of the current endpoint geometry
    current_geom = nearest_path.geometry
    
    # Maximum connection distance in kilometers
    max_connection_dist = 0.3  # 300 meters
    
    # Loop until we reach target distance or run out of connected paths
    while current_distance < target_distance_km and len(remaining_paths) > 0:
        # Calculate distance from each remaining path to our current position
        remaining_paths['distance_to_route'] = remaining_paths.geometry.distance(current_geom) / 1000
        
        # Find paths that are close enough to be considered connected
        close_paths = remaining_paths[remaining_paths['distance_to_route'] < max_connection_dist]
        
        if len(close_paths) == 0:
            # No more connected paths - increase search radius and try again
            if max_connection_dist < 1.0:  # Cap at 1km
                max_connection_dist += 0.2
                continue
            else:
                break  # Give up if no paths within 1km
        
        # Choose the next path - prioritize longer paths
        next_path = close_paths.sort_values('length_km', ascending=False).iloc[0]
        
        # Add to our route
        route_paths.append(next_path)
        current_distance += next_path['length_km']
        
        # Update current position and remove path from consideration
        current_geom = next_path.geometry
        remaining_paths = remaining_paths.drop(next_path.name)
    
    # If we didn't get close to the target distance, find some nearby paths to fill in
    min_target = target_distance_km - tolerance_km
    if current_distance < min_target and len(remaining_paths) > 0:
        # Find nearby paths to add
        remaining_paths['distance_to_start'] = remaining_paths.geometry.distance(start_point_proj) / 1000
        nearby_paths = remaining_paths[remaining_paths['distance_to_start'] < 3].sort_values('length_km', ascending=False)
        
        for _, path in nearby_paths.iterrows():
            if current_distance >= min_target:
                break
            route_paths.append(path)
            current_distance += path['length_km']
    
    if len(route_paths) == 0:
        return None
    
    # Create a GeoDataFrame with the route
    import geopandas as gpd
    route_gdf = gpd.GeoDataFrame(route_paths)
    route_gdf = route_gdf.set_crs(gdf_proj.crs)
    
    return route_gdf

def find_connected_paths(gdf, start_point, target_distance_km, tolerance_km=2.0):
    """
    Find connected paths that form a route of approximately the target distance
    
    Parameters:
    - gdf: GeoDataFrame with cycling paths
    - start_point: Shapely Point object representing starting location
    - target_distance_km: Desired cycling distance in kilometers
    - tolerance_km: Acceptable deviation from target distance
    
    Returns:
    - route_gdf: GeoDataFrame with connected paths forming the route
    """
    from shapely.ops import nearest_points
    import geopandas as gpd
    import numpy as np
    
    # Copy the GeoDataFrame to avoid modifying the original
    gdf_work = gdf.copy()
    
    # Find the nearest path to the starting point
    nearest_idx = gdf_work.distance(start_point).idxmin()
    nearest_path = gdf_work.loc[nearest_idx]
    
    # Start building the route with the nearest path
    route_paths = [nearest_path]
    current_distance = nearest_path['length_km']
    
    # Remove the path we just added from consideration
    gdf_work = gdf_work.drop(nearest_idx)
    
    # Keep track of the last endpoint
    last_geom = nearest_path.geometry
    
    # Find paths that connect to our growing route
    max_attempts = 50  # Prevent infinite loops
    attempts = 0
    
    while current_distance < target_distance_km - tolerance_km and len(gdf_work) > 0 and attempts < max_attempts:
        attempts += 1
        
        # Calculate distance from each remaining path to the last path in our route
        gdf_work['dist_to_route'] = gdf_work.geometry.distance(last_geom)
        
        # Find connected or very close paths (within 100 meters)
        connected_paths = gdf_work[gdf_work['dist_to_route'] < 0.1]  # 100 meters
        
        if len(connected_paths) == 0:
            break  # No more connected paths
        
        # Sort by length to prioritize longer paths
        connected_paths = connected_paths.sort_values('length_km', ascending=False)
        
        # Take the first connected path
        next_path_idx = connected_paths.index[0]
        next_path = gdf_work.loc[next_path_idx]
        
        # Add to our route
        route_paths.append(next_path)
        current_distance += next_path['length_km']
        
        # Update the last geometry and remove the path from consideration
        last_geom = next_path.geometry
        gdf_work = gdf_work.drop(next_path_idx)
    
    # Check if we've reached at least the minimum desired distance
    min_desired = target_distance_km - tolerance_km
    
    if current_distance < min_desired:
        # Not enough connected paths to reach the target distance
        return None
    
    # Create a GeoDataFrame with our route
    route_gdf = gpd.GeoDataFrame(route_paths)
    
    return route_gdf

def create_route_map(gdf, route_gdf, start_area, stats):
    """
    Create a Folium map with the suggested cycling route
    
    Parameters:
    - gdf: Original GeoDataFrame with all cycling paths
    - route_gdf: GeoDataFrame with the selected route
    - start_area: Starting area name
    - stats: Dictionary with route statistics
    
    Returns:
    - map_html: HTML string of the Folium map
    """
    import folium
    from folium import plugins
    
    # Get coordinates for starting area
    known_locations = {
        "downtown dubai": (25.1972, 55.2744),
        "dubai marina": (25.0765, 55.1403),
        "jumeirah beach": (25.2048, 55.2708),
        "al qudra": (24.9983, 55.3012),
        "nad al sheba": (25.1649, 55.3240),
        "sports city": (25.0387, 55.2269),
        "business bay": (25.1864, 55.2769),
        "dubai creek": (25.2454, 55.3273),
        "jumeirah lake towers": (25.0699, 55.1407),
        "jlt": (25.0699, 55.1407),
        "palm jumeirah": (25.1124, 55.1390),
        "international city": (25.1539, 55.4074),
        "al barsha": (25.1107, 55.2054),
        "meydan": (25.1511, 55.2998),
        "silicon oasis": (25.1188, 55.3889),
        "hatta": (24.8175, 56.1335),
        "festival city": (25.2262, 55.3492),
        "knowledge park": (25.0990, 55.1658),
        "internet city": (25.0909, 55.1535),
        "deira": (25.2697, 55.3070),
        "al karama": (25.2480, 55.3020)
    }
    
    # Find coordinates for the starting area
    start_coords = None
    for area, coords in known_locations.items():
        if area in start_area.lower() or start_area.lower() in area:
            start_coords = coords
            break
    
    if start_coords is None:
        # Default to Dubai center if no match
        start_coords = (25.2048, 55.2708)
    
    # Create base map
    m = folium.Map(location=start_coords, zoom_start=13, tiles="CartoDB positron")
    
    # Add all cycling paths in light gray
    folium.GeoJson(
        gdf,
        name='All Cycling Paths',
        style_function=lambda x: {
            'color': '#CCCCCC',
            'weight': 2,
            'opacity': 0.5
        }
    ).add_to(m)
    
    # Add the suggested route with bright colors
    route = folium.GeoJson(
        route_gdf,
        name='Suggested Route',
        style_function=lambda x: {
            'color': '#FF5500',
            'weight': 5,
            'opacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['Name_EN'] if 'Name_EN' in route_gdf.columns else ['Name'] if 'Name' in route_gdf.columns else [],
            aliases=['Path Name:'] if 'Name_EN' in route_gdf.columns or 'Name' in route_gdf.columns else [],
            localize=False,
            sticky=True,
            style='background-color: white; color: black; font-family: Arial, sans-serif; border: 1px solid black; border-radius: 3px;'
        )
    ).add_to(m)
    
    # Add starting marker
    folium.Marker(
        location=start_coords,
        popup=f"Starting point: {start_area}",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    # Add distance markers every 5km
    route_gdf['cumulative_distance'] = route_gdf['length_km'].cumsum()
    
    # Extract points at approximate 5km intervals
    distance_markers = []
    current_marker = 5  # First marker at 5km
    
    for i, row in route_gdf.iterrows():
        if row['cumulative_distance'] >= current_marker:
            # Find a point on this path segment
            if hasattr(row.geometry, 'interpolate'):
                # For LineString
                point = row.geometry.interpolate(0.5, normalized=True)
                distance_markers.append((point.x, point.y, current_marker))
                current_marker += 5
            elif hasattr(row.geometry, 'geoms') and len(row.geometry.geoms) > 0:
                # For MultiLineString, use the first line
                line = row.geometry.geoms[0]
                point = line.interpolate(0.5, normalized=True)
                distance_markers.append((point.x, point.y, current_marker))
                current_marker += 5
    
    # Add distance markers
    for lon, lat, dist in distance_markers:
        folium.Marker(
            location=[lat, lon],
            popup=f"{dist}km mark",
            icon=folium.Icon(color='blue', icon='flag', prefix='fa')
        ).add_to(m)
    
    # Add stats box
    stats_html = f'''
    <div style="position: fixed; 
                top: 5px; right: 5px; 
                background-color: white; 
                padding: 5px; 
                border-radius: 2px;
                box-shadow: 0 0 5px rgba(0,0,0,0.3);
                z-index: 1000;
                font-family: Arial, sans-serif;
                max-width: 100px;">
        <h4>Suggested Cycling Route</h4>
        <ul style="padding-left: 10px; list-style-type: none;">
            <li><b>Starting Area:</b> {stats['start_area']}</li>
            <li><b>Total Distance:</b> {stats['total_distance_km']} km</li>
            <li><b>Segments:</b> {stats['number_of_segments']}</li>
            <li><b>Route Type:</b> {stats['route_type']}</li>
            {f"<li><b>Note:</b> {stats['note']}</li>" if 'note' in stats else ""}
        </ul>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen option
    plugins.Fullscreen().add_to(m)
    
    # Add measure tool for distances
    plugins.MeasureControl(
        position='bottomleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles'
    ).add_to(m)
    
    return m._repr_html_()

# ============ MAIN APPLICATION FUNCTIONS ============

def process_file(file, map_type, zoom_level, map_style="positron"):
    """Process uploaded KML file and generate map"""
    if file is None:
        return None, "Please upload a KML file.", None, None
    
    # Load the data
    gdf, load_msg = load_kml_data(file.name)
    
    if gdf is None:
        return None, load_msg, None, None
        
    # Store in global variable for future updates
    globals()['gdf'] = gdf
        
    # Try to add an English name column if not present
    if 'Name_EN' not in gdf.columns:
        try:
            gdf['Name_EN'] = gdf.apply(get_english_name, axis=1)
            print("Added English names column")
        except Exception as e:
            print(f"Could not generate English names: {e}")
    
    # Create map
    map_html, map_msg = create_map(gdf, map_type=map_type, zoom_level=int(zoom_level), map_style=map_style)
    
    # Get statistics
    stats = get_stats(gdf)
    
    # Create distribution plot
    plot_html, _ = plot_path_distribution(gdf)
    
    return map_html, f"{load_msg}\n{map_msg}", stats, plot_html

def update_map(map_type, zoom_level, map_style="positron"):
    """Update the map visualization based on selected options"""
    if 'gdf' not in globals() or globals()['gdf'] is None:
        return None, "Please upload a KML file first.", None, None
    
    # Create new map with current options
    map_html, map_msg = create_map(globals()['gdf'], map_type=map_type, 
                                   zoom_level=int(zoom_level), map_style=map_style)
    
    # Get statistics
    stats = get_stats(globals()['gdf'])
    
    # Create distribution plot
    plot_html, _ = plot_path_distribution(globals()['gdf'])
    
    return map_html, "Map updated successfully", stats, plot_html

def dubai_cycling_app_with_route_suggester():
    """Main application function with route suggestion feature"""
    # Initialize global variables
    globals()['gdf'] = None
    
    # Setup auto-loading of default KML file
    default_kml_path = setup_default_kml()
    print(f"Default KML path: {default_kml_path}")
    
    # Define interface
    with gr.Blocks(title="Dubai Cycling Paths Explorer") as app:
        gr.Markdown("# Dubai RTA Cycling Paths Explorer")
        
        # Create tabs for main interface
        with gr.Tabs():
            with gr.Tab("Map Viewer"):
                gr.Markdown("Upload a KML file containing Dubai's RTA cycling path data or use the default.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Add a dropdown to select from available KML files
                        available_files = get_available_kml_files()
                        available_file_names = [f.name for f in available_files]
                        
                        if default_kml_path and default_kml_path.exists():
                            default_file_name = default_kml_path.name
                            if default_file_name not in available_file_names:
                                available_file_names.insert(0, default_file_name)
                        
                        file_dropdown = gr.Dropdown(
                            choices=available_file_names,
                            value=default_kml_path.name if default_kml_path.exists() else None,
                            label="Select Existing KML File",
                            interactive=True
                        )
                        
                        # Regular file upload for new files
                        file_input = gr.File(label="Or Upload New KML File")
                        
                        # Add checkbox to set uploaded file as default
                        set_as_default = gr.Checkbox(
                            label="Set uploaded file as default",
                            value=False
                        )
                        
                        map_type = gr.Radio(
                            ["basic", "cluster", "heatmap"], 
                            label="Map Visualization Type",
                            value="basic"
                        )
                        zoom_level = gr.Slider(
                            minimum=9, 
                            maximum=15, 
                            value=11, 
                            step=1, 
                            label="Zoom Level"
                        )
                        map_style = gr.Radio(
                            ["positron", "voyager", "osm", "satellite"], 
                            label="Map Style",
                            value="positron",
                            info="positron: Clean map with English labels, voyager: Detailed map, osm: Standard OpenStreetMap, satellite: Satellite imagery"
                        )
                        load_btn = gr.Button("Load Selected KML & Generate Map")
                        update_btn = gr.Button("Update Map")
                        status = gr.Textbox(label="Status")
                        stats_output = gr.Textbox(label="Cycling Path Statistics", lines=5)
                        
                    with gr.Column(scale=2):
                        map_output = gr.HTML(label="Map")
                        plot_output = gr.HTML(label="Path Length Distribution")
            
            # Tab for route suggestions
            with gr.Tab("Route Suggester"):
                gr.Markdown("## Find a Cycling Route")
                gr.Markdown("Let us suggest a cycling route based on your preferences")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Inputs for route suggestion
                        start_area = gr.Dropdown(
                            choices=[
                                "Downtown Dubai", "Dubai Marina", "Jumeirah Beach", 
                                "Al Qudra", "Nad Al Sheba", "Sports City", 
                                "Business Bay", "Dubai Creek", "Jumeirah Lake Towers",
                                "Palm Jumeirah", "International City", "Al Barsha",
                                "Meydan", "Silicon Oasis", "Hatta", "Festival City",
                                "Knowledge Park", "Internet City", "Deira", "Al Karama"
                            ],
                            value="Dubai Marina",
                            label="Starting Area"
                        )
                        
                        target_distance = gr.Slider(
                            minimum=1, 
                            maximum=50,
                            value=10,
                            step=1,
                            label="Target Distance (km)"
                        )
                        
                        tolerance = gr.Slider(
                            minimum=0.5,
                            maximum=5,
                            value=2,
                            step=0.5,
                            label="Distance Tolerance (±km)"
                        )
                        
                        find_route_btn = gr.Button("Find Route")
                        route_status = gr.Textbox(label="Status")
                        
                    with gr.Column(scale=2):
                        route_map = gr.HTML(label="Suggested Route")
                        route_stats = gr.JSON(label="Route Details")
        
        # Define function to load from dropdown selection
        def load_from_dropdown(selected_file, map_type, zoom_level, map_style):
            if not selected_file:
                return None, "Please select a KML file from the dropdown.", None, None
            
            # Find the file path
            try:
                file_path = None
                data_dir = Path("data")
                potential_path = data_dir / selected_file
                
                if potential_path.exists():
                    file_path = potential_path
                else:
                    # Try to find the file in the available files
                    for avail_file in available_files:
                        if avail_file.name == selected_file:
                            file_path = avail_file
                            break
                
                if not file_path:
                    return None, f"Could not find file: {selected_file}", None, None
                
                # Create a mock file object that has a 'name' attribute
                class MockFile:
                    def __init__(self, path):
                        self.name = str(path)
                
                mock_file = MockFile(file_path)
                return process_file(mock_file, map_type, zoom_level, map_style)
            
            except Exception as e:
                return None, f"Error loading file from dropdown: {str(e)}", None, None
        
        # Modified process_file function to handle default setting
        def process_file_with_default(file, map_type, zoom_level, map_style, set_default):
            result = process_file(file, map_type, zoom_level, map_style)
            
            # If requested, set this file as the default
            if file and set_default:
                try:
                    replace_default_kml(file.name)
                    # Update dropdown choices
                    new_files = get_available_kml_files()
                    file_dropdown.choices = [f.name for f in new_files]
                    file_dropdown.value = Path(file.name).name
                    result = (result[0], result[1] + "\nFile set as new default.", result[2], result[3])
                except Exception as e:
                    result = (result[0], result[1] + f"\nError setting as default: {str(e)}", result[2], result[3])
            
            return result
        
        # Auto-load the default file when the app starts
        def autoload_default():
            if default_kml_path and default_kml_path.exists():
                try:
                    class MockFile:
                        def __init__(self, path):
                            self.name = str(path)
                    
                    mock_file = MockFile(default_kml_path)
                    map_html, msg, stats, plot = process_file(mock_file, "basic", 11, "positron")
                    return map_html, f"Auto-loaded default KML file: {default_kml_path.name}", stats, plot
                except Exception as e:
                    return None, f"Error auto-loading default file: {str(e)}", None, None
            else:
                return None, "No default KML file available. Please upload a file.", None, None
        
        # Function to handle route suggestion
        def suggest_route(start_area, target_distance, tolerance):
            try:
                if 'gdf' not in globals() or globals()['gdf'] is None:
                    return None, "Please load a KML file with cycling paths first.", None
                
                print(f"Finding routes starting from {start_area}, target: {target_distance}km ±{tolerance}km")
                
                # Try finding a route with better connectivity
                route_gdf, stats = find_cycling_route(
                    globals()['gdf'], 
                    start_area, 
                    target_distance, 
                    tolerance,
                    use_better_connectivity=True  # Now this parameter exists
                )
                
                # Check if we have a valid route
                if route_gdf is None or stats.get("total_distance_km", 0) < target_distance - tolerance:
                    # Try with a simpler approach if the better one didn't work
                    route_gdf, stats = find_cycling_route(
                        globals()['gdf'],
                        start_area,
                        target_distance,
                        tolerance,
                        use_better_connectivity=False  # Fallback to simpler method
                    )
                
                # Create the route map
                if route_gdf is not None:
                    map_html = create_route_map(
                        globals()['gdf'],
                        route_gdf,
                        start_area,
                        stats
                    )
                    return map_html, f"Found a {stats['route_type']} route of {stats['total_distance_km']}km near {start_area}", stats
                else:
                    return None, "Could not find a suitable route with the current settings.", None
            
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Route suggestion error: {e}")
                print(error_details)
                return None, f"Error finding route: {str(e)}", None
        
        # Set up event handlers
        load_btn.click(
            fn=lambda file_select, file_upload, map_t, zoom, style, set_def: 
                load_from_dropdown(file_select, map_t, zoom, style) if file_select else 
                process_file_with_default(file_upload, map_t, zoom, style, set_def),
            inputs=[file_dropdown, file_input, map_type, zoom_level, map_style, set_as_default], 
            outputs=[map_output, status, stats_output, plot_output]
        )
        
        update_btn.click(
            update_map, 
            inputs=[map_type, zoom_level, map_style], 
            outputs=[map_output, status, stats_output, plot_output]
        )
        
        # Connect the find route button to the suggestion function
        find_route_btn.click(
            suggest_route,
            inputs=[start_area, target_distance, tolerance],
            outputs=[route_map, route_status, route_stats]
        )
        
        # Auto-load the default file when app starts
        app.load(
            fn=autoload_default,
            outputs=[map_output, status, stats_output, plot_output]
        )
    
    return app

# Launch the app if run directly
if __name__ == "__main__":
    app = dubai_cycling_app_with_route_suggester()
    # Modify the launch method to specify a port and show more information
    app.launch(
        server_port=7868,  # Explicitly set the port to 7868
        show_error=True,   # Show detailed errors if something goes wrong
        server_name="0.0.0.0"  # Allow access from other devices on the network
    )
