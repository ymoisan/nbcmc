import pandas as pd
import os,requests, warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlparse
import psutil, multiprocessing
from humanize import naturalsize
import geopandas as gpd
import fiona
from pathlib import Path
import tempfile
import zipfile
from urllib.request import urlretrieve
from ftplib import FTP

from pathlib import Path
import geopandas as gpd
from deltalake import DeltaTable
from deltalake.writer import write_deltalake

def write_gdf_to_delta(gdf: gpd.GeoDataFrame, table_name: str, mode: str = "overwrite") -> None:
    """
    Write a GeoDataFrame to a Delta Lake table in the current working directory.
    
    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to write
        table_name (str): Name of the Delta table (will be created as a directory)
        mode (str): Write mode - either "overwrite" or "append". Defaults to "overwrite"
    
    Returns:
        None
    """
    # Ensure table_name doesn't have file extension
    table_name = Path(table_name).stem
    
    # Convert to pandas DataFrame (Delta Lake writer doesn't directly support GeoDataFrame)
    df = gdf.to_pandas()
    
    # Write to Delta table
    write_deltalake(
        table_name,
        df,
        mode=mode
    )

# Function to extract data from XML and return as a Pandas DataFrame
def extract_xml_data_to_pd(xml_data):
    '''
Extract data from swob-ml data    
    '''
    
    # Register the namespace prefixes
    ns = {'om': 'http://www.opengis.net/om/1.0',
          'gml': 'http://www.opengis.net/gml/3.2',
          'dms': 'http://dms.ec.gc.ca/schema/point-observation/2.0',
          'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}

    tree = ET.parse(xml_data)
    root = tree.getroot()
    
    
    # Find all elements with the name "identification-elements"
    identification_element_section = root.findall('.//dms:identification-elements', ns)
    identification_elements = root.findall('.//dms:identification-elements/dms:element', ns)
    nb_identification_elements = len(identification_elements)
#    print(f'Number of identification_element sections = {len(identification_element_section)}')
#    print(f'Number of identification_elements = {nb_identification_elements}')
    
    # Find all elements with the name "element" NOT in metadata
    elements = root.findall('.//dms:element', ns)
#    print(f'Number of elements = {len(elements)}')
    
    
    # Initialize an empty dataframe with the desired column names
    df = pd.DataFrame(columns=['name', 'value', 'uom'])
    
    bad_data_records = 0 # Total of records skipped because value is MSNG or not castable to a Float64
    
    # Iterate through the elements and extract the name, value, and uom attributes
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        for element in elements[nb_identification_elements:]:
            name = element.get('name')
            value = element.get('value')
            if value == '':
                value = pd.NaN
            value = pd.to_numeric(value, errors='coerce') # Values should be float but there are strings (e.g. MSNG)
            uom = element.get('uom')
    
            # Append the extracted data to the dataframe as a new row IIF we have a numeric value
            # print(value)
            df = pd.concat([df, pd.DataFrame({'name': [name], 'value': [value], 'uom': [uom]})], ignore_index=True)
            if pd.isna(value):
                bad_data_records += 1
    
    # Iterate through the identification elements and extract the name and value attributes
    for identification_element in identification_element_section:
        for element in identification_element:
            name = element.get('name')
            value = element.get('value')
    
            # Add the extracted identification elements as columns to the dataframe
            # Cast the value to numeric types based on the name; THIS IS DUE TO INCONSISTENCIES IN THE INPUT XML DATA !!
            
            if name == "date_tm":
                date_value = pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%fZ") # casts to ns precision by default
                df[name] = date_value
            elif name in ["stn_elev", "lat", "long"]:
                numeric_value = pd.to_numeric(value, errors='coerce')
                df[name] = numeric_value
            else:
                df[name] = value
    #print(f"Skipped records due to missing or invalid MEASUREMENT data = {bad_data_records}")
    return df, bad_data_records

def get_latest_tag_and_date(repo_url):
    '''
    # Example usage:
    repo_url = "https://github.com/posit-dev/great-tables.git"
    repo_name, latest_tag, commit_date = get_latest_tag_and_date(repo_url)
    if repo_name and latest_tag and commit_date:
        print("Repository Name:", repo_name)
        print("Latest Tag:", latest_tag)
        print("Commit Date:", commit_date)
    '''
    try:
        # Parse the GitHub repository URL to extract the owner and repository name
        parsed_url = urlparse(repo_url)
        owner, repo_name = parsed_url.path.lstrip("/").split("/")[:2]
        
        # Remove ".git" from repo_name if present
        repo_name = repo_name.replace(".git", "")
        
        # Construct the GitHub API URL for tags
        tags_url = f"https://api.github.com/repos/{owner}/{repo_name}/tags"
        
        # Send a GET request to fetch tags information
        response = requests.get(tags_url)
        response.raise_for_status()  # Raise an exception for unsuccessful requests
        
        # Parse the JSON response
        tags_info = response.json()
        
        # Get the latest tag and its date
        if tags_info:
            latest_tag = tags_info[0]['name']
            tag_commit_sha = tags_info[0]['commit']['sha']
            commit_info_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{tag_commit_sha}"
            commit_response = requests.get(commit_info_url)
            commit_response.raise_for_status()
            commit_info = commit_response.json()
            commit_date = datetime.strptime(commit_info['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")
            return repo_name, latest_tag, commit_date
        else:
            print("No tags found.")
            return None, None, None
    except Exception as e:
        print("Error:", e)
        return None, None, None

def memused():
    """
    Print current memory usage information using psutil (cross-platform compatible)
    """
    # Get process-specific memory info
    process = psutil.Process()
    mem_info = process.memory_info()
    
    # Get system memory info
    sys_mem = psutil.virtual_memory()
    
    # Print memory usage information
    print(f"Process Memory Usage:")
    print(f"  RSS (Resident Set Size): {naturalsize(mem_info.rss)}")
    print(f"  VMS (Virtual Memory Size): {naturalsize(mem_info.vms)}")
    
    print("\nSystem Memory:")
    print(f"  Total: {naturalsize(sys_mem.total)}")
    print(f"  Available: {naturalsize(sys_mem.available)}")
    print(f"  Used: {naturalsize(sys_mem.used)} ({sys_mem.percent}%)")
    
    # CPU Information
    cpu_percent = process.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    print(f"\nCPU Usage:")
    print(f"  Process CPU Usage: {cpu_percent}%")
    print(f"  Total CPU Cores: {cpu_count}")

# Function to print system usage
# We should have values similar to the amount os resources we asked for
def print_system_usage():
    # Get the current memory usage
    mem_info = psutil.virtual_memory()
    # Convert memory usage to a human-readable format
    mem_usage = naturalsize(mem_info.used)
    mem_left = naturalsize(mem_info.free)
    cpu_count = multiprocessing.cpu_count()
    cpu_usage = len(os.sched_getaffinity(0))
    # Print memory usage
    print(f"Memory usage: {mem_usage}. \nFree memory : {mem_left} \nNumber of CPUs in node: {cpu_count} \nNumber of CPUs available for this process: {cpu_usage}")

def print_file_sizes_human_readable(file_paths):
    for file_path in file_paths:
        file_size = os.stat(file_path).st_size
        print(f"{file_path} : {naturalsize(file_size)}")

def get_gpkg_schema(gpkg_path_or_url, layer='geoai_buildings'):
    """
    Extract schema information from a GeoPackage file using geopandas.
    Handles both local files and URLs to zip files containing GeoPackages.
    
    Parameters:
    -----------
    gpkg_path_or_url : str or Path
        Path to local GeoPackage file or URL to zip file containing a GeoPackage
    layer : str, optional
        Layer name to extract schema from. One of:
        - 'geoai_buildings' (default)
        - 'geoai_forest'
        - 'geoai_hydro'
    
    Returns:
    --------
    dict
        Dictionary containing layer attribute information
    """
    try:
        # Create a temporary directory to store downloaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            actual_gpkg_path = gpkg_path_or_url
            
            # If input is a URL, download and extract
            if str(gpkg_path_or_url).startswith('http'):
                # Download the zip file
                zip_path = os.path.join(temp_dir, 'temp.zip')
                urlretrieve(gpkg_path_or_url, zip_path)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    
                # Find the .gpkg file in the extracted contents
                gpkg_files = [f for f in os.listdir(temp_dir) if f.endswith('.gpkg')]
                if not gpkg_files:
                    raise ValueError("No .gpkg file found in the zip archive")
                actual_gpkg_path = os.path.join(temp_dir, gpkg_files[0])
            
            # Verify the layer exists
            available_layers = fiona.listlayers(actual_gpkg_path)
            if layer not in available_layers:
                raise ValueError(f"Layer '{layer}' not found. Available layers: {available_layers}")
            
            # Read the specified layer
            gdf = gpd.read_file(actual_gpkg_path, layer=layer)
            # Get dtypes, excluding geometry
            schema_info = gdf.drop(columns='geometry').dtypes.to_dict()
                
            return schema_info
    
    except Exception as e:
        print(f"Error processing {gpkg_path_or_url}: {str(e)}")
        return None

def compare_gpkg_schemas(file_paths_or_urls, layer='geoai_buildings'):
    """
    Compare schemas of multiple GeoPackage files
    
    Parameters:
    -----------
    file_paths_or_urls : list
        List of file paths or URLs to zip files containing GeoPackages
        e.g., [
            'https://example.com/BC_CapeBall_WV03_20210702.zip',
            'https://example.com/QC_RockForest_WV03_20220930.zip'
        ]
    layer : str, optional
        Layer name to compare. One of:
        - 'geoai_buildings' (default)
        - 'geoai_forest'
        - 'geoai_hydro'
    
    Returns:
    --------
    dict
        Dictionary containing comparison results
    """
    schemas = {}
    
    # Extract schemas for all files
    for path_or_url in file_paths_or_urls:
        # Create shortened filename for display
        if path_or_url.startswith('http'):
            filename = path_or_url.split('/')[-1].replace('.zip', '')
            short_name = f"{filename[:2]}..{filename[-12:]}"
        else:
            filename = os.path.basename(path_or_url)
            short_name = f"{filename[:2]}..{filename[-12:]}"
            
        schema = get_gpkg_schema(path_or_url, layer=layer)
        if schema:
            schemas[short_name] = schema
    
    # Create comparison dictionary
    comparison = {}
    for file_id, attributes in schemas.items():
        for attr, dtype in attributes.items():
            if attr not in comparison:
                comparison[attr] = {}
            comparison[attr][file_id] = dtype
    
    return comparison

def print_schema_comparison(comparison):
    """
    Print a formatted comparison of schemas
    
    Parameters:
    -----------
    comparison : dict
        Output from compare_gpkg_schemas function
    """
    print("-" * 80)
    # Define reference attributes
    reference_attrs = {'area', 'buil_fid', 'perimeter', 'subproj_id'}
    
    print("Reference attributes are:")
    for attr in sorted(reference_attrs):
        print(f"  - {attr}")
    
    # Find additional attributes
    all_attrs = set(comparison.keys())
    additional_attrs = all_attrs - reference_attrs
    if additional_attrs:
        print("\nAdditional attributes found:")
        for attr in sorted(additional_attrs):
            files_with_attr = [file for file, dtype in comparison[attr].items()]
            print(f"  - '{attr}' found in: {sorted(files_with_attr)}")
    
    print("\nDetailed attribute types by file:")
    # Create a DataFrame for visualization
    df = pd.DataFrame(comparison).T
    print(df.to_string())
    
    # Get all unique attributes across all files
    all_files = set(region for regions in comparison.values() for region in regions.keys())
    
    # Check for attribute presence differences
    print("\nMissing reference attributes:")
    attribute_differences = False
    for attr in reference_attrs:
        if attr in comparison:
            missing_files = all_files - set(comparison[attr].keys())
            if missing_files:
                attribute_differences = True
                print(f"  - '{attr}' is missing in: {sorted(missing_files)}")
    
    if not attribute_differences:
        print("  All files contain the reference attributes")
    
    # Check for data type differences
    print("\nData type check:")
    type_differences = False
    for attr in reference_attrs:
        if attr in comparison:
            unique_types = set(str(dtype) for dtype in comparison[attr].values())
            if len(unique_types) > 1:
                type_differences = True
                print(f"  - '{attr}' has different types: {dict(comparison[attr])}")
    
    if not type_differences:
        print("  All reference attributes have consistent data types across files")

def read_and_concat_gpkgs(file_paths_or_urls, layer='geoai_buildings', target_crs='EPSG:4326'):
    """
    Read and concatenate GeoPackage layers from multiple files into a single GeoDataFrame.
    
    Parameters:
    -----------
    file_paths_or_urls : list
        List of file paths or URLs to zip files containing GeoPackages
    layer : str, optional
        Layer name to read. One of:
        - 'geoai_buildings' (default)
        - 'geoai_forest'
        - 'geoai_hydro'
    target_crs : str, optional
        Target coordinate reference system (default: 'EPSG:4326' - WGS84)
    
    Returns:
    --------
    geopandas.GeoDataFrame
        Concatenated GeoDataFrame with all features
    dict
        Processing statistics and metadata
    """
    gdfs = []
    stats = {
        'total_files': len(file_paths_or_urls),
        'successful_reads': 0,
        'failed_reads': 0,
        'total_features': 0,
        'errors': [],
        'crs_transforms': []
    }
    
    for path_or_url in file_paths_or_urls:
        try:
            # Create a temporary directory for each file
            with tempfile.TemporaryDirectory() as temp_dir:
                actual_gpkg_path = path_or_url
                
                # If input is a URL, download and extract
                if str(path_or_url).startswith('http'):
                    # Download the zip file
                    zip_path = os.path.join(temp_dir, 'temp.zip')
                    urlretrieve(path_or_url, zip_path)
                    
                    # Extract the zip file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find the .gpkg file
                    gpkg_files = [f for f in os.listdir(temp_dir) if f.endswith('.gpkg')]
                    if not gpkg_files:
                        raise ValueError("No .gpkg file found in the zip archive")
                    actual_gpkg_path = os.path.join(temp_dir, gpkg_files[0])
                
                # Read the GeoPackage layer
                gdf = gpd.read_file(actual_gpkg_path, layer=layer)
                
                # Record original CRS
                original_crs = gdf.crs
                
                # Transform CRS if needed
                if gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)
                    stats['crs_transforms'].append({
                        'file': path_or_url,
                        'from': str(original_crs),
                        'to': target_crs
                    })
                
                # Extract filename and province code
                filename = os.path.basename(path_or_url)
                province_code = filename.split('_')[0]  # Gets 'BC', 'NB', etc.
                
                # Add source file information and province code
                gdf['source_file'] = filename
                gdf['province_code'] = province_code
                gdf['processing_timestamp'] = pd.Timestamp.now()
                gdf['original_crs'] = str(original_crs)
                
                gdfs.append(gdf)
                stats['successful_reads'] += 1
                stats['total_features'] += len(gdf)
                
        except Exception as e:
            stats['failed_reads'] += 1
            stats['errors'].append({
                'file': path_or_url,
                'error': str(e)
            })
            print(f"Error processing {path_or_url}: {str(e)}")
            continue
    
    if not gdfs:
        raise ValueError("No GeoPackages were successfully processed")
    
    # Concatenate all GeoDataFrames
    final_gdf = pd.concat(gdfs, ignore_index=True)
    
    return final_gdf, stats

def prepare_gdf_for_delta(gdf):
    """
    Prepare a GeoDataFrame for Delta Lake storage by:
    1. Converting geometries to WKB format
    2. Standardizing data types
    3. Handling missing values
    4. Adding quality indicators and comments
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame to prepare
    
    Returns:
    --------
    pandas.DataFrame
        Prepared DataFrame ready for Delta Lake storage
    dict
        Data quality statistics
    """
    quality_stats = {
        'total_rows': len(gdf),
        'null_counts': {},
        'data_types': {},
        'invalid_geometries': 0,
        'type_conversion_issues': {},
        'duplicate_buil_fids': 0
    }
    
    # Create a copy to avoid modifying the original
    df = gdf.copy()
    
    # Initialize quality comments column
    df['quality_comments'] = ''
    
    # Store the CRS before converting to WKB
    geometry_crs = str(gdf.crs)
    
    # Convert geometry to WKB and store as 'geometry'
    df['geometry'] = gdf.geometry.apply(lambda geom: geom.wkb if geom else None)
    
    # Add geometry CRS information
    df['geometry_crs'] = geometry_crs
    
    # Convert to regular DataFrame (drop GeoDataFrame-specific stuff)
    df = pd.DataFrame(df)
    
    # Record null counts for each column
    quality_stats['null_counts'] = df.isnull().sum().to_dict()
    
    # Record data types
    quality_stats['data_types'] = df.dtypes.astype(str).to_dict()
    
    # Count invalid geometries (None or empty WKB)
    quality_stats['invalid_geometries'] = df['geometry'].isnull().sum()
    
    # Add data quality indicator
    df['has_valid_geometry'] = df['geometry'].notnull()
    
    # Check for NaN values in numeric columns before conversion
    numeric_cols = ['area', 'perimeter']
    for col in numeric_cols:
        if col in df.columns:
            original_nulls = df[col].isnull()
            if original_nulls.any():
                df.loc[original_nulls, 'quality_comments'] += f'NaN in file for {col}; '
    
    # Ensure consistent data types for common fields
    type_mapping = {
        'area': 'float64',
        'perimeter': 'float64',
        'buil_fid': 'int64',
        'subproj_id': 'string'
    }
    
    for col, dtype in type_mapping.items():
        if col in df.columns:
            try:
                # Store original values that are not null but will become null
                before_conversion = df[col].notnull()
                df[col] = df[col].astype(dtype)
                after_conversion = df[col].notnull()
                
                # Identify rows where conversion failed (was not null before, but is null after)
                failed_conversion = before_conversion & ~after_conversion
                if failed_conversion.any():
                    df.loc[failed_conversion, 'quality_comments'] += f'Error processing {col}; '
                    quality_stats['type_conversion_issues'][col] = failed_conversion.sum()
                
            except Exception as e:
                print(f"Warning: Could not convert {col} to {dtype}: {str(e)}")
                df.loc[df[col].notnull(), 'quality_comments'] += f'Column {col} type conversion error; '
    
    # Add CRS information to quality stats
    quality_stats['geometry_crs'] = geometry_crs
    
    # Clean up empty comments
    df['quality_comments'] = df['quality_comments'].str.rstrip('; ')
    
    # Check for duplicate buil_fids
    if 'buil_fid' in df.columns:
        duplicates = df[df['buil_fid'].duplicated(keep=False)]
        if not duplicates.empty:
            quality_stats['duplicate_buil_fids'] = len(duplicates)
            # Group duplicates to get all source files for each duplicate buil_fid
            for fid, group in duplicates.groupby('buil_fid'):
                source_files = group['source_file'].unique()
                comment = f'Duplicate buil_fid {fid} found in: {", ".join(source_files)}'
                df.loc[group.index, 'quality_comments'] += comment + '; '
    
    return df, quality_stats

def list_gpkg_files():
    """
    List all GPKG zip files from the FTP server with detailed timestamps
    
    Returns:
    --------
    list of dict
        Each dict contains:
        - filename: str
        - upload_timestamp: datetime
        - size: int (in bytes)
    """
    files_info = []
    try:
        # Connect to FTP server
        print("Connecting to FTP server...")
        ftp = FTP('ftp.maps.canada.ca')
        ftp.login()  # anonymous login
        
        print("\nNavigating to GPKG directory...")
        ftp.cwd('/pub/nrcan_rncan/vector/geobase_geoai_geoia/GPKG/')
        
        # Get list of zip files
        filenames = ftp.nlst()
        zip_files = [f for f in filenames if f.endswith('.zip')]
        
        # Get detailed info for each file
        for filename in zip_files:
            # Get timestamp using MDTM command
            timestamp = ftp.voidcmd(f"MDTM {filename}")[4:].strip()
            # Parse timestamp (format: YYYYMMDDhhmmss)
            dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            
            # Get file size
            size = ftp.size(filename)
            
            files_info.append({
                'filename': filename,
                'upload_timestamp': dt,
                'size': size
            })
        
        # Sort by timestamp, newest first
        files_info.sort(key=lambda x: x['upload_timestamp'], reverse=True)
        
        print("\nGPKG ZIP files and their timestamps:")
        print("-" * 70)
        for file_info in files_info:
            # Convert size to MB with appropriate decimal places
            size_mb = file_info['size'] / (1024 * 1024)
            if size_mb >= 1:
                size_str = f"{size_mb:.1f} MB"
            else:
                size_str = f"{size_mb:.3f} MB"
            
            print(f"{file_info['filename']:<50} {file_info['upload_timestamp'].strftime('%Y-%m-%d %H:%M:%S'):>20} {size_str:>10}")
            
        print(f"\nTotal ZIP files found: {len(files_info)}")
        
    except Exception as e:
        print(f"Error accessing FTP: {str(e)}")
    finally:
        try:
            ftp.quit()
        except:
            pass
    
    return files_info

def get_all_gpkg_data(layer='geoai_buildings', target_crs='EPSG:4326'):
    """
    Get all GeoPackage data from the FTP server into a single GeoDataFrame,
    including upload timestamps.
    
    Parameters:
    -----------
    layer : str, optional
        Layer name to read. One of:
        - 'geoai_buildings' (default)
        - 'geoai_forest'
        - 'geoai_hydro'
    target_crs : str, optional
        Target coordinate reference system (default: 'EPSG:4326' - WGS84)
    
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with all features, ready for Delta Lake
    dict
        Processing statistics and metadata
    """
    start_time = datetime.now()
    
    # Get list of files with timestamps
    files_info = list_gpkg_files()
    
    if not files_info:
        raise ValueError("No files found on FTP server")
    
    # Create URLs for all files
    base_url = "https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/geobase_geoai_geoia/GPKG/"
    file_urls = [f"{base_url}{file_info['filename']}" for file_info in files_info]
    
    # Read and combine all GeoPackages
    combined_gdf, processing_stats = read_and_concat_gpkgs(file_urls, layer=layer, target_crs=target_crs)
    
    # Create a mapping of filenames to upload timestamps
    timestamp_map = {info['filename']: info['upload_timestamp'] for info in files_info}
    
    # Add upload timestamp to the DataFrame
    combined_gdf['upload_timestamp'] = combined_gdf['source_file'].map(timestamp_map)
    
    # Prepare for Delta Lake
    final_df, quality_stats = prepare_gdf_for_delta(combined_gdf)
    
    # Calculate total processing time
    processing_time = datetime.now() - start_time
    
    # Update processing stats
    processing_stats.update({
        'expected_files': len(files_info),
        'upload_date_range': {
            'earliest': min(info['upload_timestamp'] for info in files_info),
            'latest': max(info['upload_timestamp'] for info in files_info)
        },
        'processing_time': processing_time,
        'processed_at': datetime.now(),
        'total_size_mb': sum(info['size'] for info in files_info) / (1024 * 1024)
    })
    
    # Print summary report
    print("\n" + "=" * 80)
    print(f"Processing Summary Report")
    print("=" * 80)
    print(f"FTP Source: /pub/nrcan_rncan/vector/geobase_geoai_geoia/GPKG/")
    print(f"Layer Processed: {layer}")
    print(f"Processing Date: {processing_stats['processed_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Processing Time: {processing_time}")
    print("-" * 80)
    print("\nFile Statistics:")
    print(f"  Total ZIP files found:     {processing_stats['expected_files']:>6}")
    print(f"  Successfully processed:     {processing_stats['successful_reads']:>6}")
    print(f"  Files with errors:         {processing_stats['failed_reads']:>6}")
    print(f"  Total features extracted:   {processing_stats['total_features']:>6}")
    print(f"  Total data size:           {processing_stats['total_size_mb']:.1f} MB")
    
    if quality_stats['invalid_geometries'] > 0:
        print("\nData Quality Issues:")
        print(f"  Invalid geometries:        {quality_stats['invalid_geometries']:>6}")
    
    if quality_stats['type_conversion_issues']:
        print("\nType Conversion Issues:")
        for col, count in quality_stats['type_conversion_issues'].items():
            print(f"  {col}:  {count:>6}")
    
    print("\nTimestamp Range:")
    print(f"  Earliest file: {processing_stats['upload_date_range']['earliest'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Latest file:   {processing_stats['upload_date_range']['latest'].strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if quality_stats['duplicate_buil_fids'] > 0:
        print("\nDuplicate Building IDs:")
        print(f"  Total duplicates found:    {quality_stats['duplicate_buil_fids']:>6}")
        duplicates = df[df['buil_fid'].duplicated(keep=False)]
        print("\nSample of duplicates:")
        for fid, group in duplicates.groupby('buil_fid').head(3):
            print(f"\nbuil_fid {fid} appears in:")
            for _, row in group.iterrows():
                print(f"  - {row['source_file']} (area: {row['area']}, perimeter: {row['perimeter']})")
    else:
        print("\nNo duplicate buil_fids found in the dataset")
    
    return final_df, processing_stats

def verify_geoparquet_bbox(file_path):
    """
    Verify that a GeoParquet file has bbox information
    
    Parameters:
    -----------
    file_path : str
        Path to the GeoParquet file
    
    Returns:
    --------
    bool
        True if bbox information is present
    """
    import pyarrow.parquet as pq
    
    # Read the parquet file metadata
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    
    # Get the file metadata (stored as a serialized JSON string)
    file_metadata = metadata.metadata
    
    # Check for GeoParquet metadata
    has_bbox = False
    if b'geo' in file_metadata:
        import json
        geo_meta = json.loads(file_metadata[b'geo'].decode('utf-8'))
        has_bbox = 'bbox' in geo_meta.get('columns', {}).get('geometry', {})
        bbox_value = geo_meta.get('columns', {}).get('geometry', {}).get('bbox')
        return has_bbox, bbox_value
    return False, None

def clean_province_code(code):
    """
    Clean province codes by removing numbers and known suffixes
    
    Parameters:
    -----------
    code : str
        Province code to clean
    
    Returns:
    --------
    str
        Cleaned province code
    """
    if pd.isna(code):
        return code
    
    # Extract first two characters for standard province codes
    base_code = str(code)[:2].upper()
    
    # Map for special cases
    if base_code == 'PE' or 'PEI' in str(code).upper():
        return 'PE'
    
    return base_code

def save_processed_data(df, base_path="processed_data", original_size_mb=None, partition_by=None, geometry_encoding="wkb"):
    """
    Save processed DataFrame to parquet format
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    base_path : str, optional
        Base path for saving files (default: "processed_data")
    original_size_mb : float, optional 
        Original file size in MB for comparison
    partition_by : str, optional
        Column to partition by
    geometry_encoding : str, optional
        Encoding for geometry column ('wkb' or 'geoarrow')
        
    Returns:
    --------
    Path
        Full path to the saved file
    """
    from shapely import wkb
    from shapely.geometry.base import BaseGeometry
    import geopandas as gpd
    from pathlib import Path
    import pandas as pd
    from datetime import datetime
    
    # Handle base path - use current directory by default
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename_parts = ["geoai_buildings", timestamp]
    if partition_by:
        filename_parts.append(f"partition_{partition_by}")
    filename_parts.append(geometry_encoding)
    filename = "_".join(filename_parts) + ".parquet"
    
    # If input is already a GeoDataFrame, make a copy
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy()
    else:
        # Convert to GeoDataFrame if it isn't already
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Remove rows with None geometries
    gdf = gdf.dropna(subset=['geometry'])
    
    # Add bounding box column as a struct following GeoParquet 1.1 spec
    import pyarrow as pa
    
    # Create bbox struct array with named fields
    bbox_list = [
        {
            'minx': bounds[0],
            'miny': bounds[1],
            'maxx': bounds[2],
            'maxy': bounds[3]
        }
        for bounds in gdf.geometry.bounds.values
    ]
    bbox_array = pa.array(bbox_list, type=pa.struct([
        ('minx', pa.float64()),
        ('miny', pa.float64()),
        ('maxx', pa.float64()),
        ('maxy', pa.float64())
    ]))
    
    # Add bbox column to the GeoDataFrame
    gdf['bbox'] = bbox_array
    
    # Convert to pandas DataFrame and handle geometry at the last moment
    df_save = pd.DataFrame(gdf)
    
    # Convert geometries to WKB for saving
    def convert_geometry(geom):
        if isinstance(geom, BaseGeometry):
            return geom.wkb
        if isinstance(geom, (bytes, str)):
            return geom
        raise ValueError(f"Unexpected geometry type: {type(geom)}")
        
    df_save['geometry'] = df_save['geometry'].apply(convert_geometry)
    
    # Save to parquet
    if partition_by:
        df_save.to_parquet(
            base_path / filename,
            partition_cols=[partition_by],
            engine='pyarrow',
            index=False
        )
        print(f"Saved partitioned data to: {(base_path / filename).absolute()}")
    else:
        output_path = base_path / filename
        df_save.to_parquet(
            output_path,
            engine='pyarrow',
            index=False
        )
        print(f"Saved data to: {output_path.absolute()}")
        
    # Print size comparison if original size was provided
    if original_size_mb:
        saved_size = sum(f.stat().st_size for f in Path(base_path).rglob('*.parquet')) / (1024 * 1024)
        print(f"Original size: {original_size_mb:.2f} MB")
        print(f"Saved size: {saved_size:.2f} MB")
        print(f"Compression ratio: {original_size_mb/saved_size:.2f}x")
    
    return base_path / filename

def analyze_overlaps(df, merge=False, min_area=1.0):
    """
    Quick analysis of overlapping building footprints
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with WKB geometry column
    merge : bool, optional
        If True, merge overlapping polygons while keeping non-overlapping ones (default: False)
    min_area : float, optional
        Minimum area in square meters for valid buildings (default: 1.0)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - overlap_stats: Basic statistics about overlaps
        - merged_footprints: GeoDataFrame with all geometries (merged where overlapping) if merge=True
        
    Notes:
    ------
    - For merged features:
      - subproj_id is kept only if all merged features share the same value, else None
      - buil_fid is set to -merged_count (e.g., -3 for a merge of 3 features)
    """
    import geopandas as gpd
    import pandas as pd
    from shapely import wkb
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from functools import reduce
    
    # If input is already a GeoDataFrame, make a copy
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy()
    else:
        # Convert WKB to GeoDataFrame
        gdf = gpd.GeoDataFrame(df.copy(), geometry=df['geometry'].apply(wkb.loads), crs="EPSG:4326")
    
    # Convert to appropriate UTM zone for accurate area calculations
    utm_crs = get_utm_crs(gdf.total_bounds[0], gdf.total_bounds[1])
    gdf_utm = gdf.to_crs(utm_crs)
    
    # Filter out tiny polygons using UTM area
    gdf_utm = gdf_utm[gdf_utm.geometry.area >= min_area]
    
    # Convert back to WGS84 for consistent output
    gdf = gdf_utm.to_crs("EPSG:4326")
    
    # Ensure buil_fid and subproj_id are integers and drop rows with NA
    if 'buil_fid' in gdf.columns and 'subproj_id' in gdf.columns:
        gdf = gdf.dropna(subset=['buil_fid', 'subproj_id'])
        gdf['buil_fid'] = gdf['buil_fid'].astype('Int64')
        gdf['subproj_id'] = gdf['subproj_id'].astype('Int64')
    
    # Use sjoin to find overlaps more efficiently
    potential_overlaps = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
    
    # Remove self-intersections
    overlaps = potential_overlaps[potential_overlaps.index != potential_overlaps.index_right]
    
    # Basic statistics
    overlap_stats = {
        'total_buildings': len(gdf),
        'buildings_with_overlaps': len(set(overlaps.index) | set(overlaps.index_right)),
        'overlap_pairs': len(overlaps)
    }
    
    results = {'overlap_stats': overlap_stats}
    
    if merge:
        # Get indices of overlapping features
        overlapping_idx = set(overlaps.index) | set(overlaps.index_right)
        
        # Split into overlapping and non-overlapping
        overlapping = gdf.loc[list(overlapping_idx)]
        non_overlapping = gdf.loc[~gdf.index.isin(overlapping_idx)]
        
        # Convert overlapping features to UTM for merging and area calculation
        overlapping_utm = overlapping.to_crs(utm_crs)
        
        # Process overlapping geometries in groups
        merged_features = []
        skip_idx = set()
        
        for idx in overlapping_idx:
            if idx in skip_idx:
                continue
                
            # Get all features that overlap with this one
            related = overlaps[overlaps.index == idx].index_right.tolist()
            related.append(idx)
            
            # Get geometries and data for this group
            group_data = overlapping_utm.loc[related]
            group_geoms = group_data.geometry.tolist()
            
            # Merge using union of all geometries (in UTM)
            merged = reduce(lambda x, y: x.union(y), group_geoms)
            
            # Only keep if result is a simple Polygon and has sufficient area
            if isinstance(merged, Polygon) and merged.area >= min_area:
                # Check if all features in group have same subproj_id
                unique_subproj_ids = group_data['subproj_id'].unique()
                subproj_id = pd.NA if len(unique_subproj_ids) > 1 else unique_subproj_ids[0]
                
                # For merged features, use negative count as buil_fid
                merged_count = len(related)
                merged_buil_fid = -merged_count
                
                merged_features.append({
                    'geometry': merged,  # UTM geometry
                    'area': merged.area,
                    'perimeter': merged.length,
                    'subproj_id': subproj_id,
                    'buil_fid': merged_buil_fid
                })
                skip_idx.update(related)
            else:
                # If merge results in MultiPolygon or tiny area, keep original features with original metrics
                for _, row in group_data.iterrows():
                    merged_features.append({
                        'geometry': row.geometry,
                        'area': overlapping.loc[row.name, 'area'],  # Keep original area
                        'perimeter': overlapping.loc[row.name, 'perimeter'],  # Keep original perimeter
                        'subproj_id': row.subproj_id,
                        'buil_fid': row.buil_fid
                    })
        
        # Create GeoDataFrame with merged overlapping geometries
        if merged_features:
            # Convert merged geometries back to WGS84
            merged_overlapping = gpd.GeoDataFrame(merged_features, crs=utm_crs).to_crs("EPSG:4326")
        else:
            merged_overlapping = gpd.GeoDataFrame(
                columns=['geometry', 'area', 'perimeter', 'subproj_id', 'buil_fid'],
                geometry='geometry',
                crs="EPSG:4326"
            )
        
        # Keep non-overlapping features as they are
        if len(non_overlapping) > 0:
            final_gdf = gpd.GeoDataFrame(pd.concat([
                merged_overlapping,
                non_overlapping
            ], ignore_index=True))
        else:
            final_gdf = merged_overlapping
        
        # Ensure integer types only if we have data
        if len(final_gdf) > 0:
            final_gdf['subproj_id'] = final_gdf['subproj_id'].astype('Int64')
            final_gdf['buil_fid'] = final_gdf['buil_fid'].astype('Int64')
        
        results['merged_footprints'] = final_gdf
        results['merge_stats'] = {
            'original_count': len(gdf),
            'final_count': len(final_gdf),
            'overlapping_merged': len(overlapping),
            'non_overlapping': len(non_overlapping),
            'reduction': len(gdf) - len(final_gdf)
        }
    
    return results

def to_polars(df):
    """
    Convert pandas/geopandas DataFrame to polars DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        DataFrame to convert
    
    Returns:
    --------
    polars.DataFrame
        Converted DataFrame
    """
    import polars as pl
    
    # Convert to pandas first if it's a GeoDataFrame
    if hasattr(df, 'geometry'):
        df = pd.DataFrame(df)
    
    # Convert to polars
    return pl.from_pandas(df)

def get_building_stats(gdf):
    """
    Calculate statistics for building footprints
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with building footprints
    
    Returns:
    --------
    dict
        Dictionary containing:
        - count: Total number of buildings
        - total_area: Total area of all buildings
        - mean_area: Mean building area
        - median_area: Median building area
        - area_std: Standard deviation of building areas
        - total_perimeter: Total perimeter of all buildings
        - mean_perimeter: Mean building perimeter
        - median_perimeter: Median building perimeter
        - perimeter_std: Standard deviation of building perimeters
    """
    import numpy as np
    
    stats = {
        'count': len(gdf),
        'total_area': gdf.area.sum(),
        'mean_area': gdf.area.mean(),
        'median_area': gdf.area.median(),
        'area_std': gdf.area.std(),
        'total_perimeter': gdf.geometry.length.sum(),
        'mean_perimeter': gdf.geometry.length.mean(),
        'median_perimeter': gdf.geometry.length.median(),
        'perimeter_std': gdf.geometry.length.std()
    }
    
    return stats

def load_processed_data(path, as_geodataframe=True):
    """
    Load processed data from parquet format
    
    Parameters:
    -----------
    path : str or Path
        Path to parquet file or directory
    as_geodataframe : bool, optional
        If True, return as GeoDataFrame, else as pandas DataFrame (default: True)
    
    Returns:
    --------
    geopandas.GeoDataFrame or pandas.DataFrame
        Loaded data
    """
    import pandas as pd
    import geopandas as gpd
    from pathlib import Path
    from shapely import wkb
    
    path = Path(path)
    
    # Handle directory vs file
    if path.is_dir():
        df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(str(path))
    
    if as_geodataframe and 'geometry' in df.columns:
        # Convert WKB to geometry objects
        df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if x else None)
        return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    return df