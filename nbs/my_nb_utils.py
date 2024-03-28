import pandas as pd
import os,requests, warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlparse
import resource, psutil, multiprocessing
from humanize import naturalsize

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
    # Get the current resource usage
    usage = resource.getrusage(resource.RUSAGE_SELF)
    
    # Define a dictionary to map resource names to their human-readable descriptions
    resource_descriptions = {
        'ru_utime': 'User time',
        'ru_stime': 'System time',
        'ru_maxrss': 'Max. Resident Set Size',
        'ru_ixrss': 'Shared Memory Size',
        'ru_idrss': 'Unshared Memory Size',
        'ru_isrss': 'Stack Size',
        'ru_minflt': 'Page faults not requiring I/O',
        'ru_majflt': 'Page faults requiring I/O',
        'ru_nswap': 'Number of swap outs',
        'ru_inblock': 'Block input operations',
        'ru_oublock': 'Block output operations',
        'ru_msgsnd': 'Messages sent',
        'ru_msgrcv': 'Messages received',
        'ru_nsignals': 'Signals received',
        'ru_nvcsw': 'Voluntary context switches',
        'ru_nivcsw': 'Involuntary context switches',
    }
    
    # Print each resource usage with a human-readable description
    for name, desc in resource_descriptions.items():
        value = getattr(usage, name)
        # Convert memory-related values to a human-readable format
        if 'rss' in name or 'mem' in name:
            value = naturalsize(value)
        print(f'{desc} ({name}) = {value}')

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