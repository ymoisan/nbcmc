from pathlib import Path
import struct
import subprocess
import shutil
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt
import rasterio
from affine import Affine
from tqdm.notebook import tqdm
import threading, time, sys
from rasterio.windows import Window, from_bounds as window_from_bounds
from functools import partial
import json
import pygltflib  # For parsing GLB/glTF files
import numpy as np
import fiona
from shapely.geometry import shape
from shapely.ops import triangulate

# Material presets constant for USD shading
MATERIAL_PRESETS = {
    'metallic': {
        'diffuseColor': (0.2, 0.5, 0.8),
        'metallic': 0.8,
        'roughness': 0.3
    },
    'matte': {
        'diffuseColor': (0.8, 0.8, 0.8),
        'metallic': 0.0,
        'roughness': 0.8
    },
    'plastic': {
        'diffuseColor': (0.2, 0.8, 0.4),
        'metallic': 0.1,
        'roughness': 0.4,
        'clearcoat': 0.5
    },
    'terrain': {
        'diffuseColor': (0.4, 0.35, 0.25),
        'metallic': 0.0,
        'roughness': 1.0,
        'occlusion': 1.0
    },
    'building': {
        'diffuseColor': (0.8, 0.8, 0.8),
        'metallic': 0.0,
        'roughness': 0.6
    }
}

def check_converter_availability():
    usdgltf_path = shutil.which('usdgltf')
    usdzip_path = shutil.which('usdzip')
    
    print(f"usdgltf available: {usdgltf_path is not None}")
    print(f"usdzip available: {usdzip_path is not None}")
    
    return usdgltf_path or usdzip_path

def extract_gltf_from_b3dm(b3dm_path: Path, gltf_output_path: Path = None):
    if gltf_output_path is None:
        gltf_output_path = b3dm_path.with_suffix('.glb')
    
    with b3dm_path.open('rb') as f:
        # Get file size
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        f.seek(0)  # Back to start
        print(f"File size: {file_size} bytes")

        # Read the header (28 bytes)
        header = f.read(28)
        print(f"Read {len(header)} bytes for header")
        print(f"Header bytes (hex): {header.hex()}")
        
        if len(header) < 28:
            raise ValueError(f"Invalid b3dm file: {b3dm_path.name} (file too small, got {len(header)} bytes)")
        
        try:
            magic, version, byte_length, ftJSONLength, ftBinaryLength, btJSONLength, btBinaryLength = struct.unpack('<4sI5I', header)
            print(f"Magic: {magic}")
            print(f"Version: {version}")
            print(f"Byte length: {byte_length}")
            print(f"Feature Table JSON Length: {ftJSONLength}")
            print(f"Feature Table Binary Length: {ftBinaryLength}")
            print(f"Batch Table JSON Length: {btJSONLength}")
            print(f"Batch Table Binary Length: {btBinaryLength}")
            
            if byte_length > file_size:
                raise ValueError(f"Invalid b3dm file: declared byte length ({byte_length}) larger than file size ({file_size})")
        except struct.error as e:
            raise ValueError(f"Invalid b3dm file: {b3dm_path.name} (could not unpack header): {e}")
        
        if magic != b'b3dm':
            raise ValueError(f"Invalid b3dm file: {b3dm_path.name} (wrong magic number: {magic})")
        
        # Calculate the start of the glTF data
        gltf_start = 28 + ftJSONLength + ftBinaryLength + btJSONLength + btBinaryLength
        print(f"Calculated glTF start: {gltf_start}")
        
        if gltf_start >= file_size:
            raise ValueError(f"Invalid b3dm file: {b3dm_path.name} (glTF start {gltf_start} beyond file size {file_size})")
        
        f.seek(gltf_start)
        
        # The remaining bytes are the glTF data
        gltf_data = f.read()
        print(f"Read {len(gltf_data)} bytes for glTF data")
        
        with gltf_output_path.open('wb') as gltf_file:
            gltf_file.write(gltf_data)
    
    return gltf_output_path

def convert_glb_to_usd(glb_path: Path, usd_path: Path = None) -> Path:
    """Convert GLB to USD using gltf2usd"""
    if usd_path is None:
        usd_path = glb_path.with_suffix('.usda')
    
    try:
        from gltf2usd import convert
        convert.gltf2usd(
            str(glb_path),
            str(usd_path)
        )
        return usd_path
    except ImportError:
        raise ImportError("Please install gltf2usd: pip install gltf2usd")

def create_stage(output_path):
    """
    Creates and initializes a new USD stage.
    
    Args:
        output_path (str or Path): Path to the output USD file
        
    Returns:
        stage: The initialized USD stage
    """
    # Convert input path to string if necessary and ensure .usdc extension
    if isinstance(output_path, Path):
        output_path = str(output_path)
    if not output_path.endswith('.usdc'):
        if output_path.endswith('.usda'):
            print("\nNote: Output will be saved as .usdc")
        output_path = str(Path(output_path).with_suffix('.usdc'))
    
    # Create a new binary USD stage
    stage = Usd.Stage.CreateNew(output_path)
    stage.SetMetadata("upAxis", "Y")
    stage.SetMetadata("metersPerUnit", 1.0)
    
    # Define a world-level Xform and designate it as the default prim
    worldXform = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(worldXform.GetPrim())
    
    return stage

def create_dem_layer(dem_data, transform, scale_factor=1.0, material_type=None):
    """
    Creates a DEM mesh from the provided data.
    
    Args:
        dem_data: The DEM data array
        transform: The geotransform
        scale_factor: Factor to scale the coordinates
        material_type: Type of material to apply
        
    Returns:
        dict: Contains the mesh data and material settings
    """
    # Generate vertices and topology
    points = []
    height, width = dem_data.shape
    overall_total = height + (height - 1)
    
    overall = tqdm(total=overall_total, desc="Overall DEM processing", unit="rows")
    for i in tqdm(range(height), desc="Generating vertices", unit="row"):
        for j in range(width):
            x, y = transform * (j, i)
            x = x * scale_factor
            y = y * scale_factor
            z = float(dem_data[i, j])
            if z < -1000 or z > 10000:
                z = 0
            points.append(Gf.Vec3f(x, y, z))
        overall.update(1)

    # Build mesh topology: each cell (pixel) is split into two triangles.
    faceVertexIndices = []
    faceVertexCounts = []
    cell_rows = height - 1
    for i in tqdm(range(cell_rows), desc="Generating mesh topology", unit="row"):
        for j in range(width - 1):
            idx0 = i * width + j
            idx1 = (i + 1) * width + j
            idx2 = (i + 1) * width + (j + 1)
            idx3 = i * width + (j + 1)
            # Triangle 1.
            faceVertexIndices.extend([idx0, idx1, idx2])
            faceVertexCounts.append(3)
            # Triangle 2.
            faceVertexIndices.extend([idx0, idx2, idx3])
            faceVertexCounts.append(3)
        overall.update(1)
    overall.close()

    return {
        'points': points,
        'faceVertexCounts': faceVertexCounts,
        'faceVertexIndices': faceVertexIndices,
        'material_type': material_type
    }

def add_layer_to_stage(stage, layer_data, parent_path="/World", stats_callback=None):
    """
    Adds a layer to the USD stage at the specified path.
    
    Args:
        stage: The USD stage to add to
        layer_data: Dict containing the layer data (points, faces, indices, material_type)
        parent_path: Path in the USD hierarchy where to add the layer
        stats_callback: Optional function to compute and print mesh statistics
    """
    meshPrim = UsdGeom.Mesh.Define(stage, parent_path)
    
    # Set mesh data
    meshPrim.CreateSubdivisionSchemeAttr().Set("none")
    points_array = Vt.Vec3fArray(layer_data['points'])
    meshPrim.CreatePointsAttr().Set(points_array)
    
    # Then set the topology
    faceVertexCounts_array = Vt.IntArray(layer_data['faceVertexCounts'])
    faceVertexIndices_array = Vt.IntArray(layer_data['faceVertexIndices'])
    meshPrim.CreateFaceVertexCountsAttr().Set(faceVertexCounts_array)
    meshPrim.CreateFaceVertexIndicesAttr().Set(faceVertexIndices_array)
    
    # Set up the material (and shader) for this layer using a unique path based on parent_path.
    material_type = layer_data.get('material_type')
    if material_type in MATERIAL_PRESETS:
        preset = MATERIAL_PRESETS[material_type]
        # Create a unique material path by combining the parent_path with a suffix.
        material_path = parent_path.rstrip("/") + "/Material"
        mat = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/PreviewShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        # Apply preset properties from MATERIAL_PRESETS
        for prop, value in preset.items():
            if isinstance(value, (tuple, list)):
                shader.CreateInput(prop, Sdf.ValueTypeNames.Color3f).Set(value)
            else:
                shader.CreateInput(prop, Sdf.ValueTypeNames.Float).Set(value)
        # Create and connect the surface output
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader_output)
        # Bind the material to the mesh primitive
        UsdShade.MaterialBindingAPI.Apply(meshPrim.GetPrim())
        UsdShade.MaterialBindingAPI(meshPrim.GetPrim()).Bind(mat)
        # Remove displayColor if present
        if meshPrim.GetPrim().HasAttribute("displayColor"):
            meshPrim.RemoveProperty("displayColor")

    # Add debug information
    print("\nMesh Statistics:")
    print(f"Number of points: {len(points_array)}")
    print(f"Number of faces: {len(faceVertexCounts_array)}")
    min_z = min(p[2] for p in points_array)
    max_z = max(p[2] for p in points_array)
    print(f"Z range: {min_z} to {max_z}")
    
    if stats_callback:
        stats_callback(points_array, layer_data)

def save_stage(stage, show_spinner=True):
    """
    Saves the USD stage to disk.
    
    Args:
        stage: The USD stage to save
        show_spinner: Whether to show a spinner during save
    """
    if show_spinner:
        spinner_done = threading.Event()
        spinner_chars = "|/-\\"
        idx = 0
        spinner_thread = threading.Thread(target=lambda: print(f"\rWriting USD file... {spinner_chars[idx % len(spinner_chars)]}", end='', flush=True))
        spinner_thread.start()
        
        try:
            stage.GetRootLayer().Save()
        finally:
            spinner_done.set()
            spinner_thread.join()
    else:
        stage.GetRootLayer().Save()

def create_layer_from_geotiff(geo_tiff_path, sample_factor=1, scale_factor=1.0, material_type=None, bbox=None):
    """
    Creates a mesh layer from a GeoTIFF file, typically used for Digital Elevation Models (DEM).
    
    Args:
        geo_tiff_path: Path to the input GeoTIFF file
        sample_factor: Factor to downsample the input data (1 = no downsampling)
        scale_factor: Factor to scale the coordinates
        material_type: Type of material to apply to the mesh
        bbox: Optional bounding box to clip the data (west, south, east, north)
    
    Returns:
        dict: Layer data containing points, faces, indices and material settings
    """
    try:
        with rasterio.open(geo_tiff_path) as src:
            file_bounds = src.bounds
            
            if bbox is not None:
                west, south, east, north = bbox
                if (west < file_bounds.left or south < file_bounds.bottom or
                    east > file_bounds.right or north > file_bounds.top):
                    print("Error: Requested bbox is outside the DEM bounds.")
                    print(f"DEM bounds: {file_bounds}")
                    print(f"Requested bbox: {bbox}")
                    return
                    
                window = window_from_bounds(west, south, east, north, src.transform)
                dem_data = src.read(1, window=window)
                transform = src.window_transform(window)
            else:
                dem_data = src.read(1)
                transform = src.transform

            # Apply sample_factor before starting progress bars
            if sample_factor > 1:
                dem_data = dem_data[::sample_factor, ::sample_factor]
                transform = transform * Affine.scale(sample_factor)

            return create_dem_layer(dem_data, transform, scale_factor, material_type)
            
    except MemoryError as me:
        import traceback
        traceback.print_exc()
        msg = ("MemoryError occurred during DEM generation. This likely means your system "
               "ran out of available RAM. Please try again with a higher sample_factor.")
        print(msg)
        return None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = (f"An error occurred: {e}. Aborting computation and deleting any partial outputs. "
               "Please fix the issue and try again.")
        print(msg)
        return None

# In script.py

def complete_material_network(stage):
    """
    Completes the material network by adding a basic preview shader for a complete surface output.
    This is useful so that usdview can resolve a preview material.
    """
    from pxr import UsdShade, Sdf

    # Try to retrieve the material that was bound to /World/Material.
    material = UsdShade.Material.Get(stage, "/World/Material")
    if not material:
        print("Warning: No material found at /World/Material to complete the network.")
        return

    # Define a basic preview shader (UsdPreviewSurface) under the material.
    previewShader = UsdShade.Shader.Define(stage, "/World/Material/PreviewShader")
    previewShader.CreateIdAttr("UsdPreviewSurface")
    # For example, set a diffuse color and roughness. Adjust these parameters as needed.
    previewShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.8, 0.8))
    previewShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    # Create an output that the material can use.
    previewSurfaceOutput = previewShader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    
    # Connect the material's surface output to the preview shader's output.
    material.CreateSurfaceOutput().ConnectToSource(previewSurfaceOutput)
    print("Completed the material network by adding a preview surface shader.")

def finalize_usd_stage(usd_output_path):
    """
    Loads a USD stage from the given output path, completes the material network,
    and then saves the stage. Returns the stage.
    """
    from pxr import Usd
    # If usd_output_path is not a string, convert it to a string.
    if not isinstance(usd_output_path, str):
        usd_output_path = str(usd_output_path)
    stage = Usd.Stage.Open(usd_output_path)
    if not stage:
        print("Error: Could not open stage at", usd_output_path)
        return None
    complete_material_network(stage)
    stage.GetRootLayer().Save()
    print(f"Finalized USD stage saved at {usd_output_path}")
    return stage

def add_dem_to_stage(stage, dem_layer):
    """
    Specialized function for adding a DEM to the stage.
    
    Args:
        stage: The USD stage to add to
        dem_layer: Dict containing the DEM mesh data and material settings
    """
    def dem_stats(points_array, mesh_data):
        print("\nDEM Statistics:")
        print(f"Number of points: {len(points_array)}")
        print(f"Number of faces: {len(mesh_data['faceVertexCounts'])}")
        min_z = min(p[2] for p in points_array)
        max_z = max(p[2] for p in points_array)
        print(f"Z range: {min_z} to {max_z}")
    
    add_layer_to_stage(stage, dem_layer, "/World/DEM", dem_stats)

def read_tileset(tileset_path: Path) -> dict:
    """Read and parse the tileset.json file"""
    with open(tileset_path) as f:
        return json.load(f)

def create_layer_from_3dtiles(tileset_path: Path, material_type=None, scale_factor=1.0, z_offset=0.0) -> dict:
    """
    Creates a mesh layer by reading and combining geometries from a 3D Tiles dataset.
    Preserves original materials from glTF data if present, otherwise uses a default material.
    
    Args:
        tileset_path: Path to the tileset.json file of the 3D Tiles dataset
        material_type: Type of material to apply to the mesh
        scale_factor: Factor to scale the coordinates
        z_offset: Vertical offset for buildings
        
    Returns:
        dict: Layer data containing combined points, faces, indices and material settings
             from all b3dm files in the tileset
    """
    # Default material for meshes without materials
    DEFAULT_MATERIAL = {
        'pbr_metallic_roughness': {
            'baseColorFactor': [0.7, 0.7, 0.7, 1.0],
            'metallicFactor': 0.0,
            'roughnessFactor': 0.6
        },
        'double_sided': True,
        'alpha_mode': 'OPAQUE'
    }
    
    # Read tileset
    tileset = read_tileset(tileset_path)
    
    # Initialize collections for all meshes
    all_points = []
    all_face_counts = []
    all_face_indices = []
    all_materials = []  # Store material data from glTF
    point_offset = 0
    meshes_with_materials = 0
    total_meshes = 0
    success_count = 0
    
    # Get base path for b3dm files
    base_path = tileset_path.parent
    
    # Count total files to process
    def count_nodes(node):
        count = 0
        if 'content' in node:
            count += 1
        if 'children' in node:
            for child in node['children']:
                count += count_nodes(child)
        return count
    
    total_files = count_nodes(tileset['root'])
    print(f"Found {total_files} b3dm files to process")
    from tqdm import tqdm
    pbar = tqdm(total=total_files, desc="Processing b3dm files", unit="file")
    
    # Process the node and all of its children recursively.
    def process_node(node):
        nonlocal point_offset, meshes_with_materials, total_meshes, success_count
        if 'content' in node:
            total_meshes += 1
            file_processed = False
            try:
                b3dm_path = base_path / node['content']['uri']
                print(f"\nProcessing {b3dm_path.name}...")

                if not b3dm_path.exists():
                    raise FileNotFoundError(f"B3DM file not found at: {b3dm_path}")

                temp_gltf = extract_gltf_from_b3dm(b3dm_path)
                gltf = pygltflib.GLTF2().load(str(temp_gltf))

                # Get the first mesh
                mesh_primitive = gltf.meshes[0].primitives[0]

                # Check for POSITION attribute
                if not (hasattr(mesh_primitive.attributes, "POSITION") and mesh_primitive.attributes.POSITION is not None):
                    print(f"Mesh in {b3dm_path.name} does not have POSITION attribute, skipping this file.")
                else:
                    pos_index = mesh_primitive.attributes.POSITION
                    pos_accessor = gltf.accessors[pos_index]
                    if pos_accessor.bufferView is None:
                        print(f"Mesh in {b3dm_path.name} has POSITION accessor missing bufferView, skipping this file.")
                    else:
                        pos_data = get_data_from_accessor(gltf, pos_accessor)
                        points = [Gf.Vec3f(p[0], p[1], p[2] * scale_factor + z_offset) for p in pos_data]

                    idx_accessor = gltf.accessors[mesh_primitive.indices]
                    if idx_accessor.bufferView is None:
                        print(f"Mesh in {b3dm_path.name} has indices accessor missing bufferView, skipping this file.")
                    else:
                        indices = get_data_from_accessor(gltf, idx_accessor)
                        if not isinstance(indices, list):
                            print(f"Indices for mesh in {b3dm_path.name} are not a list, skipping this file.")
                        else:
                            if mesh_primitive.material is not None:
                                meshes_with_materials += 1
                                material = gltf.materials[mesh_primitive.material]
                                all_materials.append({
                                    'index_offset': len(all_face_indices),
                                    'index_count': len(indices),
                                    'pbr_metallic_roughness': material.pbrMetallicRoughness,
                                    'double_sided': material.doubleSided,
                                    'alpha_mode': material.alphaMode
                                })
                            else:
                                print(f"No material found in glTF for mesh in {b3dm_path.name}, applying default material")
                                all_materials.append({
                                    'index_offset': len(all_face_indices),
                                    'index_count': len(indices),
                                    **DEFAULT_MATERIAL
                                })

                            # Only add geometry if both points and indices were valid.
                            if 'points' in locals():
                                all_points.extend(points)
                                all_face_counts.extend([3] * (len(indices) // 3))
                                all_face_indices.extend([i + point_offset for i in indices])
                                point_offset += len(points)
                                file_processed = True

                temp_gltf.unlink()
            except Exception as e:
                print(f"Error processing {b3dm_path.name}: {str(e)}")
                print("Skipping this file...")
            finally:
                pbar.update(1)
                if file_processed:
                    success_count += 1
        # Recursively process child nodes if they exist.
        if 'children' in node:
            for child in node['children']:
                process_node(child)
    
    # Start processing from root
    process_node(tileset['root'])
    
    # Print material usage summary
    if meshes_with_materials < total_meshes:
        print(f"\nMaterial Summary:")
        print(f"- {meshes_with_materials}/{total_meshes} meshes had materials in source files")
        print(f"- {total_meshes - meshes_with_materials} meshes using default material")
        print(f"Default material properties: {DEFAULT_MATERIAL}")
    
    pbar.close()
    print(f"Successfully processed {success_count} out of {total_files} b3dm files.")
    
    # After constructing all_points, all_face_counts, and all_face_indices
    print(f"Total points: {len(all_points)}")
    print(f"Total face counts: {len(all_face_counts)}")
    print(f"Total face indices: {len(all_face_indices)}")
    
    # Optionally, print some sample data
    print("Sample points:", all_points[:5])  # Print first 5 points
    print("Sample face counts:", all_face_counts[:5])  # Print first 5 face counts
    print("Sample face indices:", all_face_indices[:10])  # Print first 10 face indices

    result = {
        'points': all_points,
        'faceVertexCounts': all_face_counts,
        'faceVertexIndices': all_face_indices,
        'materials': all_materials
    }
    # Set material_type so that add_layer_to_stage can assign the fill color;
    # use the provided material_type if given, else default to "building"
    result['material_type'] = material_type if material_type is not None else "building"
    return result

def get_data_from_accessor(gltf, accessor):
    """
    Extracts data from a glTF accessor in a binary glTF (GLB) file.
    Handles both regular and sparse accessors.
    """
    # Debugging: Print accessor details
    print(f"Accessor details: {accessor}")
    
    # Determine number of components based on accessor type
    type_to_num = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16
    }
    if accessor.type not in type_to_num:
        raise ValueError(f"Unsupported accessor type: {accessor.type}")
    num_components = type_to_num[accessor.type]
    accessor_count = accessor.count
    
    # Map glTF component type to numpy dtype
    dtype_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    component_type = accessor.componentType
    if component_type not in dtype_map:
        raise ValueError(f"Unsupported component type: {component_type}")
    dtype = dtype_map[component_type]
    
    # Handle sparse accessor
    if accessor.sparse is not None:
        # Create base array (zeros)
        base_data = np.zeros((accessor_count, num_components), dtype=dtype)
        
        # Get indices from sparse
        indices_accessor = accessor.sparse.indices
        indices_view = gltf.bufferViews[indices_accessor.bufferView]
        indices_offset = (indices_view.byteOffset or 0) + (indices_accessor.byteOffset or 0)
        indices_bytes = gltf.binary_blob[indices_offset:indices_offset + indices_view.byteLength]
        indices = np.frombuffer(indices_bytes, dtype=dtype_map[indices_accessor.componentType])
        
        # Get values from sparse
        values_accessor = accessor.sparse.values
        values_view = gltf.bufferViews[values_accessor.bufferView]
        values_offset = (values_view.byteOffset or 0) + (values_accessor.byteOffset or 0)
        values_bytes = gltf.binary_blob[values_offset:values_offset + values_view.byteLength]
        values = np.frombuffer(values_bytes, dtype=dtype).reshape(-1, num_components)
        
        # Put sparse values in their correct positions
        base_data[indices] = values
        return base_data.tolist()
    
    # Handle regular accessor
    if accessor.bufferView is None:
        # If no bufferView and no sparse data, return zeros
        if accessor.type == "SCALAR":
            return [0] * accessor_count
        else:
            return [[0] * num_components for _ in range(accessor_count)]
    
    # Debugging: Print bufferView index
    print(f"BufferView index: {accessor.bufferView}")
    
    bufferView = gltf.bufferViews[accessor.bufferView]
    bufferView_offset = bufferView.byteOffset or 0
    accessor_offset = accessor.byteOffset or 0
    total_offset = bufferView_offset + accessor_offset
    
    # Compute the total number of values and required bytes
    total_count = accessor_count * num_components
    itemsize = np.dtype(dtype).itemsize
    nbytes = total_count * itemsize
    
    # Extract the binary data from gltf.binary_blob
    data_bytes = gltf.binary_blob[total_offset:total_offset+nbytes]
    arr = np.frombuffer(data_bytes, dtype=dtype)
    if accessor.type != "SCALAR":
        arr = arr.reshape((accessor_count, num_components))
    return arr.tolist()

def calculate_normal(v0, v1, v2):
    # Calculate the normal of the triangle defined by vertices v0, v1, v2
    edge1 = np.array(v1) - np.array(v0)
    edge2 = np.array(v2) - np.array(v0)
    normal = np.cross(edge1, edge2)
    normal_length = np.linalg.norm(normal)
    if normal_length == 0:
        return np.array([0, 0, 0])
    return normal / normal_length

def create_layer_with_normals(geopkg_path, material_type='building', scale_factor=1.0, z_offset=0.0):
    all_points = []
    all_face_counts = []
    all_face_indices = []
    all_normals = []
    point_offset = 0

    # Open the GeoPackage and iterate over its features.
    with fiona.open(geopkg_path) as src:
        for feature in src:
            geom = shape(feature['geometry'])
            if geom.geom_type == 'MultiPolygon':
                polygons = list(geom.geoms)
            elif geom.geom_type == 'Polygon':
                polygons = [geom]
            else:
                continue

            roof_min_z = feature['properties'].get('roof_min_Z', 0)
            ground_z = feature['properties'].get('ground_Z', 0)
            ground_buffer = 0.1
            base_z = ground_z + ground_buffer

            for poly in polygons:
                exterior_coords = list(poly.exterior.coords)[:-1]

                bottom_vertices = []
                top_vertices = []
                
                for coord in exterior_coords:
                    x, y = coord[0], coord[1]
                    bottom_vertices.append(Gf.Vec3f(x * scale_factor, y * scale_factor, base_z + z_offset))
                    top_vertices.append(Gf.Vec3f(x * scale_factor, y * scale_factor, roof_min_z + z_offset))
                
                vertices = bottom_vertices + top_vertices
                num_vertices = len(bottom_vertices)

                all_points.extend(vertices)
                
                bottom_indices = list(range(num_vertices))
                all_face_counts.append(num_vertices)
                all_face_indices.extend([i + point_offset for i in bottom_indices])
                
                top_indices = list(range(num_vertices, 2 * num_vertices))
                all_face_counts.append(num_vertices)
                all_face_indices.extend([i + point_offset for i in top_indices])
                
                for i in range(num_vertices):
                    next_i = (i + 1) % num_vertices
                    wall_face = [
                        i,
                        next_i,
                        next_i + num_vertices,
                        i + num_vertices
                    ]
                    all_face_counts.append(4)
                    all_face_indices.extend([idx + point_offset for idx in wall_face])

                    # Calculate normals for the wall face
                    v0 = all_points[wall_face[0] + point_offset]
                    v1 = all_points[wall_face[1] + point_offset]
                    v2 = all_points[wall_face[2] + point_offset]
                    normal = calculate_normal(v0, v1, v2)
                    all_normals.extend([normal] * 4)  # Same normal for all vertices of the face

                point_offset += len(vertices)

    return {
         'points': all_points,
         'faceVertexCounts': all_face_counts,
         'faceVertexIndices': all_face_indices,
         'normals': all_normals,
         'material_type': material_type
    }

if __name__ == "__main__":
    # Example usage of layer creation and stage management
    stage = create_stage("terrain_with_buildings.usdc")
    
    # Create and add DEM layer
    dem_layer = create_layer_from_geotiff("dem.tif", material_type='terrain')
    add_layer_to_stage(stage, dem_layer, "/World/DEM")
    
    # Create and add buildings layer from GeoPackage
    buildings_gpkg_layer = create_layer_with_normals("buildings.gpkg", material_type='building', scale_factor=100000, z_offset=50.0)
    add_layer_to_stage(stage, buildings_gpkg_layer, "/World/Buildings")
    
    # Save stage
    save_stage(stage)