import open3d as o3d
import numpy as np
import os
import copy 
from PIL import Image # Keep for dummy texture creation if needed

try:
    print(f"Open3D version: {o3d.__version__}")
except Exception as e:
    print(f"Could not get Open3D version: {e}")

def load_obj_and_process_alpha_texture(obj_path, 
                                       output_ply_path=None,
                                       alpha_threshold=50): # Alpha value (0-255) below which pixels are considered transparent
    """
    Loads an .obj file with an RGBA texture. Creates a new point cloud where each point
    corresponds to a vertex of a triangle, colored using its specific UV coordinate.
    Points sampling fully or mostly transparent areas (based on alpha_threshold) are skipped.
    """
    if not os.path.exists(obj_path):
        print(f"Error: OBJ file not found at {obj_path}")
        return None, None

    print(f"Loading OBJ file from: {obj_path}")
    try:
        # enable_post_processing helps with materials, print_progress is useful for large files
        mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True, print_progress=True)
    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        return None, None

    if not mesh.has_vertices() or not mesh.has_triangles():
        print("Error: Loaded mesh has no vertices or no triangles.")
        return None, None

    print(f"Successfully loaded mesh with {len(mesh.vertices)} unique vertices and {len(mesh.triangles)} triangles.")

    if not mesh.has_triangle_uvs() or not mesh.textures:
        print("Error: Mesh does not have triangle UVs or textures. Cannot create colored point cloud.")
        o3d.visualization.draw_geometries([mesh], window_name="Mesh (No Texture/UVs for Face Coloring)")
        return mesh, None 

    print(f"Mesh has triangle_uvs: True, Number of textures: {len(mesh.textures)}")

    tex_dims = None
    num_img_channels = 0
    try:
        tex = mesh.textures[0] # Assuming the first texture is the relevant one
        np_tex_data = np.asarray(tex) # This should be uint8 if from typical PNG
        if np_tex_data.ndim >= 2:
            img_height, img_width = np_tex_data.shape[0], np_tex_data.shape[1]
            num_img_channels = np_tex_data.shape[2] if np_tex_data.ndim == 3 else 1
            tex_dims = (img_width, img_height, num_img_channels)
            print(f"Texture 0 dimensions: W={img_width}, H={img_height}, C={num_img_channels}")
            if num_img_channels < 4:
                print("Warning: Texture does not seem to have an Alpha channel (expected 4 channels for transparency).")
        else:
            print("Error: Texture 0 numpy array has too few dimensions.")
            return mesh, None
    except Exception as e:
        print(f"Error getting texture 0 dimensions or data: {e}")
        return mesh, None
    
    image_width, image_height, _ = tex_dims
    # texture_data_np is already np_tex_data

    print("\nVisualizing the original loaded mesh (textures should appear if loaded correctly)...")
    o3d.visualization.draw_geometries([mesh], window_name="Original Textured OBJ Mesh")

    print("Creating new colored point cloud from triangle faces (skipping transparent areas)...")
    
    valid_points_list = []
    valid_colors_list = []
    
    all_vertices_np = np.asarray(mesh.vertices)
    all_triangles_np = np.asarray(mesh.triangles)
    all_triangle_uvs_np = np.asarray(mesh.triangle_uvs)

    skipped_transparent_points = 0

    for i in range(len(all_triangles_np)):
        tri_v_indices = all_triangles_np[i]
        
        for j in range(3): # For each vertex in this triangle
            vertex_index = tri_v_indices[j]
            uv_index_in_triangle_uvs_array = i * 3 + j
            
            point_coord = all_vertices_np[vertex_index]
            
            u, v = all_triangle_uvs_np[uv_index_in_triangle_uvs_array]
            pix_x = int(u * (image_width - 1))
            pix_y = int((1.0 - v) * (image_height - 1)) # Invert V for image lookup

            if 0 <= pix_x < image_width and 0 <= pix_y < image_height:
                color_val_from_texture = np_tex_data[pix_y, pix_x] # uint8 usually, potentially RGBA
                
                alpha_val = 255 # Default to opaque if no alpha channel
                if num_img_channels == 4:
                    alpha_val = color_val_from_texture[3]
                
                if alpha_val < alpha_threshold:
                    skipped_transparent_points += 1
                    continue # Skip this point if its alpha value is below threshold

                # Extract RGB and normalize
                rgb_from_texture_uint8 = np.array([128,128,128]) # Default gray if single channel
                if num_img_channels >= 3: # RGB or RGBA
                    rgb_from_texture_uint8 = color_val_from_texture[:3]
                elif num_img_channels == 1: # Grayscale
                    rgb_from_texture_uint8 = np.array([color_val_from_texture]*3) 
                
                sampled_color_rgb_float = rgb_from_texture_uint8 / 255.0
                
                valid_points_list.append(point_coord)
                valid_colors_list.append(sampled_color_rgb_float)
            else:
                # UV out of bounds, skip this point as well, or assign a debug color
                # For now, we skip it to avoid adding points with arbitrary colors
                skipped_transparent_points += 1 # Counting out-of-bounds as "skipped"
                continue
            
    print(f"Skipped {skipped_transparent_points} points presumed to be transparent or UVs out of bounds.")
    
    if not valid_points_list:
        print("No valid (non-transparent) points found to create point cloud.")
        return mesh, None

    colored_pcd_from_faces = o3d.geometry.PointCloud()
    colored_pcd_from_faces.points = o3d.utility.Vector3dVector(np.array(valid_points_list))
    colored_pcd_from_faces.colors = o3d.utility.Vector3dVector(np.array(valid_colors_list))
    print(f"Created point cloud with {len(colored_pcd_from_faces.points)} valid points from faces.")

    if colored_pcd_from_faces.has_points():
        print("\nVisualizing the new colored point cloud (transparent areas possibly skipped)...")
        o3d.visualization.draw_geometries([colored_pcd_from_faces], window_name="Colored Point Cloud (Alpha Processed)")

        if output_ply_path:
            try:
                o3d.io.write_point_cloud(output_ply_path, colored_pcd_from_faces, write_ascii=False)
                print(f"Successfully saved new colored point cloud to {output_ply_path}")
            except Exception as e:
                print(f"Error saving new colored PLY: {e}")
    
    return mesh, colored_pcd_from_faces


if __name__ == '__main__':
    try:
        print(f"Open3D version: {o3d.__version__}")
    except Exception as e:
        print(f"Could not get Open3D version: {e}")

    obj_file_path = "C:/Users/admin/Desktop/ColorMapping.obj" 
    # IMPORTANT: Ensure the .mtl file referenced by ColorMapping.obj correctly points to a
    # PNG file (or other format with alpha) if transparency is expected.
    # If Graphene_Bottle.jpg was the texture, and it's a JPG, it has NO alpha.
    # You must use a PNG with actual transparency for this script to work as intended regarding alpha.
    
    output_ply_file_path = "C:/Users/admin/Desktop/ColorMapping_points_alpha_filtered.ply" 

    # --- Parameter for Alpha Threshold ---
    # Alpha values are typically 0 (fully transparent) to 255 (fully opaque).
    # Points sampling pixels with alpha < ALPHA_CUTOFF_VALUE will be skipped.
    ALPHA_CUTOFF_VALUE = 50  # Adjust this: e.g., 10 to skip nearly fully transparent, 
                             # 250 to skip anything not almost fully opaque.

    # --- Fallback to dummy files if the specified OBJ path doesn't exist ---
    if not os.path.exists(obj_file_path):
        print(f"\nWARNING: Specified OBJ file not found at '{obj_file_path}'.")
        # ... (dummy file creation logic from previous script can be here if needed for testing) ...
        # For this example, we'll assume if the main obj_file_path is not found, we exit or raise error.
        print("Please provide a valid path to your .obj file.")
        exit()
    # --- End of dummy file fallback ---

    loaded_mesh, extracted_pcd = load_obj_and_process_alpha_texture(
        obj_file_path, 
        output_ply_file_path,
        alpha_threshold=ALPHA_CUTOFF_VALUE
    )

    if loaded_mesh:
        print("\nFinished processing.")
    else:
        print("\nProcessing failed.")