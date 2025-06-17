# script_v15_save_relative_pose.py
import open3d as o3d
import numpy as np
import os
import copy
import sys

print(f"Open3D version: {o3d.__version__}")

# --- Helper function to create transformation matrix ---
def create_transformation_matrix(R, t):
    """
    Creates a 4x4 homogeneous transformation matrix from a 3x3 rotation matrix and a 3x1 translation vector.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten() # Ensure t is a flat array for assignment
    return T

# --- Function to load GraspGroup data from gg.txt and get best gripper pose ---
def get_best_gripper_pose_from_gg_file(gg_file_path):
    """
    Loads grasp data from a gg.txt file, sorts by score (first column, descending),
    and returns the 4x4 transformation matrix of the best grasp.
    Format of gg.txt rows (comma-separated, first line is header):
    score, width, height, depth, R(0,0), R(0,1), ..., R(2,2), T(0), T(1), T(2), [optional_other_data...]
    """
    if not os.path.exists(gg_file_path):
        print(f"Error: Grasp data file '{gg_file_path}' not found.")
        return None
    try:
        # Load data, skipping the header row
        grasp_data_array = np.loadtxt(gg_file_path, delimiter=',', skiprows=1)
        
        # Handle case where gg.txt might have only one data row (loadtxt returns 1D array)
        if grasp_data_array.ndim == 1:
            grasp_data_array = grasp_data_array.reshape(1, -1)
            
        if grasp_data_array.shape[0] == 0:
            print(f"No grasp data found in {gg_file_path} after loading (or only header was present).")
            return None
            
        print(f"Loaded {grasp_data_array.shape[0]} grasps from {gg_file_path}")

        # Sort by score (first column, index 0) in descending order
        # argsort returns indices that would sort the array. [::-1] reverses it for descending.
        sorted_indices = np.argsort(grasp_data_array[:, 0])[::-1]
        grasp_data_array_sorted_by_score = grasp_data_array[sorted_indices]
        
        # Get the parameters for the best grasp (first row after sorting)
        best_grasp_params = grasp_data_array_sorted_by_score[0]
        
        # Ensure the row has enough data for rotation (9 elements) and translation (3 elements)
        # Score (idx 0) + 3 params (width, height, depth) + 9 R_elements + 3 T_elements = 16 elements minimum
        if best_grasp_params.shape[0] < 16:
            print("Error: Best grasp data row in gg.txt does not have enough columns for R and T.")
            print(f"Expected at least 16 columns, got {best_grasp_params.shape[0]}. Row data: {best_grasp_params}")
            return None
            
        # Extract rotation matrix (elements 4 to 12) and translation vector (elements 13 to 15)
        # Indices are: score[0], width[1], height[2], depth[3], Rxx[4]...Rzz[12], Tx[13], Ty[14], Tz[15]
        rotation_matrix = best_grasp_params[4:13].reshape((3, 3))
        translation_vector = best_grasp_params[13:16]
        
        T_cam_graspnet_gripper = create_transformation_matrix(rotation_matrix, translation_vector)
        
        print(f"Best GraspNet Gripper Pose from file (T_cam_graspnet_gripper), Score: {best_grasp_params[0]:.6f}")
        return T_cam_graspnet_gripper
        
    except ValueError as ve:
        print(f"Error parsing numeric data in gg.txt: {ve}. Ensure all data rows are numeric after the header.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading or parsing gg.txt: {e}")
        return None

# --- Function to define Object Pose using OBB ---
def get_graspnet_object_pose_in_cam_frame(object_pcd_path_graspnet):
    """
    Loads an object point cloud and determines its pose (T_cam_graspnet_object)
    in the camera frame using its Oriented Bounding Box (OBB).
    Returns the 4x4 transformation matrix and the Open3D point cloud object.
    """
    if not os.path.exists(object_pcd_path_graspnet):
        print(f"Error: GraspNet Object PLY file '{object_pcd_path_graspnet}' not found.")
        return None, None
    try: 
        object_pcd_o3d = o3d.io.read_point_cloud(object_pcd_path_graspnet)
        if not object_pcd_o3d.has_points():
            print(f"Error: Point cloud loaded from '{object_pcd_path_graspnet}' is empty.")
            return None, None
    except Exception as e:
        print(f"Error loading GraspNet object PLY '{object_pcd_path_graspnet}' with Open3D: {e}")
        return None, None
    
    R_cam_graspnet_object = np.identity(3) # Default rotation
    try:
        # Compute OBB. For some simple/symmetric objects, OBB might be axis-aligned
        # depending on the initial orientation of the point cloud.
        obb = object_pcd_o3d.get_oriented_bounding_box()
        object_center_graspnet = obb.center
        R_cam_graspnet_object = obb.R 
        print(f"Using OBB for GraspNet object pose. Center: {object_center_graspnet}, Rotation from OBB used.")
    except RuntimeError as e:
        print(f"Failed to compute OBB for GraspNet object, using centroid and identity rotation: {e}")
        object_center_graspnet = object_pcd_o3d.get_center()
        # R_cam_graspnet_object remains identity
        
    T_cam_graspnet_object = create_transformation_matrix(R_cam_graspnet_object, object_center_graspnet)
    print("GraspNet Object Pose in Camera/World Frame (T_cam_graspnet_object):")
    # print(T_cam_graspnet_object) 
    return T_cam_graspnet_object, object_pcd_o3d


if __name__ == '__main__':
    base_path = r"C:\Users\admin\Desktop\mug" 
    grasp_data_filename = "mug.txt" 
    object_ply_filename = "mug.ply" 
    
    # Output file for the relative transformation matrix
    relative_pose_output_filename = "relative_gripper_to_object_pose.txt" # Original name
    # Changed to match the filename used in Opt_9.py for consistency
    # relative_pose_output_filename = "relative_gripper_to_object_pose_CONDITIONAL_REORIENT.txt"


    gg_file_path = os.path.join(base_path, grasp_data_filename)
    object_ply_path_graspnet = os.path.join(base_path, object_ply_filename)
    relative_pose_output_path = os.path.join(base_path, relative_pose_output_filename)

    # Call the function and expect a single 4x4 matrix (or None)
    T_cam_graspnet_gripper = get_best_gripper_pose_from_gg_file(gg_file_path)
    
    if not os.path.exists(object_ply_path_graspnet):
        print(f"CRITICAL: Object-only PLY file '{object_ply_path_graspnet}' (from GraspNet) not found. Exiting.")
        sys.exit() # Use sys.exit() for a cleaner exit
        
    T_cam_graspnet_object, graspnet_object_pcd_for_vis = get_graspnet_object_pose_in_cam_frame(object_ply_path_graspnet)

    if T_cam_graspnet_gripper is not None and T_cam_graspnet_object is not None:
        # Calculate Object pose IN GraspNet Gripper frame: T_G_O = inv(T_C_G) * T_C_O
        # This represents how the object is positioned and oriented relative to the gripper's coordinate system.
        T_graspnet_gripper_TO_graspnet_object = np.linalg.inv(T_cam_graspnet_gripper) @ T_cam_graspnet_object
        
        print("\n--- Relative Transformation Matrix ---")
        print("T_graspnet_gripper_TO_graspnet_object (GraspNet Object's pose relative to GraspNet Gripper's coordinate system):")
        print(T_graspnet_gripper_TO_graspnet_object)

        # --- Save the relative transformation matrix to a .txt file ---
        try:
            np.savetxt(relative_pose_output_path, T_graspnet_gripper_TO_graspnet_object, fmt='%.8f')
            print(f"\nSuccessfully saved relative transformation matrix to: {relative_pose_output_path}")
        except Exception as e:
            print(f"\nError saving relative transformation matrix: {e}")

        # --- Visualization (Optional, kept from V15 for verification) ---
        if graspnet_object_pcd_for_vis is not None and graspnet_object_pcd_for_vis.has_points():
            # Visualize in Camera Frame
            geometries_in_camera_view = []
            
            # Ensure point cloud has colors for visualization
            if not graspnet_object_pcd_for_vis.has_colors(): 
                graspnet_object_pcd_for_vis.paint_uniform_color([0.7,0.7,0.7]) # Default gray
            geometries_in_camera_view.append(graspnet_object_pcd_for_vis)

            # Gripper visualization
            gripper_frame_vis_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0])
            gripper_frame_vis_cam.transform(T_cam_graspnet_gripper)
            gripper_origin_marker_cam = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            gripper_origin_marker_cam.paint_uniform_color([0.9, 0.1, 0.1]) # Red for gripper
            gripper_origin_marker_cam.transform(T_cam_graspnet_gripper)
            geometries_in_camera_view.extend([gripper_frame_vis_cam, gripper_origin_marker_cam])

            # Object visualization
            object_frame_vis_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0,0,0])
            object_frame_vis_cam.transform(T_cam_graspnet_object)
            object_origin_marker_cam = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            object_origin_marker_cam.paint_uniform_color([0.1, 0.9, 0.1]) # Green for object
            object_origin_marker_cam.transform(T_cam_graspnet_object)
            geometries_in_camera_view.extend([object_frame_vis_cam, object_origin_marker_cam])
            
            print("\nVisualizing in Camera Frame (GraspNet's perspective):")
            o3d.visualization.draw_geometries(geometries_in_camera_view, 
                                              window_name="GRASPNET CAM VIEW: Object (Green Origin), Gripper (Red Origin)")

            # Visualize Object in GraspNet Gripper Frame
            # To do this, we transform the object's point cloud (which is in camera frame)
            # by the inverse of the gripper's pose in camera frame.
            # P_gripper_frame = inv(T_cam_gripper) * P_cam_frame
            object_pcd_in_gn_gripper_frame = copy.deepcopy(graspnet_object_pcd_for_vis)
            object_pcd_in_gn_gripper_frame.transform(np.linalg.inv(T_cam_graspnet_gripper))

            gn_gripper_is_at_origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0])
            print("\nVisualizing Object IN GRASPNET GRIPPER FRAME (GraspNet Gripper is at origin):")
            o3d.visualization.draw_geometries([object_pcd_in_gn_gripper_frame, gn_gripper_is_at_origin_frame],
                                              window_name="GRASPNET GRIPPER VIEW: Object relative to Gripper (at origin)")
        else:
            print("GraspNet object point cloud for visualization is empty or None.")
    else:
        print("Could not determine GraspNet gripper or GraspNet object pose. Cannot calculate or save relative transformation.")

