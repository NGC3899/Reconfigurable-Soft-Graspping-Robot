import open3d as o3d
import numpy as np
import copy
import os

def load_point_clouds(target_combined_path, source_path):
    print(f"Loading target combined cloud from: {target_combined_path}")
    try:
        target_combined_cloud = o3d.io.read_point_cloud(target_combined_path)
        if not target_combined_cloud.has_points():
            print(f"Error: Target combined cloud is empty after loading: {target_combined_path}")
            return None, None
    except Exception as e:
        print(f"Error loading target combined cloud {target_combined_path}: {e}")
        return None, None

    print(f"Loading source high-res cloud from: {source_path}")
    try:
        source_high_res_cloud = o3d.io.read_point_cloud(source_path)
        if not source_high_res_cloud.has_points():
            print(f"Error: Source high-res cloud is empty after loading: {source_path}")
            return target_combined_cloud, None
    except Exception as e:
        print(f"Error loading source high-res cloud {source_path}: {e}")
        return target_combined_cloud, None
        
    print("Point clouds loaded successfully.")
    return target_combined_cloud, source_high_res_cloud

def separate_target_and_gripper_by_color(combined_cloud, gripper_color_val=np.array([0.5, 0.5, 0.5]), color_tolerance=0.02):
    target_object_cloud = o3d.geometry.PointCloud()
    gripper_cloud_vis = o3d.geometry.PointCloud()

    if not target_object_cloud.has_colors():
        target_object_cloud.points = o3d.utility.Vector3dVector(np.empty((0,3)))
        target_object_cloud.colors = o3d.utility.Vector3dVector(np.empty((0,3)))
    if not gripper_cloud_vis.has_colors():
        gripper_cloud_vis.points = o3d.utility.Vector3dVector(np.empty((0,3)))
        gripper_cloud_vis.colors = o3d.utility.Vector3dVector(np.empty((0,3)))

    if not combined_cloud.has_points():
        print("Warning: Combined cloud is empty, cannot separate.")
        return target_object_cloud, gripper_cloud_vis 

    if not combined_cloud.has_colors():
        print("Warning: Combined cloud has no colors for separation. Using whole cloud as target.")
        target_object_cloud = copy.deepcopy(combined_cloud)
        return target_object_cloud, gripper_cloud_vis

    points = np.asarray(combined_cloud.points)
    colors = np.asarray(combined_cloud.colors)
    
    lower_bound = np.clip(gripper_color_val - color_tolerance, 0.0, 1.0)
    upper_bound = np.clip(gripper_color_val + color_tolerance, 0.0, 1.0)
    
    gripper_mask_r = (colors[:, 0] >= lower_bound[0]) & (colors[:, 0] <= upper_bound[0])
    gripper_mask_g = (colors[:, 1] >= lower_bound[1]) & (colors[:, 1] <= upper_bound[1])
    gripper_mask_b = (colors[:, 2] >= lower_bound[2]) & (colors[:, 2] <= upper_bound[2])
    gripper_mask = gripper_mask_r & gripper_mask_g & gripper_mask_b

    num_gripper_points = np.sum(gripper_mask)
    num_object_points = len(points) - num_gripper_points

    print(f"Color separation: Found {num_object_points} obj points, {num_gripper_points} gripper points.")
    
    if num_object_points > 0:
        target_object_cloud.points = o3d.utility.Vector3dVector(points[~gripper_mask])
        target_object_cloud.colors = o3d.utility.Vector3dVector(colors[~gripper_mask])
    else:
        print("Warning: No object points found after color separation. Target object will be empty.")
        
    if num_gripper_points > 0:
        gripper_cloud_vis.points = o3d.utility.Vector3dVector(points[gripper_mask])
        gripper_cloud_vis.colors = o3d.utility.Vector3dVector(colors[gripper_mask])
    else:
        print("Warning: No gripper points found by color separation.")
             
    return target_object_cloud, gripper_cloud_vis


def estimate_normals_if_missing(pcd, radius_multiplier=2.0, default_search_radius=0.01, max_nn=30):
    pcd_copy = copy.deepcopy(pcd) 
    if not pcd_copy.has_normals():
        if not pcd_copy.has_points() or len(pcd_copy.points) < 3: 
            print("Warning: Cannot estimate normals for point cloud with < 3 points.")
            return pcd_copy
        actual_radius = default_search_radius * radius_multiplier
        print(f"Estimating normals (radius={actual_radius:.4f}, max_nn={max_nn})...")
        pcd_copy.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=actual_radius, max_nn=max_nn))
        if not pcd_copy.has_normals(): 
            print("Warning: Normal estimation did not result in normals.")
    return pcd_copy

def preprocess_point_cloud_for_features(pcd_original, voxel_size):
    if not pcd_original.has_points():
        print("Warning: Original point cloud is empty before preprocessing for features.")
        empty_pcd = o3d.geometry.PointCloud()
        if pcd_original.has_colors():
             empty_pcd.colors = o3d.utility.Vector3dVector(np.empty((0,3)))
        return empty_pcd, None 
    
    pcd_input = copy.deepcopy(pcd_original) 
        
    print(f"Preprocessing for features: Original point count = {len(pcd_input.points)}, Voxel size = {voxel_size:.4f}")
    pcd_down = pcd_input.voxel_down_sample(voxel_size)
    print(f"Downsampled point count = {len(pcd_down.points)}")


    if not pcd_down.has_points() or len(pcd_down.points) < 30: 
        print(f"Warning: Point cloud has too few points ({len(pcd_down.points)}) after voxel down sample (voxel_size {voxel_size}) for robust FPFH computation.")
        return pcd_down, None 

    radius_normal = voxel_size * 2
    print(f"Estimating normals for FPFH with radius {radius_normal:.4f}")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    if not pcd_down.has_normals():
        print("Warning: Failed to estimate normals for FPFH. Cannot compute FPFH.")
        return pcd_down, None

    radius_feature = voxel_size * 5
    print(f"Computing FPFH features with radius {radius_feature:.4f}")
    try:
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        if pcd_fpfh is None or pcd_fpfh.num() == 0 or pcd_fpfh.dimension() == 0:
            print("Warning: FPFH computation resulted in empty or invalid features.")
            return pcd_down, None
    except Exception as e:
        print(f"Error during FPFH computation: {e}")
        return pcd_down, None
        
    return pcd_down, pcd_fpfh


def global_registration_fpfh_ransac(source_down, target_down, source_fpfh, target_fpfh, 
                                    voxel_size, max_iterations=5000, confidence=0.95): 
    if source_fpfh is None or target_fpfh is None or \
       not source_down.has_points() or not target_down.has_points():
        print("Error: Empty point clouds or FPFH features for global registration.")
        return np.identity(4), 0.0
    if source_fpfh.num() != len(source_down.points) or \
       target_fpfh.num() != len(target_down.points):
        print("Error: Feature count mismatch with point count for global registration.")
        print(f"  Source FPFH num: {source_fpfh.num()}, Source points: {len(source_down.points)}")
        print(f"  Target FPFH num: {target_fpfh.num()}, Target points: {len(target_down.points)}")
        return np.identity(4), 0.0


    distance_threshold = voxel_size * 1.5 
    print(f"Global RANSAC: Distance threshold = {distance_threshold:.4f}")
    
    ransac_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=max_iterations, confidence=confidence) 
    print(f"RANSAC Criteria: Max iterations = {ransac_criteria.max_iteration}, Confidence = {ransac_criteria.confidence}")
    print("Starting RANSAC calculation (this may take some time)...")

    result_transform = np.identity(4) 
    fitness = 0.0
    try:
        reg_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, 
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
            ransac_n=3, 
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=ransac_criteria) 
        result_transform = reg_result.transformation
        fitness = reg_result.fitness
        print("RANSAC calculation finished.")
        print("Global RANSAC registration result:", reg_result) 
    except RuntimeError as e:
        print(f"RuntimeError during RANSAC registration: {e}. RANSAC failed.")
    except Exception as e:
        print(f"Unexpected error during RANSAC registration: {e}. RANSAC failed.")
        
    return result_transform, fitness


def refine_registration_icp(source_pcd, target_pcd, initial_transformation, 
                            max_correspondence_distance, 
                            icp_type="point_to_plane", 
                            max_iterations=50,
                            relative_fitness=1e-7, relative_rmse=1e-7):
    print(f"ICP Refinement ({icp_type}): Max correspondence dist = {max_correspondence_distance:.4f}, Max iter = {max_iterations}")

    source_pcd_copy = copy.deepcopy(source_pcd)
    target_pcd_copy = copy.deepcopy(target_pcd)

    if not source_pcd_copy.has_points() or not target_pcd_copy.has_points():
        print("Error: Source or target for ICP is empty.")
        return initial_transformation, 0.0

    if icp_type == "point_to_plane":
        normal_est_radius = max_correspondence_distance * 1.0 
        if normal_est_radius < 0.001: normal_est_radius = 0.001 
        
        source_pcd_copy = estimate_normals_if_missing(source_pcd_copy, default_search_radius=normal_est_radius, radius_multiplier=1.0) 
        target_pcd_copy = estimate_normals_if_missing(target_pcd_copy, default_search_radius=normal_est_radius, radius_multiplier=1.0)
        if not source_pcd_copy.has_normals() or not target_pcd_copy.has_normals():
            print("Warning: Normals for PointToPlane ICP missing/failed. Falling back to PointToPoint.")
            icp_type = "point_to_point"
    
    if icp_type == "point_to_plane":
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else: 
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    
    fitness = 0.0
    transformation = initial_transformation
    try:
        print("Starting ICP calculation...")
        result = o3d.pipelines.registration.registration_icp(
            source_pcd_copy, target_pcd_copy, 
            max_correspondence_distance, 
            initial_transformation,
            estimation_method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=relative_fitness, 
                relative_rmse=relative_rmse, 
                max_iteration=max_iterations)
        )
        print("ICP calculation finished.")
        print(f"ICP ({icp_type}) refinement result:", result)
        transformation = result.transformation
        fitness = result.fitness
    except RuntimeError as e:
        print(f"RuntimeError during ICP refinement: {e}. Returning initial transformation.")
    except Exception as e:
        print(f"Unexpected error during ICP refinement: {e}. Returning initial transformation.")

    return transformation, fitness


def visualize_scene_with_original_colors(fixed_geometries_list, movable_geometry, 
                                           transformation=np.identity(4), 
                                           window_name="Scene",
                                           point_show_normal=False):
    display_list = []
    default_fixed_color = [0.6, 0.6, 0.9] 
    default_other_fixed_color = [0.5, 0.8, 0.5] 
    default_movable_color = [0.9, 0.6, 0.2]

    for i, geom in enumerate(fixed_geometries_list):
        if not geom or not geom.has_points(): 
            print(f"Warning: Fixed geometry at index {i} is None or empty in '{window_name}'. Skipping.")
            continue
        g_copy = copy.deepcopy(geom)
        if not g_copy.has_colors(): 
            if i == 0: 
                g_copy.paint_uniform_color(default_fixed_color) 
            else: 
                g_copy.paint_uniform_color(default_other_fixed_color)
        display_list.append(g_copy)

    if movable_geometry and movable_geometry.has_points():
        movable_copy = copy.deepcopy(movable_geometry)
        movable_copy.transform(transformation)
        if not movable_copy.has_colors(): 
            movable_copy.paint_uniform_color(default_movable_color) 
        display_list.append(movable_copy) 
    else:
        print(f"Warning: Movable geometry is None or empty in '{window_name}'. Skipping its visualization.")
    
    if not display_list:
        print(f"Warning: No geometries to display for window '{window_name}'.")
        return

    print(f"Visualizing: {len(display_list)} geometries for window '{window_name}'")
    o3d.visualization.draw_geometries(display_list, window_name=window_name, point_show_normal=point_show_normal)


if __name__ == "__main__":
    base_path = r"C:\Users\admin\Desktop" 
    target_combined_filename = "GB_Coarse.ply" 
    source_filename = "Graphene_Bottle.ply"  
    target_combined_path = os.path.join(base_path, target_combined_filename)
    source_high_res_path = os.path.join(base_path, source_filename)
    output_dir = os.path.join(base_path, "registration_output_v8_manual_rot") 
    os.makedirs(output_dir, exist_ok=True)

    SOURCE_SCALE_FACTOR = 1
    
    # --- USER-DEFINED MANUAL ROTATION ANGLES (degrees) ---
    # 修改这些值来在初始阶段手动旋转CAD模型
    manual_rotate_x_deg = 0.0  # 绕X轴旋转的角度 (度)
    manual_rotate_y_deg = 0.0  # 绕Y轴旋转的角度 (度)
    manual_rotate_z_deg = 0.0  # 绕Z轴旋转的角度 (度)
    # --- END USER-DEFINED MANUAL ROTATION ---
    
    # 增加此值以减少RANSAC的点数（如果单次迭代仍然太慢）
    voxel_size_global_registration = 0.5 # 例如 5mm
    print(f"INFO: Using voxel_size_global_registration = {voxel_size_global_registration}")
    
    # RANSAC 迭代次数和置信度 - 可调整以平衡速度和质量
    default_ransac_iterations = 10
    default_ransac_confidence = 0.95  

    final_icp_max_distance = voxel_size_global_registration * 1.5 
    final_icp_iterations = 100
    final_icp_type = "point_to_plane" 

    target_combined_cloud, source_high_res_cloud_orig = load_point_clouds(target_combined_path, source_high_res_path)
    if target_combined_cloud is None or source_high_res_cloud_orig is None: 
        print("Failed to load point clouds. Exiting.")
        exit()

    print(f"\n--- Scaling Source Point Cloud (CAD Model) by factor: {SOURCE_SCALE_FACTOR} ---")
    source_cad_model_scaled = copy.deepcopy(source_high_res_cloud_orig)
    if not source_cad_model_scaled.has_points():
        print("Error: Original source CAD model is empty before scaling. Exiting.")
        exit()
        
    if SOURCE_SCALE_FACTOR != 1.0: 
        source_cad_model_scaled.scale(SOURCE_SCALE_FACTOR, center=source_cad_model_scaled.get_center())
    print(f"Scaled CAD model point count: {len(source_cad_model_scaled.points)}")

    # --- APPLY USER-DEFINED MANUAL ROTATION ---
    if manual_rotate_x_deg != 0.0 or manual_rotate_y_deg != 0.0 or manual_rotate_z_deg != 0.0:
        print(f"\n--- Applying User-Defined Manual Rotation ---")
        print(f"  Rotating by X: {manual_rotate_x_deg} deg, Y: {manual_rotate_y_deg} deg, Z: {manual_rotate_z_deg} deg")
        
        center_for_manual_rotation = source_cad_model_scaled.get_center() 
        
        if manual_rotate_x_deg != 0.0:
            R_x_mat = source_cad_model_scaled.get_rotation_matrix_from_xyz((np.deg2rad(manual_rotate_x_deg), 0, 0))
            source_cad_model_scaled.rotate(R_x_mat, center=center_for_manual_rotation)
        
        if manual_rotate_y_deg != 0.0:
            R_y_mat = source_cad_model_scaled.get_rotation_matrix_from_xyz((0, np.deg2rad(manual_rotate_y_deg), 0))
            source_cad_model_scaled.rotate(R_y_mat, center=center_for_manual_rotation)

        if manual_rotate_z_deg != 0.0:
            R_z_mat = source_cad_model_scaled.get_rotation_matrix_from_xyz((0, 0, np.deg2rad(manual_rotate_z_deg)))
            source_cad_model_scaled.rotate(R_z_mat, center=center_for_manual_rotation)
            
        print("Manual rotation applied.")
    # --- END MANUAL ROTATION ---
    
    if source_cad_model_scaled.has_points():
        visualize_scene_with_original_colors(
            [target_combined_cloud], 
            source_cad_model_scaled, 
            window_name="[VIS 1] Initial State (After Manual Rotation if any)" # Window name updated
        )
    else:
        print("Skipping VIS 1: Scaled CAD model is empty (possibly after manual rotation if it was initially empty).")


    print("\n--- Separating target object from gripper in the Realsense scan ---")
    target_obj_scan, gripper_visualization_cloud = separate_target_and_gripper_by_color(target_combined_cloud)
    print(f"Target object scan point count: {len(target_obj_scan.points)}")
    print(f"Gripper visualization cloud point count: {len(gripper_visualization_cloud.points)}")

    is_target_combined_due_to_separation_failure = False
    if target_combined_cloud.has_points() and target_obj_scan.has_points() and \
       not gripper_visualization_cloud.has_points() and \
       len(target_obj_scan.points) == len(target_combined_cloud.points):
        is_target_combined_due_to_separation_failure = True
        print("INFO: Color separation was likely ineffective; target_obj_scan is the whole combined cloud.")

    if not target_obj_scan.has_points():
        print("Error: Target object scan is empty after color separation. Exiting.")
        exit()
    if not source_cad_model_scaled.has_points(): # Check again after manual rotation
        print("Error: Scaled CAD model is empty (possibly after manual rotation). Exiting.")
        exit()

    print("\n--- Initial Centroid Alignment (CAD Model to Target Scan) ---")
    # Important: Centroid alignment should use the current state of source_cad_model_scaled (after manual rotation)
    target_scan_centroid = target_obj_scan.get_center()
    source_cad_current_centroid = source_cad_model_scaled.get_center() # Use current centroid
    T_centroid_align = np.identity(4)
    T_centroid_align[:3, 3] = target_scan_centroid - source_cad_current_centroid # Align current CAD to target
    
    # Apply centroid alignment to a copy for this step's transform
    current_best_T_cad_to_scan = copy.deepcopy(T_centroid_align)
    
    # For visualization, transform a fresh copy of the (manually rotated) source_cad_model_scaled
    # This current_best_T_cad_to_scan is the transform to apply to the *manually rotated* source to align its centroid
    # to the target_obj_scan's centroid.

    print("\n--- Visualizing SCALED CAD Model after Centroid Alignment to Target Scan ---")
    visualize_scene_with_original_colors(
        [target_obj_scan, gripper_visualization_cloud] if gripper_visualization_cloud.has_points() else [target_obj_scan], 
        source_cad_model_scaled, # Pass the manually rotated CAD model
        current_best_T_cad_to_scan, # Apply the centroid alignment transform
        window_name="[VIS 2] Centroid Alignment"
    )
    
    # The current_best_T_cad_to_scan at this point is just T_centroid_align relative to the
    # manually rotated source_cad_model_scaled's frame.
    # For RANSAC, we need a transformation from the *original* (only scaled) CAD model frame
    # to the target scan frame.
    # Or, more simply, RANSAC operates on the current state of source_cad_model_scaled and target_obj_scan.
    # The T_ransac_absolute it returns will be a transform from source_cad_model_scaled's *current* frame
    # (i.e., after manual rotation) to target_obj_scan's frame.

    # Let T_manual_rot be the implicit transformation from manual rotation.
    # T_centroid_align is calculated based on source_cad_model_scaled (which includes T_manual_rot).
    # So current_best_T_cad_to_scan = T_centroid_align_on_manually_rotated_source
    
    # For RANSAC, we pass source_cad_model_scaled (which is already manually rotated)
    # and target_obj_scan. RANSAC will then find a T_ransac that aligns these two.
    # This T_ransac will be a transform for the *manually rotated* source.
    
    print("\n--- Performing Global Registration (FPFH + RANSAC) ---")
    print("Preparing manually rotated CAD model for FPFH...")
    # Use the current state of source_cad_model_scaled (which includes manual rotation)
    source_cad_down, source_cad_fpfh = preprocess_point_cloud_for_features(
        source_cad_model_scaled, 
        voxel_size_global_registration
    )
    print("Preparing target object scan for FPFH...")
    target_scan_down, target_scan_fpfh = preprocess_point_cloud_for_features(
        target_obj_scan, 
        voxel_size_global_registration
    )

    global_reg_fitness = 0.0 
    T_after_ransac = copy.deepcopy(current_best_T_cad_to_scan) # Initialize with centroid align transform

    if source_cad_down.has_points() and target_scan_down.has_points() and \
       source_cad_fpfh is not None and target_scan_fpfh is not None:
        
        current_ransac_iter = default_ransac_iterations
        current_ransac_conf = default_ransac_confidence
        
        if is_target_combined_due_to_separation_failure:
            print("WARNING: Color separation failed, target includes gripper. Using more lenient RANSAC parameters.")
            current_ransac_iter = max(1000, int(default_ransac_iterations / 2)) # Ensure at least 1000
            current_ransac_conf = 0.90  
            
        print(f"Running RANSAC global registration (iter: {current_ransac_iter}, conf: {current_ransac_conf})...")
        # RANSAC will find a transform T_s_to_t to align source_cad_down (from manually rotated source)
        # to target_scan_down. This T_s_to_t is what we want to apply to source_cad_model_scaled.
        T_ransac_from_current_source_to_target, global_reg_fitness = global_registration_fpfh_ransac(
            source_cad_down,    
            target_scan_down,
            source_cad_fpfh,    
            target_scan_fpfh,
            voxel_size_global_registration,
            max_iterations=current_ransac_iter, 
            confidence=current_ransac_conf
        )
        
        if global_reg_fitness > 0.05: 
             # T_ransac_from_current_source_to_target is the transform to apply to the 
             # *manually rotated* source_cad_model_scaled to align it with target_obj_scan.
             T_after_ransac = T_ransac_from_current_source_to_target 
             print(f"Global RANSAC successful. Fitness: {global_reg_fitness:.4f}.")
        else:
             print(f"Global RANSAC fitness very low ({global_reg_fitness:.4f}) or RANSAC failed. Keeping previous (centroid) alignment for ICP.")
             # T_after_ransac remains the centroid alignment transform
    else:
        print("Skipping Global RANSAC Registration: Preprocessed data or FPFH features missing / too sparse. Using centroid alignment for ICP.")

    current_best_T_cad_to_scan = T_after_ransac # Update overall best transform so far

    print("\n--- Visualizing after Global Registration (or Centroid if RANSAC failed/skipped) ---")
    visualize_scene_with_original_colors(
        [target_obj_scan, gripper_visualization_cloud] if gripper_visualization_cloud.has_points() else [target_obj_scan], 
        source_cad_model_scaled, # This is the manually rotated source
        current_best_T_cad_to_scan, # This transform aligns the manually rotated source to the target
        window_name="[VIS 3] After Global Registration Attempt"
    )

    print("\n--- Performing Final ICP Refinement ---")
    # ICP refines the current_best_T_cad_to_scan, which applies to the manually rotated source_cad_model_scaled
    source_for_final_icp = copy.deepcopy(source_cad_model_scaled) # Already manually rotated
    target_for_final_icp = copy.deepcopy(target_obj_scan)

    if not source_for_final_icp.has_points() or not target_for_final_icp.has_points():
        print("Error: Point clouds for final ICP are empty. Cannot perform refinement.")
    else:
        final_T_from_current_source_to_target, final_icp_fitness = refine_registration_icp(
            source_for_final_icp, 
            target_for_final_icp, 
            current_best_T_cad_to_scan, # Initial guess is transform for manually rotated source
            final_icp_max_distance,
            icp_type=final_icp_type,
            max_iterations=final_icp_iterations,
            relative_fitness=1e-7, relative_rmse=1e-7
        )
        
        if final_icp_fitness > 0.01 : 
            if global_reg_fitness < 0.05 or final_icp_fitness > global_reg_fitness * 0.8: 
                current_best_T_cad_to_scan = final_T_from_current_source_to_target
                print(f"Used ICP refined transformation. Fitness: {final_icp_fitness:.4f}")
            else:
                print(f"ICP fitness ({final_icp_fitness:.4f}) not significantly better than Global Reg ({global_reg_fitness:.4f}). Kept Global Reg result.")
        else:
            print(f"Final ICP fitness very low ({final_icp_fitness:.4f}). Kept previous transformation.")

    # current_best_T_cad_to_scan is now the refined transform to apply to the
    # *manually rotated* source_cad_model_scaled to align it to target_obj_scan.

    # To get the *total* transformation from the *original unrotated* source_high_res_cloud_orig
    # to the final aligned pose, we need to compose the manual rotation and current_best_T_cad_to_scan.
    # However, for visualization and saving, it's simpler to transform the manually rotated source_cad_model_scaled
    # by current_best_T_cad_to_scan, OR, apply all transformations sequentially to the absolute original.

    # Let's reconstruct the final transformed CAD model from the absolute original
    source_cad_model_final_transformed = copy.deepcopy(source_high_res_cloud_orig)
    if SOURCE_SCALE_FACTOR != 1.0:
        source_cad_model_final_transformed.scale(SOURCE_SCALE_FACTOR, center=source_cad_model_final_transformed.get_center())

    # Apply manual rotation first
    if manual_rotate_x_deg != 0.0 or manual_rotate_y_deg != 0.0 or manual_rotate_z_deg != 0.0:
        center_orig_scaled = source_cad_model_final_transformed.get_center()
        if manual_rotate_x_deg != 0.0:
            R_x_mat = source_cad_model_final_transformed.get_rotation_matrix_from_xyz((np.deg2rad(manual_rotate_x_deg), 0, 0))
            source_cad_model_final_transformed.rotate(R_x_mat, center=center_orig_scaled)
        if manual_rotate_y_deg != 0.0:
            R_y_mat = source_cad_model_final_transformed.get_rotation_matrix_from_xyz((0, np.deg2rad(manual_rotate_y_deg), 0))
            source_cad_model_final_transformed.rotate(R_y_mat, center=center_orig_scaled)
        if manual_rotate_z_deg != 0.0:
            R_z_mat = source_cad_model_final_transformed.get_rotation_matrix_from_xyz((0, 0, np.deg2rad(manual_rotate_z_deg)))
            source_cad_model_final_transformed.rotate(R_z_mat, center=center_orig_scaled)
            
    # Then apply the alignment transformation (current_best_T_cad_to_scan)
    source_cad_model_final_transformed.transform(current_best_T_cad_to_scan)    

    print("\nFinal transformation matrix stored (current_best_T_cad_to_scan) is the one that aligns the *manually pre-rotated* source CAD model to the target scan.")
    print(current_best_T_cad_to_scan)


    print("\n--- Visualizing Final Registered State (CAD model aligned to Target Scan) ---")
    if source_cad_model_final_transformed.has_points():
        visualize_scene_with_original_colors(
            [target_combined_cloud], 
            source_cad_model_final_transformed, # This is now fully transformed from original
            transformation=np.identity(4), 
            window_name="[VIS 4] Final Alignment: CAD on Scan"
        )
    else:
        print("Skipping Final Visualization: Transformed CAD model is empty.")
    
    print("\n--- Saving Results ---")
    if source_cad_model_final_transformed.has_points():
        o3d.io.write_point_cloud(os.path.join(output_dir, "source_cad_model_aligned_to_scan.ply"), source_cad_model_final_transformed)
    if target_obj_scan.has_points():
        o3d.io.write_point_cloud(os.path.join(output_dir, "target_obj_from_scan.ply"), target_obj_scan)
    if gripper_visualization_cloud.has_points():
        o3d.io.write_point_cloud(os.path.join(output_dir, "gripper_from_scan.ply"), gripper_visualization_cloud)
    
    # It might be more useful to save the total transformation from the original unrotated CAD
    # However, current_best_T_cad_to_scan is what was optimized by ICP/RANSAC on the manually rotated source.
    # For simplicity, we save this one. If total transform is needed, it requires composing manual rotation.
    np.savetxt(os.path.join(output_dir, "final_align_transform_for_manually_rotated_source.txt"), current_best_T_cad_to_scan)
    print(f"Saved results and transformation to {output_dir}")
    print("\n--- Script Finished ---")