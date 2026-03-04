import cv2
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from PIL import Image
from copy import deepcopy
import random
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import re
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

output_dir = "output_images"
cv2.setLogLevel(3) 
object_dimensions = {
    0: [0.58, 1.07, 0.58],  # Example for a barrel
    1: [0.3, 1, 0.2],  # Example for a vertical panel
    2: [0.368, 0.9, 0.368],  # Example for a cone
    3: [10,10,10] # Example for a guardrail
}

idx_to_class_name = {
    0: "Barrel",
    1: "Channelizer",
    2: "Cone",
    3: "Guardrail",
}


# Function to save the camera image in KITTI image_2 folder
def save_kitti_images(images, output_folder, frame_idx, front_camera_flag):
    # Define camera names and their corresponding KITTI folder names
    camera_folders = {
        "FRONT_ZOOMED_OUT_CAMERA": "image_2",
        "FRONT_CAMERA": "image_1",
        "FRONT_ZOOMED_IN_CAMERA": "image_0"
    }
    
    for camera_name, image in images.items():
        if camera_name in camera_folders:
            folder_name = camera_folders[camera_name]
            folder_path = os.path.join(output_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            filename = os.path.join(folder_path, f"{frame_idx}.png")
            cv2.imwrite(filename, image)
        else:
            print(f"Warning: Unknown camera type {camera_name}")

# Function to save the combined LiDAR points as a .bin file in KITTI velodyne folder.
def save_kitti_velodyne(points, output_folder, frame_idx):
    filename = os.path.join(output_folder, f"{frame_idx}.bin")
    points_with_intensity_one = np.hstack((points, np.ones((points.shape[0], 1))))
    points_with_intensity_one.astype(np.float32).tofile(filename)

def save_kitti_calib(camera_params, extrinsics, output_folder, frame_idx, front_camera_flag):

    filename = os.path.join(output_folder, f"{frame_idx}.txt")
    with open(filename, "w") as f:
        # Camera projections (P0-P2)
        if front_camera_flag:
            for cam_idx, cam_key in enumerate(["FRONT_ZOOMED_IN_CAMERA", 
                                            "FRONT_CAMERA", 
                                            "FRONT_ZOOMED_OUT_CAMERA","FRONT_ZOOMED_OUT_CAMERA"]):
                cam = camera_params[cam_key]
                fx, fy, cx, cy = cam["projection"]
                P = np.array([[fx, 0, cx, 0],
                            [0, fy, cy, 0],
                            [0, 0, 1, 0]])
                f.write(f"P{cam_idx}: {' '.join([f'{x:.6f}' for x in P.flatten()])}\n")
        else:
            for cam_idx, cam_key in enumerate(["FRONT_ZOOMED_IN_CAMERA", 
                                            "FRONT_ZOOMED_OUT_CAMERA", 
                                            "FRONT_ZOOMED_OUT_CAMERA","FRONT_ZOOMED_OUT_CAMERA"]):
                cam = camera_params[cam_key]
                fx, fy, cx, cy = cam["projection"]
                P = np.array([[fx, 0, cx, 0],
                            [0, fy, cy, 0],
                            [0, 0, 1, 0]])
                f.write(f"P{cam_idx}: {' '.join([f'{x:.6f}' for x in P.flatten()])}\n")

        f.write(f"R0_rect: {' '.join([f'{x:.6f}' for x in np.eye(3).flatten()])}\n")

        # === Write Tr for each camera matching P0,P1,P2 ===
        # 0) FRONT_ZOOMED_IN_CAMERA  -> Tr_velo_to_cam_0
        R0_cam = quaternion_to_rotation_matrix(camera_params["FRONT_ZOOMED_IN_CAMERA"]["rotation"])
        t0_cam = np.array(camera_params["FRONT_ZOOMED_IN_CAMERA"]["translation"]).reshape(3,1)
        Tr0 = np.hstack((R0_cam, t0_cam))
        f.write("Tr_velo_to_cam_0: " + " ".join(f"{x:.6f}" for x in Tr0.flatten()) + "\n")
        f.write("Tr_imu_to_cam_0: " + " ".join(f"{x:.6f}" for x in Tr0.flatten()) + "\n")

        if front_camera_flag:
            # 1) FRONT_CAMERA            -> Tr_velo_to_cam_1
            R1_cam = quaternion_to_rotation_matrix(camera_params["FRONT_CAMERA"]["rotation"])
            t1_cam = np.array(camera_params["FRONT_CAMERA"]["translation"]).reshape(3,1)
            Tr1 = np.hstack((R1_cam, t1_cam))
            f.write("Tr_velo_to_cam_1: " + " ".join(f"{x:.6f}" for x in Tr1.flatten()) + "\n")
            f.write("Tr_imu_to_cam_1: " + " ".join(f"{x:.6f}" for x in Tr1.flatten()) + "\n")

        # 2) FRONT_ZOOMED_OUT_CAMERA -> Tr_velo_to_cam  (and _2 for completeness)
        R2_cam = quaternion_to_rotation_matrix(camera_params["FRONT_ZOOMED_OUT_CAMERA"]["rotation"])
        t2_cam = np.array(camera_params["FRONT_ZOOMED_OUT_CAMERA"]["translation"]).reshape(3,1)
        Tr2 = np.hstack((R2_cam, t2_cam))
        f.write("Tr_velo_to_cam: "   + " ".join(f"{x:.6f}" for x in Tr2.flatten()) + "\n")  # cam2 (KITTI canonical)
        f.write("Tr_imu_to_cam: "    + " ".join(f"{x:.6f}" for x in Tr2.flatten()) + "\n")
        f.write("Tr_velo_to_cam_2: " + " ".join(f"{x:.6f}" for x in Tr2.flatten()) + "\n")
        f.write("Tr_imu_to_cam_2: "  + " ".join(f"{x:.6f}" for x in Tr2.flatten()) + "\n")

        return Tr2, np.eye(3)

def save_kitti_label(projected_corners, bboxes_3d, classes_per_corner, output_folder, frame_idx, front_camera_flag, Tr_velo_to_cam, R0_rect):
    # Assume projected_corners is a 2D NumPy array with shape (8 * num_boxes, 2)
    num_corners_per_box = 8
    num_boxes = len(projected_corners) // num_corners_per_box
    filename = os.path.join(output_folder, f"{frame_idx}.txt")
    
    with open(filename, "w") as f:
        for i in range(num_boxes):
            box = projected_corners[i * num_corners_per_box:(i + 1) * num_corners_per_box]
            box_3d = bboxes_3d[i * num_corners_per_box:(i + 1) * num_corners_per_box]
            center_of_bbox = np.mean(box_3d, axis=0) 
            x_min, y_min = np.min(box, axis=0)
            x_max, y_max = np.max(box, axis=0)
            class_num=classes_per_corner[i * num_corners_per_box]
            # print(class_num)
            class_name=idx_to_class_name[int(class_num)]
            #Updated to fix the ordering and sign issue wrt KITTI format

            x_vehicle, y_vehicle, z_vehicle = center_of_bbox[0], center_of_bbox[1], center_of_bbox[2]
            # Transform from vehicle frame to camera frame
            # The bbox center is already in vehicle coordinates, so we need to transform it to camera coordinates
            # using the camera's rotation and translation
            R_cam = Tr_velo_to_cam[:, :3]
            t_cam = Tr_velo_to_cam[:, 3]
            center_of_bbox = R_cam @ np.array([x_vehicle, y_vehicle, z_vehicle]) + t_cam

            #KITTI marks annotation location as center of bottom face of bbox
            updated_location = np.array([center_of_bbox[0], center_of_bbox[1]+object_dimensions[int(class_num)][1]/2, center_of_bbox[2]])

            #For dimensions, first one is height when actually writing the label, second is length, third is width. In below code.

            label_line = (
                f"{class_name} 0.00 0 0.0 {x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f} "
                f"{object_dimensions[int(class_num)][1]} {object_dimensions[int(class_num)][2]} {object_dimensions[int(class_num)][0]} "
                f"{updated_location[0]} {updated_location[1]} {updated_location[2]} 0.0\n"
            )
            f.write(label_line)

from datetime import datetime
import os, json

def save_vehicle_states(frame_idx, ref_ts_prefix, vehicle_states_list, out_dir, original_folder_name):
    """
    Save the vehicle state nearest to ref_ts_prefix (e.g., '120241124T153141.282408')
    into out_dir/<frame_idx>.json. Includes the original timestamp string.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Parse the ref timestamp for matching
    try:
        ref_dt = _parse_fileprefix_ts(ref_ts_prefix)
    except Exception:
        ref_dt = _parse_iso_ts(ref_ts_prefix)

    # Build index of states and pick nearest
    times_sorted, states_sorted = _build_state_index(vehicle_states_list)
    nearest = _nearest_state(times_sorted, states_sorted, ref_dt, max_gap_s=2.0)

    if nearest is None:
        if times_sorted:
            diffs = [abs((t - ref_dt).total_seconds()) for t in times_sorted]
            nearest = states_sorted[int(diffs.index(min(diffs)))]
        else:
            nearest = {}

    # Wrap with original timestamp string
    out_data = {
        "ref_timestamp_prefix": ref_ts_prefix,  
        "original_folder_name": original_folder_name,
        "closest_vehicle_state": nearest
    }

    # Ensure JSON-serializable
    def _jsonable(o):
        if isinstance(o, datetime): return o.isoformat()
        if isinstance(o, dict):     return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, list):     return [_jsonable(v) for v in o]
        return o

    out_path = os.path.join(out_dir, f"{frame_idx}.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(out_data), f, ensure_ascii=False, indent=4)

# Convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(quat):
   w, x, y, z  = quat
   return np.array([
       [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
       [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
       [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
   ])

def transform_to_vehicle_frame(xyz_points, extrinsic):
        R_add = np.array([[0,1,0],[1,0,0],[0,0,-1]])
        quat = extrinsic["rotation_wxyz"]
        rotation_matrix = quaternion_to_rotation_matrix(quat) @ R_add
        translation = np.array(extrinsic["translation"])
        return (xyz_points @ rotation_matrix.T) + translation


def load_lidar_raw(lidar_file, intensity_threshold, velocity_mps, scan_duration,
                   theta_deg, offset, yaw_rate_rad_s=0.0):
    points = np.fromfile(lidar_file, dtype=np.float32).reshape((-1, 6))
    x_raw, y_raw, z_raw, intensity = points[:,0], points[:,1], points[:,2], points[:,3]

    azimuth = np.rad2deg(np.arctan2(x_raw, y_raw))
    azimuth = ((azimuth - offset) + 360.0) % 360.0
    rel_time = (azimuth / 360.0) * scan_duration                     # [0, scan_duration]

    theta_rad = np.deg2rad(theta_deg)

    # Translational motion in LiDAR frame
    delta_x = velocity_mps * rel_time * np.cos(theta_rad)
    delta_y = velocity_mps * rel_time * np.sin(theta_rad)

    # NEW: rotational deskew around Z (planar yaw change)
    delta_yaw = yaw_rate_rad_s * rel_time                             # radians
    c, s = np.cos(delta_yaw), np.sin(delta_yaw)

    # rotate each point by -delta_yaw to a common time (end-of-scan)
    x_rot =  c * x_raw + s * y_raw
    y_rot = -s * x_raw + c * y_raw

    # then subtract the translational motion during the scan
    x_deskewed = x_rot - delta_x
    y_deskewed = y_rot - delta_y

    mask = intensity > intensity_threshold
    return np.stack([x_deskewed[mask], y_deskewed[mask], z_raw[mask]], axis=1)


# Project LIDAR points to image plane
def project_lidar_to_camera(point_cloud, intrinsic, rotation, translation, distortion,camera_name):
    # Apply camera rotation and translation

    points_transformed = point_cloud
    points_transformed = (rotation @ points_transformed.T).T  + translation
    # Filter points in front of the camera
    points_in_front = points_transformed[points_transformed[:, 2] > 0]
    # Calculate depth (distance from camera)
    depths = points_in_front[:, 2]
    min_depth, max_depth = depths.min(), depths.max()
    normalized_depths = (depths - min_depth) / (max_depth - min_depth)
    # Project points onto 2D image plane
    img_points, _ = cv2.projectPoints(points_in_front, np.zeros((3, 1)), np.zeros((3, 1)), intrinsic, distortion)
    return img_points.reshape(-1, 2), normalized_depths, points_in_front

def project_lidar_to_camera_v2(lidar_points, params):
    """
    Project LiDAR points into the camera frame.
    
    Args:
        lidar_points (numpy.ndarray): Nx3 array of LiDAR points.
        params (dict): Camera parameters including projection matrix and distortion coefficients.
    
    Returns:
        numpy.ndarray: 2D points on the image plane,
        numpy.ndarray: Depth values of the projected points.
        numpy.ndarray: 3D points in the camera frame.
    """

    new_cam_mtx = np.array([
        [params['new_projection'][0], 0, params['new_projection'][2]], 
        [0, params['new_projection'][1], params['new_projection'][3]], 
        [0, 0, 1]
    ])

    # Project LiDAR points into the camera frame
    points_transformed = lidar_points
    rotation = quaternion_to_rotation_matrix(params["rotation"])
    translation = np.array(params["translation"])
    points_transformed = rotation.dot(points_transformed.T)  + translation.reshape(-1, 1)

    # points_in_front = points_transformed[2, :] > 0

    depth = points_transformed[2, :]
    mask = depth > 0  # Filter points in front of the camera
    points_transformed = points_transformed[:, mask]
    depth = depth[mask]

    # Project points onto the 2D image plane
    img_points = new_cam_mtx.dot(points_transformed)
    point2D = img_points[:2, :] / img_points[2:, :]
    point2D = point2D.T

    return point2D, depth, points_transformed.T

def project_bbox_to_camera(point_cloud, params):
    fx, fy, cx, cy = params["projection"]
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    distortion = np.array(params["distortion"])
    points_in_front = point_cloud
    # Calculate depth (distance from camera)
    depths = np.linalg.norm(points_in_front, axis=1)
    # Project points onto 2D image plane
    img_points, _ = cv2.projectPoints(points_in_front, np.zeros((3, 1)), np.zeros((3, 1)), intrinsic, distortion)

    return img_points.reshape(-1, 2), depths, points_in_front


def project_3dpoint_to_camera(point_cloud, params):
    fx, fy, cx, cy = params["projection"]
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    distortion = np.array(params["distortion"])
    img_points, _ = cv2.projectPoints(point_cloud, np.zeros((3, 1)), np.zeros((3, 1)), intrinsic, distortion)
    return img_points.reshape(-1, 2)


import time

def process_data(folder_path, timestamp, extrinsics, camera_params, camera_name, right_lidar_flag, all_items_in_folder):
    # t_total_start = time.perf_counter()


    start_filter = time.perf_counter()
    files = [f for f in all_items_in_folder if timestamp in f]
    end_filter = time.perf_counter()
    filter_time = end_filter - start_filter

    # print(f"Filtering took {filter_time:.6f} seconds")

    # t_init_start = time.perf_counter()
    image_files = {
        "FRONT_CAMERA": None,
        "FRONT_ZOOMED_IN_CAMERA": None,
        "FRONT_ZOOMED_OUT_CAMERA": None
    }
    lidar_files = {f"lidar{i}": None for i in range(6)}
    # t_init_end = time.perf_counter()
    # print(f"[Timing] Dict initialization: {t_init_end - t_init_start:.4f}s")

    # t_classify_start = time.perf_counter()
    for f in files:
        if "Front_Camera" in f:
            image_files["FRONT_CAMERA"] = os.path.join(folder_path, f)
        elif "Front_Zoomed_In_Camera" in f:
            image_files["FRONT_ZOOMED_IN_CAMERA"] = os.path.join(folder_path, f)
        elif "Front_Zoomed_Out_Camera" in f:
            image_files["FRONT_ZOOMED_OUT_CAMERA"] = os.path.join(folder_path, f)
        elif "lidar0" in f:
            lidar_files["lidar0"] = os.path.join(folder_path, f)
        elif "lidar1" in f:
            lidar_files["lidar1"] = os.path.join(folder_path, f)
        elif "lidar2" in f:
            lidar_files["lidar2"] = os.path.join(folder_path, f)
        elif "lidar3" in f:
            lidar_files["lidar3"] = os.path.join(folder_path, f)
        elif "lidar4" in f:
            lidar_files["lidar4"] = os.path.join(folder_path, f)
        elif "lidar5" in f:
            lidar_files["lidar5"] = os.path.join(folder_path, f)
    # t_classify_end = time.perf_counter()
    # print(f"[Timing] File classification: {t_classify_end - t_classify_start:.4f}s")

    # LIDAR loading
    # t_load_start = time.perf_counter()
    vehicle_speed_mps = 0
    scan_period_s = 0.1
    raw0 = load_lidar_raw(lidar_files['lidar0'],0,vehicle_speed_mps,scan_period_s,15,45)
    raw1 = load_lidar_raw(lidar_files['lidar1'],0,vehicle_speed_mps,scan_period_s,45,90)
    raw2 = load_lidar_raw(lidar_files['lidar2'],0,vehicle_speed_mps,scan_period_s,-45,90)
    raw3 = load_lidar_raw(lidar_files['lidar3'],0,vehicle_speed_mps,scan_period_s,180,90)
    if right_lidar_flag:
        raw4 = load_lidar_raw(lidar_files['lidar4'],0,vehicle_speed_mps,scan_period_s,-180,90)
    raw5 = load_lidar_raw(lidar_files['lidar5'],0,vehicle_speed_mps,scan_period_s,360,90)
    # t_load_end = time.perf_counter()
    # print(f"[Timing] LIDAR loading: {t_load_end - t_load_start:.4f}s")

    # Transformations
    # t_transform_start = time.perf_counter()
    pts0 = transform_to_vehicle_frame(raw0, extrinsics.get('lidar0', {}))
    pts1 = transform_to_vehicle_frame(raw1, extrinsics.get('lidar1', {}))
    pts2 = transform_to_vehicle_frame(raw2, extrinsics.get('lidar2', {}))
    pts3 = transform_to_vehicle_frame(raw3, extrinsics.get('lidar3', {}))
    if right_lidar_flag:
        pts4 = transform_to_vehicle_frame(raw4, extrinsics.get('lidar4', {}))
    pts5 = transform_to_vehicle_frame(raw5, extrinsics.get('lidar5', {}))
    # t_transform_end = time.perf_counter()
    # print(f"[Timing] Transformations: {t_transform_end - t_transform_start:.4f}s")

    # Concatenation
    # t_concat_start = time.perf_counter()
    calibrated_points = []
    calibrated_points.append(pts0)
    calibrated_points.append(pts1)
    calibrated_points.append(pts2)
    calibrated_points.append(pts3)
    if right_lidar_flag:
        calibrated_points.append(pts4)
    calibrated_points.append(pts5)
    if not calibrated_points:
        print("[ERROR] No valid LIDAR data after calibration.")
        return
    combined_points = np.vstack(calibrated_points)
    raw_combined_points = deepcopy(combined_points)
    # t_concat_end = time.perf_counter()
    # print(f"[Timing] Concatenation: {t_concat_end - t_concat_start:.4f}s")

    # Filtering
    # t_filter_start = time.perf_counter()
    x, y, z = combined_points[:, 0], combined_points[:, 1], combined_points[:, 2]
    mask_outside_rect = ~((x >= -1.0) & (x <= 4.0) & (y >= -1) & (y <= 1))
    mask_above_ground = z > 0.2
    combined_points = combined_points[mask_outside_rect & mask_above_ground]
    return combined_points, raw_combined_points

def create_mask_from_segments(segments, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for segment in segments:
        points = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 255)
    return mask

def cluster_points(points, eps=1, min_samples=1):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(points)
    return clusters

def create_bounding_box(points):
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    bbox=o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    corners = np.asarray(bbox.get_box_points())
    
    return corners, bbox

def create_fixed_bounding_box(centroid, length=object_dimensions[0][0], width=object_dimensions[0][1], height=object_dimensions[0][2]):
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2
    
    min_bound = centroid - [half_length, half_width, half_height]
    max_bound = centroid + [half_length, half_width, half_height]

    
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    corners = np.asarray(bbox.get_box_points())
    
    return corners, bbox

def find_cluster_with_smallest_centroid_norm(clusters, points_3d, segments, camera_name):
    bounding_boxes = []
    all_corners = []

    # Dictionary to store clusters per segment
    segment_clusters = {i: [] for i in range(len(segments))}

    for cluster_id in set(clusters):
        if cluster_id == -1:
            continue
        cluster_points = points_3d[clusters == cluster_id]
        centroid_lidar = np.mean(cluster_points, axis=0)
        centroid_2d = project_3dpoint_to_camera(centroid_lidar.reshape(1, 3),camera_params[camera_name])[0]

        # Assign the cluster to the first segment that contains the 2D centroid
        for i, segment in enumerate(segments):
            segment_array = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
            if cv2.pointPolygonTest(segment_array, tuple(map(int, centroid_2d)), False) >= 0:
                segment_clusters[i].append((cluster_id, centroid_lidar, cluster_points))
                break

    # For each segment, choose the best cluster (e.g. with smallest centroid norm)
    for segment_id, cluster_list in segment_clusters.items():
        if not cluster_list:
            continue
        best_cluster = min(cluster_list, key=lambda x: np.linalg.norm(x[1]))
        centroid = best_cluster[1]
        bbox_corners, bounding_box = create_fixed_bounding_box(centroid)
        bounding_box.color = np.random.rand(3)
        bounding_boxes.append(bounding_box)
        all_corners.append(bbox_corners)
        

    return bounding_boxes, all_corners


def create_lidar_only_bboxes(clusters, points_3d, camera_name, camera_params):
    """
    Create 3D bounding boxes purely from LiDAR clusters when no 2D segments are available.
    Uses default class (0 = Barrel) for all detections.
    """
    bounding_boxes = []
    all_corners = []
    all_corner_classes = []
    
    for cluster_id in set(clusters):
        if cluster_id == -1:
            continue
        cluster_points = points_3d[clusters == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        
        # Create bbox with default dimensions (Barrel)
        bbox_corners, bounding_box = create_fixed_bounding_box(
            centroid, 
            object_dimensions[0][0],  # length
            object_dimensions[0][1],  # width  
            object_dimensions[0][2]   # height
        )
        
        bounding_box.color = np.random.rand(3)
        bounding_boxes.append(bounding_box)
        all_corners.append(bbox_corners)
        all_corner_classes.append(np.full((bbox_corners.shape[0], 1), 0))  # Default class 0 (Barrel)
    
    return bounding_boxes, all_corners, all_corner_classes


def find_cluster_with_smallest_centroid_norm_and_assign_classes(clusters, points_3d, segments, classes, camera_name, camera_params):
    bounding_boxes = []
    all_corners = []
    all_corner_classes = []
    # Dictionary to store clusters per segment
    segment_clusters = {i: [] for i in range(len(segments))}

    for cluster_id in set(clusters):
        if cluster_id == -1:
            continue
        cluster_points = points_3d[clusters == cluster_id]
        centroid_lidar = np.mean(cluster_points, axis=0)
        centroid_2d = project_3dpoint_to_camera(centroid_lidar.reshape(1, 3),camera_params[camera_name])[0]

        # Assign the cluster to the first segment that contains the 2D centroid
        for i, segment in enumerate(segments):
            segment_array = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
            if cv2.pointPolygonTest(segment_array, tuple(map(int, centroid_2d)), False) >= 0:
                segment_clusters[i].append((cluster_id, centroid_lidar, cluster_points))
                break

    # For each segment, choose the best cluster (e.g. with smallest centroid norm)
    for segment_id, cluster_list in segment_clusters.items():
        if not cluster_list:
            continue
        best_cluster = min(cluster_list, key=lambda x: np.linalg.norm(x[1]))
        centroid = best_cluster[1]
        bbox_corners, bounding_box = create_fixed_bounding_box(centroid, object_dimensions[int(classes[segment_id])][0], object_dimensions[int(classes[segment_id])][1], object_dimensions[int(classes[segment_id])][2])

        #New logic to ensure at least one point is inside the bounding box
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(np.min(bbox_corners, axis=0), 
                                                np.max(bbox_corners, axis=0))
        points_in_bbox = bbox.get_point_indices_within_bounding_box(
                        o3d.utility.Vector3dVector(best_cluster[2]))

        bounding_box.color = np.random.rand(3)
        bounding_boxes.append(bounding_box)
        all_corners.append(bbox_corners)
        all_corner_classes.append(np.full((bbox_corners.shape[0], 1), classes[segment_id]))
        # print(bbox_corners.shape)
        # print(np.full((bbox_corners.shape[0], 1), classes[segment_id]).shape)


    return bounding_boxes, all_corners, all_corner_classes


def get_depth_map(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    depth = pipe(image)["depth"]
    depth_map = np.array(depth)
    
    return depth_map

def linear_regression_model(base_depth,camera_name):
    if camera_name == "FRONT_CAMERA":
        return 0.5276*base_depth - 0.3461
    elif camera_name == "FRONT_ZOOMED_IN_CAMERA":
        return 0.7184*base_depth +0.8311
    elif camera_name == "FRONT_ZOOMED_OUT_CAMERA":
        return 0.1950*base_depth -0.8686



def calculate_segment_depths(segments, depth_map):
    segment_depths = {}
    
    for i, segment in enumerate(segments):
        segment_array = np.array(segment, dtype=np.int32).reshape((-1, 2))
        x_min, y_min = np.min(segment_array, axis=0)
        x_max, y_max = np.max(segment_array, axis=0)
        
        sampled_depths = []
        for _ in range(10000):
            while True:
                x = np.random.randint(x_min, x_max)
                y = np.random.randint(y_min, y_max)
                if cv2.pointPolygonTest(segment_array, (x, y), False) >= 0:
                    sampled_depths.append(depth_map[y, x])
                    break
        
        avg_depth = np.mean(sampled_depths)
        segment_depths[i] = avg_depth
    
    return segment_depths


def project_2d_to_3d(point_2d, depth, intrinsic_matrix, rotation_matrix, translation_vector):
    """
    Projects a 2D image point with known depth back into 3D world coordinates.

    Parameters:
    - point_2d: Tuple or array-like (u, v) pixel coordinates.
    - depth: Scalar depth value corresponding to the pixel.
    - intrinsic_matrix: Camera intrinsic parameters (3x3 matrix).
    - rotation_matrix: Camera rotation matrix (3x3 matrix), world-to-camera.
    - translation_vector: Camera translation vector (3x1 vector), world-to-camera.

    Returns:
    - point_3d: The 3D point coordinates in world space.
    """
    # Compute inverse of intrinsic matrix
    inv_intrinsic = np.linalg.inv(intrinsic_matrix)

    # Convert pixel coordinates to homogeneous coordinates
    homogeneous_pixel = np.array([point_2d[0], point_2d[1], 1])

    # Compute the point in camera coordinate system
    camera_point = inv_intrinsic @ homogeneous_pixel * depth

    return camera_point

def create_3d_bounding_box(center, dimensions):
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = dimensions
    bbox.R = np.eye(3)  # Assuming no rotation
    return bbox

def draw_3d_bounding_box(image, projected_points, color=(0, 255, 0)):
    edges = [(0, 1),(1, 7),(7,2),(2,0), #front face
            (3,6),(6,4),(4,5),(5,3), #back face
            (0,3),(2,5),(7,4),(1,6) ]  # Connecting edges

    for edge in edges:
        pt1 = tuple(map(int, projected_points[edge[0]]))
        pt2 = tuple(map(int, projected_points[edge[1]]))
        cv2.line(image, pt1, pt2, color, 2)

    return image

# Main processing
def process_image(camera_name, image, depth_map, segments, camera_params):
    # Calculate segment depths
    params = camera_params[camera_name]
    intrinsic = np.array([
        [params['projection'][0], 0, params['projection'][2]],
        [0, params['projection'][1], params['projection'][3]],
        [0, 0, 1]
    ])
    rotation = quaternion_to_rotation_matrix(params["rotation"])
    translation = np.array(params["translation"])
    distortion = np.array(params["distortion"])

    segment_depths = calculate_segment_depths(segments, depth_map)

    # Create a copy of the original image for visualization
    image_with_segments = image.copy()

    # Visualize each segment with its average depth
    for i, segment in enumerate(segments):
        points = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.polylines(image_with_segments, [points], isClosed=True, color=color, thickness=2)

        M = cv2.moments(points)
        if M['m00'] != 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        else:
            cx, cy = points[0][0]

        avg_depth = segment_depths.get(i, 0)
        cv2.putText(image_with_segments, f"{i}", (cx, cy-pow((-1),i+1)*50), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        # cv2.imshow("Segments with Average Depths", image_with_segments)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Calculate 3D centroids and create bounding boxes
    bounding_boxes_3d = []
    for i, segment in enumerate(segments):
        x_values = [pair[0] for pair in segment]
        y_values = [pair[1] for pair in segment]

        cx = sum(x_values) / len(x_values)
        cy = sum(y_values) / len(y_values)

        avg_depth = segment_depths.get(i, 0)
        # center_3d = pixel_to_3d(cx, cy, avg_depth, camera_intrinsics)
        center_3d = project_2d_to_3d((cx,cy), avg_depth, intrinsic, rotation, translation)

        bbox_3d = create_3d_bounding_box(center_3d, object_dimensions[0])
        bounding_boxes_3d.append(bbox_3d)

    #Draw the x y z axis looking like axes on the o3d window
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # Convert each OrientedBoundingBox to a LineSet
    line_sets = [obb_to_lineset(obb) for obb in bounding_boxes_3d]

    # Visualize all geometries
    geometries = line_sets + [coordinate_frame]
    # o3d.visualization.draw_geometries(geometries)
    #Get corners of the bounding boxes
    bounding_boxes_3d = [bbox.get_box_points() for bbox in bounding_boxes_3d]
    final_corners = np.vstack(bounding_boxes_3d)

    # #Project these corners back to the image.
    projected_corners, depths , _ = project_bbox_to_camera(final_corners, camera_params[camera_name])

    projected_corners = projected_corners.astype(int)

    # Step 5: Color the points based on depth and overlay them on the image
    cmap = plt.get_cmap("autumn")
    colors = (cmap(depths / np.max(depths))[:, :3] * 255).astype(np.uint8)

    image_with_bbox = image.copy()
    for (point, color) in zip(projected_corners, colors):
        x, y = point
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.circle(image_with_bbox, (x, y), 2, color_bgr, -1)

    image_path_with_bbox = os.path.join(output_dir, f"{camera_name}_with_bbox_points_in_process_img.png")
    cv2.imwrite(image_path_with_bbox, image_with_bbox)

    image_with_drawn_boxes = image.copy()

    # Iterate through the projected corners in chunks of 8
    for i in range(0, len(projected_corners), 8):
        bbox_corners = projected_corners[i:i+8]
        image_with_drawn_boxes = draw_3d_bounding_box(image_with_drawn_boxes, bbox_corners)

    # Save the image with 3D bounding boxes
    output_path_with_boxes = os.path.join(output_dir, f"{camera_name}_with_3d_bounding_boxes.png")
    cv2.imwrite(output_path_with_boxes, image_with_drawn_boxes)
    # print(f"Image with 3D bounding boxes saved at: {output_path_with_boxes}")

    # # Project 3D bounding boxes back to the image
    # for bbox in bounding_boxes_3d:
    #     bbox_points = bbox.get_box_points()
    #     projected_points = project_3dpoint_to_camera(bbox_points, camera_intrinsics, None, None, distortion, camera_name)
    #     image_with_segments = draw_3d_bounding_box(image_with_segments, projected_points)

    # return image_with_segments, bounding_boxes_3d
    return final_corners

def viz_bboxes_on_image(boxes, image, camera_name, camera_params):
    final_corners = boxes

    # #Project these corners back to the image.
    projected_corners, depths , _ = project_bbox_to_camera(final_corners, camera_params[camera_name])

    projected_corners = projected_corners.astype(int)


    # Step 5: Color the points based on depth and overlay them on the image
    cmap = plt.get_cmap("autumn")
    colors = (cmap(depths / np.max(depths))[:, :3] * 255).astype(np.uint8)

    image_with_bbox = image.copy()
    for (point, color) in zip(projected_corners, colors):
        x, y = point
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.circle(image_with_bbox, (x, y), 2, color_bgr, -1)

    image_path_with_bbox = os.path.join(output_dir, f"{camera_name}_with_bbox_points_in_process_img.png")
    cv2.imwrite(image_path_with_bbox, image_with_bbox)    

    # Draw the 3D bounding boxes on the image for each segment
    image_with_drawn_boxes = image.copy()

    # Iterate through the projected corners in chunks of 8
    for i in range(0, len(projected_corners), 8):
        bbox_corners = projected_corners[i:i+8]
        image_with_drawn_boxes = draw_3d_bounding_box(image_with_drawn_boxes, bbox_corners)

    # Save the image with 3D bounding boxes
    output_path_with_boxes = os.path.join(output_dir, f"{camera_name}_with_3d_bounding_boxes.png")
    cv2.imwrite(output_path_with_boxes, image_with_drawn_boxes)

# Function to convert an OrientedBoundingBox into a LineSet
def obb_to_lineset(obb):
    # Get the center and extent of the box
    center = obb.center
    extent = obb.extent
    
    # Define the corners of the box
    corners = np.array([
        [-extent[0], -extent[1], -extent[2]],
        [extent[0], -extent[1], -extent[2]],
        [extent[0], extent[1], -extent[2]],
        [-extent[0], extent[1], -extent[2]],
        [-extent[0], -extent[1], extent[2]],
        [extent[0], -extent[1], extent[2]],
        [extent[0], extent[1], extent[2]],
        [-extent[0], extent[1], extent[2]]
    ])
    
    # Rotate the corners based on the orientation of the box
    # For simplicity, assume no rotation here; you need to apply the rotation matrix if available
    rotated_corners = corners
    
    # Translate the corners to the box's center
    translated_corners = rotated_corners + center
    
    # Define the lines connecting the corners
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])
    
    # Create a LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(translated_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    return line_set
# %%
def project_bbox_to_other_cameras(boxes, new_camera_name, old_camera_name, image, camera_params):
 
    projected_corners_new, depths , _ = project_lidar_to_camera_v2(boxes, camera_params[new_camera_name])

    projected_corners_new = projected_corners_new.astype(int)
 

    return projected_corners_new



def filter_bboxes_proximity(flattened_bboxes, threshold=1.0):
    """Filters 3D bounding boxes to keep only those with lowest centroid norm 
    within proximity threshold, preserving flattened format.
    
    Args:
        flattened_bboxes: Input array of shape (N×8, 3) with 8 corners per box
        threshold: Minimum allowed distance between box centroids (meters)
    
    Returns:
        Filtered array in flattened format with guaranteed 1m separation
    """
    # Validate and reshape input
    if len(flattened_bboxes) % 8 != 0:
        raise ValueError("Input must contain complete boxes (8 corners each)")
    num_boxes = len(flattened_bboxes) // 8
    boxes = flattened_bboxes.reshape(num_boxes, 8, 3)

    #Find the min valued z coordinate of each box and subtract it from the z coordinates of all the points in the box
    min_z = np.min(boxes[:, :, 2], axis=1)
    boxes[:, :, 2] -= min_z[:, np.newaxis]
    boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, None)  # Ensure z is non-negative
    
    # Calculate centroids and their norms
    centroids = np.mean(boxes, axis=1)
    norms = np.linalg.norm(centroids, axis=1)
    
    # Find boxes violating proximity
    dist_matrix = cdist(centroids, centroids, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)
    close_pairs = np.column_stack(np.where(dist_matrix < threshold))
    
    # Determine boxes to remove (higher norm in each pair)
    to_remove = set()
    for i, j in close_pairs:
        if i not in to_remove and j not in to_remove:
            to_remove.add(i if norms[i] > norms[j] else j)
    
    # Filter and flatten
    keep_mask = np.array([i not in to_remove for i in range(num_boxes)])
    return boxes[keep_mask].reshape(-1, 3)


def linear_regression_model_all_lidar_points(depth_map, lidar_points, camera_name, image, camera_params):
    #Project the lidar points to the image plane
    projected_points, depths, _ = project_lidar_to_camera_v2(lidar_points, camera_params[camera_name])
    #Convert projected points to integers and apply boundary filtering
    projected_points = projected_points.astype(int)

    #Only keep points whose depth is less than 100
    # valid_mask1 = depths < 30
    # projected_points = projected_points[valid_mask1]
    # depths = depths[valid_mask1]

    valid_mask = (
        (projected_points[:, 0] >= 0)
        & (projected_points[:, 0] < image.shape[1])
        & (projected_points[:, 1] >= 0)
        & (projected_points[:, 1] < image.shape[0])
    )
    projected_points = projected_points[valid_mask]
    lidar_depths = depths[valid_mask]
    #Find correspondences with depth map
    DM_depths = []
    for point in projected_points:
        #Find the average depth of a 20 pixel square around the point
        #Create all the 2D points in the square
        x, y = point
        x_min = max(0, x - 10)
        x_max = min(image.shape[1], x + 10)
        y_min = max(0, y - 10)
        y_max = min(image.shape[0], y + 10)
        square_points = []
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                square_points.append((i, j))
        #Get the depth of the square points
        square_points = np.array(square_points)
        square_depths = []
        for point in square_points:
            x, y = point
            #Get the depth of the point
            square_depths.append(depth_map[y, x])
        #Get the average depth of the square points
        square_depths = np.array(square_depths)
        #Get the average depth of the square points
        avg_depth = np.mean(square_depths)
        #Add the average depth to the list
        DM_depths.append(avg_depth)


    DM_depths = np.array(DM_depths)
    #Calculate the linear regression model

    regression_model=LinearRegression()

    # print("LIN REG STARTED")
    print(camera_name)
    print(lidar_depths.shape)
    print(DM_depths.shape)
    regression_model.fit(DM_depths.reshape(-1, 1), lidar_depths)
    #Run the model on the depth map
    depth_map = depth_map.reshape(-1, 1)
    # print(depth_map.shape)
    # print(depth_map[0])
    # print(image.shape)
    refined_depth_map = regression_model.predict(depth_map)

    regressed_dm_depths = regression_model.predict(DM_depths.reshape(-1, 1))
    # print(depth_map[0])
    # print(depth_map.shape)
    refined_depth_map = refined_depth_map.reshape(image.shape[0], image.shape[1])

    print("Lidar depths: ", lidar_depths[int(len(lidar_depths)/2)],lidar_depths[int(len(lidar_depths)/2)+1], lidar_depths[int(len(lidar_depths)/2)+2])
    print("Depth map depths: ", DM_depths[0],DM_depths[1], DM_depths[2])
    print("Refined depths: ", regressed_dm_depths[0],regressed_dm_depths[1], regressed_dm_depths[2])

    #Calculate average error for linear regression
    print("Average error for linear regression: ", np.mean(np.abs(regressed_dm_depths - lidar_depths)))

    #Display the points and fitted line as a plot
    # plt.scatter(DM_depths, lidar_depths, color='blue', label='Data Points')
    # plt.plot(DM_depths, regression_model.predict(DM_depths.reshape(-1, 1)), color='red', label='Fitted Line')
    # plt.show()

    # print("LIN REG ENDED")
    #Return the depth map
    return refined_depth_map

def linear_regression_model_only_segmented_points(all_annotated_boxes_from_lidar, new_image_depth_map, combined_points, camera_name, new_image, camera_params):
    #Find combined_points that lie within each annotated 3D bounding box
    #For each annotated box, find the points that lie within it
    #Take average depth of the points that lie within the box
    #Also for all points within the box, project to masked image and get the average depth from the depth map

    #Use these correspondences to create a linear regression model
    
    #Implement it next
    all_lidar_depths=[]
    all_DM_depths=[]
    for index in range(0,len(all_annotated_boxes_from_lidar),8):

        box = all_annotated_boxes_from_lidar[index:index+8]
        # import pdb; pdb.set_trace()

        box = box.reshape(-1, 3)
        #Get the min and max points of the box
        min_point = np.min(box, axis=0)
        max_point = np.max(box, axis=0)
        #Get the points that lie within the box
        points_within_box = []
        for point in combined_points:
            if point[0] >= min_point[0] and point[0] <= max_point[0] and point[1] >= min_point[1] and point[1] <= max_point[1] and point[2] >= min_point[2] and point[2] <= max_point[2]:
                points_within_box.append(point)
        points_within_box = np.array(points_within_box)
        # pdb.set_trace()
        #Get the average depth of the points that lie within the box
        if len(points_within_box) == 0:
            print("No points within box")
            continue
        if len(points_within_box) == 1:
            avg_lidar_depth = points_within_box[0][0]
        else:
            avg_lidar_depth = np.mean(points_within_box[:, 0])
        #Get the average depth of the points that lie within the box in the depth map
        #Get the projected points of the points that lie within the box
        projected_points, depths, points_in_front_3d = project_lidar_to_camera_v2(points_within_box, camera_params[camera_name])
        #Convert projected points to integers and apply boundary filtering
        projected_points = projected_points.astype(int)
        valid_mask = (
            (projected_points[:, 0] >= 0)
            & (projected_points[:, 0] < new_image.shape[1])
            & (projected_points[:, 1] >= 0)
            & (projected_points[:, 1] < new_image.shape[0])
        )
        projected_points = projected_points[valid_mask]
        depths = depths[valid_mask]
        points_in_front_3d = points_in_front_3d[valid_mask]
        #Get the average depth of the points that lie within the box in the depth map
        DM_depths = []
        if len(projected_points)==0:
            continue
        for point in projected_points:
            # import pdb; pdb.set_trace()
            #Find the average depth of a 20 pixel square around the point
            #Create all the 2D points in the square
            x, y = point
            x_min = max(0, x - 10)
            x_max = min(new_image.shape[1], x + 10)
            y_min = max(0, y - 10)
            y_max = min(new_image.shape[0], y + 10)
            square_points = []
            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    square_points.append((i, j))
            #Get the depth of the square points
            square_points = np.array(square_points)
            square_depths = []
            for point in square_points:
                x, y = point
                #Get the depth of the point
                square_depths.append(new_image_depth_map[y, x])
            #Get the average depth of the square points
            square_depths = np.array(square_depths)
            #Get the average depth of the square points
            avg_depth = np.mean(square_depths)
            #Add the average depth to the list
            DM_depths.append(avg_depth)

        DM_depths = np.array(DM_depths)
        avg_dm_depth = np.mean(DM_depths)

        all_lidar_depths.append(avg_lidar_depth)
        all_DM_depths.append(avg_dm_depth)
        print("Average depth of points within box: ", avg_lidar_depth)
        print("Average depth of points within box in depth map: ", avg_dm_depth)

    # all_lidar_depths = np.array(all_lidar_depths)
    # all_DM_depths = np.array(all_DM_depths)

    # linear_regression_dm_depths[camera_name].extend(all_DM_depths)
    # linear_regression_lidar_depths[camera_name].extend(all_lidar_depths)

    # regression_model=LinearRegression()
    # regression_model.fit(all_DM_depths.reshape(-1, 1), all_lidar_depths.reshape(-1, 1))
    # #Run the model on the depth map
    # depth_map = new_image_depth_map.reshape(-1, 1)
    # # print(depth_map.shape)
    # # print(depth_map[0])
    # # print(image.shape)
    # refined_depth_map = regression_model.predict(depth_map)
    # #Run the model on the depth map
    # regressed_dm_depths = regression_model.predict(all_DM_depths.reshape(-1, 1))
    # # print(depth_map[0])
    # refined_depth_map = refined_depth_map.reshape(new_image.shape[0], new_image.shape[1])
    # print("Lidar depths: ", all_lidar_depths)
    # print("Depth map depths: ", all_DM_depths)
    # print("Refined depths: ", regressed_dm_depths)
    # #Calculate average error for linear regression
    # print("Average error for linear regression: ", np.mean(np.abs(regressed_dm_depths - all_lidar_depths)))
    # #Display the points and fitted line as a plot
    # # plt.scatter(DM_depths, lidar_depths, color='blue', label='Data Points')
    # # plt.plot(DM_depths, regression_model.predict(DM_depths.reshape(-1, 1)), color='red', label='Fitted Line')
    # # plt.show()
    # # print("LIN REG ENDED")
    # #Return the refined depth map
    #return refined_depth_map

    return linear_regression_model(new_image_depth_map, camera_name)
    

import time


def segment_bbox_centers_and_classes(segments, segment_classes):
    """
    Returns:
      seg_centers: list of (cx, cy) for each 2D polygon bbox center
      seg_classes: np.array of class ids per segment
      seg_bboxes: list of (x_min, y_min, x_max, y_max) for each segment
    """
    seg_centers = []
    seg_bboxes = []
    for seg in segments:
        seg_arr = np.array(seg, dtype=np.int32).reshape((-1, 2))
        x_min, y_min = np.min(seg_arr, axis=0)
        x_max, y_max = np.max(seg_arr, axis=0)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        seg_centers.append((cx, cy))
        seg_bboxes.append((x_min, y_min, x_max, y_max))
    return seg_centers, np.array(segment_classes), seg_bboxes

def calculate_2d_bbox_iou(bbox1, bbox2):
    """Calculate IoU between two 2D bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

import numpy as np

def _pick_K(cam):
    if 'new_projection' in cam and cam['new_projection'] is not None:
        fx, fy, cx, cy = cam['new_projection']
    else:
        fx, fy, cx, cy = cam['projection']
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, fx, fy, cx, cy

def _proj_center_to_image(center_v, K, R, t):
    Xc = R @ center_v.reshape(3, 1) + t.reshape(3, 1)
    Xc, Yc, Zc = Xc[0,0], Xc[1,0], Xc[2,0]
    if Zc <= 1e-9:
        return None, (Xc, Yc, Zc)
    u = K[0,0]*Xc/Zc + K[0,2]
    v = K[1,1]*Yc/Zc + K[1,2]
    return (u, v), (Xc, Yc, Zc)

def _box_center_and_dims(corners8):
    mn = corners8.min(axis=0)
    mx = corners8.max(axis=0)
    center = 0.5*(mn + mx)
    L = mx[0] - mn[0]
    W = mx[1] - mn[1]
    H = mx[2] - mn[2]
    return center, (L, W, H)

import numpy as np

def snap_vehicle_boxes_to_segments_center_simple(
    vehicle_corners_flat,      # (M*8,3) in vehicle frame
    seg_centers,               # list of (u,v) centers from 2D segments
    cam_params,                # e.g., camera_params["FRONT_ZOOMED_IN_CAMERA"]
    max_pixel_dist=50          # <-- new arg: max pixel distance allowed
):
    if vehicle_corners_flat.size == 0 or len(seg_centers) == 0:
        return vehicle_corners_flat

    fx, fy, cx, cy = cam_params["projection"]  # intrinsics
    R = quaternion_to_rotation_matrix(cam_params["rotation"])
    t = np.array(cam_params["translation"], dtype=np.float64).reshape(3, 1)

    seg_centers_np = np.array(seg_centers, dtype=np.float64)
    num_boxes = vehicle_corners_flat.shape[0] // 8
    boxes = vehicle_corners_flat.reshape(num_boxes, 8, 3).copy()

    for b in range(num_boxes):
        corners = boxes[b]
        center_v = corners.mean(axis=0)
        center_c = R @ center_v.reshape(3,1) + t
        Xc, Yc, Zc = center_c.flatten()
        if Zc <= 1e-6:
            continue

        u = fx * Xc/Zc + cx
        v = fy * Yc/Zc + cy

        dists = np.linalg.norm(seg_centers_np - np.array([u, v]), axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > max_pixel_dist:
            continue  # skip snap if nearest center too far

        u_tgt, v_tgt = seg_centers_np[idx]
        du = u_tgt - u
        dv = v_tgt - v
        dx = (du / fx) * Zc
        dy = (dv / fy) * Zc

        boxes[b, :, 0] += dx
        boxes[b, :, 1] += dy

    return boxes.reshape(-1, 3)


def _snap_priority_ZI_then_ZO(
    boxes_flat,
    zi_segments, zi_classes,
    zo_segments, zo_classes,
    camera_params
):
    if boxes_flat.size == 0:
        return boxes_flat

    boxes = boxes_flat.reshape(-1, 8, 3).copy()
    total_boxes = boxes.shape[0]

    zi_snapped = 0
    zo_snapped = 0

    # Step 1: ZI snap
    moved_mask = np.zeros(total_boxes, dtype=bool)
    if zi_segments is not None and len(zi_segments) > 0 and "FRONT_ZOOMED_IN_CAMERA" in camera_params:
        zi_centers, _, _ = segment_bbox_centers_and_classes(zi_segments, zi_classes)
        if len(zi_centers) > 0:
            before = boxes.copy().reshape(-1, 3)
            snapped_flat = snap_vehicle_boxes_to_segments_center_simple(
                boxes.reshape(-1, 3), zi_centers, camera_params["FRONT_ZOOMED_IN_CAMERA"],
                max_pixel_dist=20   # strict threshold
            )
            after = snapped_flat.reshape(-1, 8, 3)
            moved_mask = (np.abs(after - boxes) > 1e-9).any(axis=(1, 2))
            zi_snapped = int(moved_mask.sum())
            boxes = after

    # Step 2: ZO snap for leftover boxes
    need_snap_mask = ~moved_mask
    if np.any(need_snap_mask) and zo_segments is not None and len(zo_segments) > 0 and "FRONT_ZOOMED_OUT_CAMERA" in camera_params:
        zo_centers, _, _ = segment_bbox_centers_and_classes(zo_segments, zo_classes)
        if len(zo_centers) > 0:
            subset = boxes[need_snap_mask].reshape(-1, 3)
            snapped_subset = snap_vehicle_boxes_to_segments_center_simple(
                subset, zo_centers, camera_params["FRONT_ZOOMED_OUT_CAMERA"],
                max_pixel_dist=50  # loose threshold
            ).reshape(-1, 8, 3)
            zo_snapped = need_snap_mask.sum()
            boxes[need_snap_mask] = snapped_subset

    # print(f"[Snap] ZI snapped {zi_snapped}/{total_boxes}, ZO snapped {zo_snapped}/{total_boxes}")

    return boxes.reshape(-1, 3)


def save_segment_overlay(image, segments, out_path, alpha=0.35, edge_thickness=2):
    """Draw filled polygons with transparency + thin edges and save."""
    if not segments:
        return
    overlay = image.copy()
    for seg in segments:
        pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
        color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=edge_thickness, lineType=cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, blended)


# === paste below your existing helpers in annotation_v13_helper.py ===
import open3d as o3d
from math import atan2, cos, sin
from datetime import datetime
from glob import glob


def _to_homo(X): return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
def _apply_T(X, T): return (T @ _to_homo(X).T).T[:, :3]

# def _list_ts_in_folder(folder_path):
#     items = sorted(os.listdir(folder_path))
#     ts = set()
#     for f in items:
#         if 'lidar0' in f and f.endswith('.bin'):
#             ts.add(f.split('_')[0])
#     return sorted(ts)

def _list_ts_in_folder(folder_path):
    # one-shot per folder; subsequent calls are O(1)
    import os
    if not hasattr(_list_ts_in_folder, "_cache"):
        _list_ts_in_folder._cache = {}  # {folder_path: [ts,...]}
    cache = _list_ts_in_folder._cache

    ts_list = cache.get(folder_path)
    if ts_list is not None:
        return ts_list

    items = sorted(os.listdir(folder_path))
    ts = set()
    for f in items:
        if 'lidar0' in f and f.endswith('.bin'):
            ts.add(f.split('_')[0])
    ts_list = sorted(ts)
    cache[folder_path] = ts_list
    return ts_list

def _gather_lidar_files(folder_path, ts):
    d = {f'lidar{i}': None for i in range(6)}
    for i in range(6):
        hits = glob(os.path.join(folder_path, f"{ts}*lidar{i}*.bin"))
        if hits: d[f'lidar{i}'] = hits[0]
    return d

def _estimate_yaw_rate(times_sorted, states_sorted, query_dt):
    # finite-difference on yaw using neighbors around query time
    i = bisect_left(times_sorted, query_dt)
    def yaw_of(j): return _yaw_from_state(states_sorted[j])
    if 0 < i < len(times_sorted):
        t0, t1 = times_sorted[i-1], times_sorted[i]
        y0, y1 = yaw_of(i-1), yaw_of(i)
        dt = (t1 - t0).total_seconds()
        if dt > 1e-3:
            # unwrap small jump
            dy = np.arctan2(np.sin(y1 - y0), np.cos(y1 - y0))
            return dy / dt
    return 0.0
# def _load_cloud_vehicle_frame(folder_path, ts, extrinsics, speed_mps, right_lidar_flag,
#                               times_sorted=None, states_sorted=None):
#     files = _gather_lidar_files(folder_path, ts)
#     scan_period_s = 0.1
#     cfg = {'lidar0': (15, 45), 'lidar1': (45, 90), 'lidar2': (-45, 90),
#            'lidar3': (180, 90), 'lidar4': (-180, 90), 'lidar5': (360, 90)}
#     order = ['lidar0','lidar1','lidar2','lidar3','lidar5'] + (['lidar4'] if right_lidar_flag else [])

#     # NEW: estimate yaw rate at this timestamp
#     yaw_rate = 0.0
#     if times_sorted is not None and states_sorted is not None:
#         yaw_rate = _estimate_yaw_rate(times_sorted, states_sorted, _parse_fileprefix_ts(ts))

#     chunks = []
#     for key in order:
#         p = files.get(key)
#         if not p or key not in extrinsics: 
#             continue
#         th, off = cfg[key]
#         raw = load_lidar_raw(
#             p, intensity_threshold=0.0, velocity_mps=speed_mps,
#             scan_duration=scan_period_s, theta_deg=th, offset=off,
#             yaw_rate_rad_s=yaw_rate                        # NEW
#         )
#         pts_v = transform_to_vehicle_frame(raw, extrinsics[key])
#         chunks.append(pts_v)
#     if not chunks:
#         return np.empty((0,3), dtype=np.float32)
#     cloud = np.vstack(chunks)
#     x, y, z = cloud[:,0], cloud[:,1], cloud[:,2]
#     m = (~((x >= -1.0) & (x <= 4.0) & (y >= -1) & (y <= 1))) & (z > 0.2)
#     return cloud[m]

# def _load_cloud_vehicle_frame(folder_path, ts, extrinsics, speed_mps, right_lidar_flag,
#                               times_sorted=None, states_sorted=None):
#     import time
#     import numpy as np

#     def ms(sec): return int(round(sec * 1000))

#     t_total0 = time.perf_counter()

#     t0 = time.perf_counter()
#     files = _gather_lidar_files(folder_path, ts)
#     t_gather = time.perf_counter() - t0

#     scan_period_s = 0.1
#     cfg = {'lidar0': (15, 45), 'lidar1': (45, 90), 'lidar2': (-45, 90),
#            'lidar3': (180, 90), 'lidar4': (-180, 90), 'lidar5': (360, 90)}
#     order = ['lidar0','lidar1','lidar2','lidar3','lidar5'] + (['lidar4'] if right_lidar_flag else [])

#     # estimate yaw rate at this timestamp
#     t0 = time.perf_counter()
#     yaw_rate = 0.0
#     if times_sorted is not None and states_sorted is not None:
#         yaw_rate = _estimate_yaw_rate(times_sorted, states_sorted, _parse_fileprefix_ts(ts))
#     t_yaw = time.perf_counter() - t0

#     chunks = []
#     per_sensor_logs = []
#     t_load_sum = 0.0
#     t_xform_sum = 0.0

#     for key in order:
#         t_loop0 = time.perf_counter()
#         p = files.get(key)
#         if (not p) or (key not in extrinsics):
#             per_sensor_logs.append(
#                 f"[pcd] {key}: present={bool(p)} extrinsics={key in extrinsics} skip"
#             )
#             continue

#         th, off = cfg[key]

#         # load_lidar_raw timing
#         t0 = time.perf_counter()
#         raw = load_lidar_raw(
#             p, intensity_threshold=0.0, velocity_mps=speed_mps,
#             scan_duration=scan_period_s, theta_deg=th, offset=off,
#             yaw_rate_rad_s=yaw_rate
#         )
#         t_load = time.perf_counter() - t0
#         t_load_sum += t_load

#         # transform_to_vehicle_frame timing
#         t0 = time.perf_counter()
#         pts_v = transform_to_vehicle_frame(raw, extrinsics[key])
#         t_xform = time.perf_counter() - t0
#         t_xform_sum += t_xform

#         chunks.append(pts_v)
#         per_sensor_logs.append(
#             f"[pcd] {key}: file={p is not None} raw_pts={raw.shape[0] if isinstance(raw, np.ndarray) else 0} "
#             f"load={ms(t_load)}ms xform={ms(t_xform)}ms total_loop={ms(time.perf_counter()-t_loop0)}ms"
#         )

#     if not chunks:
#         t_total = time.perf_counter() - t_total0
#         print(
#             f"[pcd] ts={ts} gather={ms(t_gather)}ms yaw_est={ms(t_yaw)}ms "
#             f"loads_sum={ms(t_load_sum)}ms xforms_sum={ms(t_xform_sum)}ms stack=0ms filter=0ms total={ms(t_total)}ms "
#             f"points=0"
#         )
#         for line in per_sensor_logs:
#             print(line)
#         return np.empty((0,3), dtype=np.float32)

#     # stack timing
#     t0 = time.perf_counter()
#     cloud = np.vstack(chunks)
#     t_stack = time.perf_counter() - t0

#     # filter timing
#     t0 = time.perf_counter()
#     x, y, z = cloud[:,0], cloud[:,1], cloud[:,2]
#     m = (~((x >= -1.0) & (x <= 4.0) & (y >= -1) & (y <= 1))) & (z > 0.2)
#     out = cloud[m]
#     t_filter = time.perf_counter() - t0

#     t_total = time.perf_counter() - t_total0

#     print(
#         f"[pcd] ts={ts} gather={ms(t_gather)}ms yaw_est={ms(t_yaw)}ms "
#         f"loads_sum={ms(t_load_sum)}ms xforms_sum={ms(t_xform_sum)}ms "
#         f"stack={ms(t_stack)}ms filter={ms(t_filter)}ms total={ms(t_total)}ms "
#         f"in_pts={cloud.shape[0]} out_pts={out.shape[0]}"
#     )
#     for line in per_sensor_logs:
#         print(line)

#     return out

# def _load_cloud_vehicle_frame(folder_path, ts, extrinsics, speed_mps, right_lidar_flag,
#                               times_sorted=None, states_sorted=None):
#     """
#     Same outputs/logic as before, but with a per-folder index cache so we don't
#     re-scan the directory or glob on every call. This removes the big 'gather=XXXms' cost.
#     Also keeps the timing prints you liked.
#     """
#     import os, time
#     import numpy as np

#     def ms(sec): return int(round(sec * 1000))

#     # --- simple per-folder cache living on the function object ---
#     if not hasattr(_load_cloud_vehicle_frame, "_folder_index"):
#         _load_cloud_vehicle_frame._folder_index = {}  # {folder_path: {ts: {lidar0: path, ...}}}

#     t_total0 = time.perf_counter()

#     # Build (or reuse) an index of {ts: {lidarX: path}}
#     t0 = time.perf_counter()
#     folder_idx = _load_cloud_vehicle_frame._folder_index.get(folder_path)
#     if folder_idx is None:
#         # First time seeing this folder: do one pass over the dir and index everything
#         folder_idx = {}
#         try:
#             for fname in os.listdir(folder_path):
#                 # we only care about lidar*.bin files
#                 if not fname.endswith(".bin") or "lidar" not in fname:
#                     continue
#                 # expected pattern "<ts>_..._lidarK....bin"
#                 # your earlier code uses f.split('_')[0] as ts
#                 ts_key = fname.split('_')[0]
#                 # find which lidar id this is
#                 if "lidar0" in fname: lidkey = "lidar0"
#                 elif "lidar1" in fname: lidkey = "lidar1"
#                 elif "lidar2" in fname: lidkey = "lidar2"
#                 elif "lidar3" in fname: lidkey = "lidar3"
#                 elif "lidar4" in fname: lidkey = "lidar4"
#                 elif "lidar5" in fname: lidkey = "lidar5"
#                 else:
#                     continue
#                 d = folder_idx.setdefault(ts_key, {})
#                 d[lidkey] = os.path.join(folder_path, fname)
#         except FileNotFoundError:
#             folder_idx = {}
#         _load_cloud_vehicle_frame._folder_index[folder_path] = folder_idx

#     # now "gather" is just a dict lookup
#     files = folder_idx.get(ts, {})  # same shape as _gather_lidar_files(...) output
#     t_gather = time.perf_counter() - t0

#     scan_period_s = 0.1
#     cfg = {'lidar0': (15, 45), 'lidar1': (45, 90), 'lidar2': (-45, 90),
#            'lidar3': (180, 90), 'lidar4': (-180, 90), 'lidar5': (360, 90)}
#     order = ['lidar0','lidar1','lidar2','lidar3','lidar5'] + (['lidar4'] if right_lidar_flag else [])

#     # estimate yaw rate at this timestamp
#     t0 = time.perf_counter()
#     yaw_rate = 0.0
#     if times_sorted is not None and states_sorted is not None:
#         yaw_rate = _estimate_yaw_rate(times_sorted, states_sorted, _parse_fileprefix_ts(ts))
#     t_yaw = time.perf_counter() - t0

#     chunks = []
#     per_sensor_logs = []
#     t_load_sum = 0.0
#     t_xform_sum = 0.0

#     for key in order:
#         t_loop0 = time.perf_counter()
#         p = files.get(key)
#         if (not p) or (key not in extrinsics):
#             per_sensor_logs.append(
#                 f"[pcd] {key}: present={bool(p)} extrinsics={key in extrinsics} skip"
#             )
#             continue

#         th, off = cfg[key]

#         # load_lidar_raw timing
#         t0 = time.perf_counter()
#         raw = load_lidar_raw(
#             p, intensity_threshold=0.0, velocity_mps=speed_mps,
#             scan_duration=scan_period_s, theta_deg=th, offset=off,
#             yaw_rate_rad_s=yaw_rate
#         )
#         t_load = time.perf_counter() - t0
#         t_load_sum += t_load

#         # transform_to_vehicle_frame timing
#         t0 = time.perf_counter()
#         pts_v = transform_to_vehicle_frame(raw, extrinsics[key])
#         t_xform = time.perf_counter() - t0
#         t_xform_sum += t_xform

#         chunks.append(pts_v)
#         per_sensor_logs.append(
#             f"[pcd] {key}: file=True raw_pts={raw.shape[0] if isinstance(raw, np.ndarray) else 0} "
#             f"load={ms(t_load)}ms xform={ms(t_xform)}ms total_loop={ms(time.perf_counter()-t_loop0)}ms"
#         )

#     if not chunks:
#         t_total = time.perf_counter() - t_total0
#         print(
#             f"[pcd] ts={ts} gather={ms(t_gather)}ms yaw_est={ms(t_yaw)}ms "
#             f"loads_sum={ms(t_load_sum)}ms xforms_sum={ms(t_xform_sum)}ms stack=0ms filter=0ms total={ms(t_total)}ms "
#             f"points=0"
#         )
#         for line in per_sensor_logs:
#             print(line)
#         return np.empty((0,3), dtype=np.float32)

#     # stack timing
#     t0 = time.perf_counter()
#     cloud = np.vstack(chunks)
#     t_stack = time.perf_counter() - t0

#     # filter timing
#     t0 = time.perf_counter()
#     x, y, z = cloud[:,0], cloud[:,1], cloud[:,2]
#     m = (~((x >= -1.0) & (x <= 4.0) & (y >= -1) & (y <= 1))) & (z > 0.2)
#     out = cloud[m]
#     t_filter = time.perf_counter() - t0

#     t_total = time.perf_counter() - t_total0

#     print(
#         f"[pcd] ts={ts} gather={ms(t_gather)}ms yaw_est={ms(t_yaw)}ms "
#         f"loads_sum={ms(t_load_sum)}ms xforms_sum={ms(t_xform_sum)}ms "
#         f"stack={ms(t_stack)}ms filter={ms(t_filter)}ms total={ms(t_total)}ms "
#         f"in_pts={cloud.shape[0]} out_pts={out.shape[0]}"
#     )
#     for line in per_sensor_logs:
#         print(line)

#     return out

def _load_cloud_vehicle_frame(folder_path, ts, extrinsics, speed_mps, right_lidar_flag,
                              times_sorted=None, states_sorted=None):
    """
    Same outputs/logic as before, but with a per-folder index cache so we don't
    re-scan the directory or glob on every call. Timing code and prints removed.
    """
    import os
    import numpy as np

    # --- simple per-folder cache living on the function object ---
    if not hasattr(_load_cloud_vehicle_frame, "_folder_index"):
        _load_cloud_vehicle_frame._folder_index = {}  # {folder_path: {ts: {lidar0: path, ...}}}

    # Build (or reuse) an index of {ts: {lidarX: path}}
    folder_idx = _load_cloud_vehicle_frame._folder_index.get(folder_path)
    if folder_idx is None:
        folder_idx = {}
        try:
            for fname in os.listdir(folder_path):
                if not fname.endswith(".bin") or "lidar" not in fname:
                    continue
                ts_key = fname.split('_')[0]
                if "lidar0" in fname: lidkey = "lidar0"
                elif "lidar1" in fname: lidkey = "lidar1"
                elif "lidar2" in fname: lidkey = "lidar2"
                elif "lidar3" in fname: lidkey = "lidar3"
                elif "lidar4" in fname: lidkey = "lidar4"
                elif "lidar5" in fname: lidkey = "lidar5"
                else:
                    continue
                d = folder_idx.setdefault(ts_key, {})
                d[lidkey] = os.path.join(folder_path, fname)
        except FileNotFoundError:
            folder_idx = {}
        _load_cloud_vehicle_frame._folder_index[folder_path] = folder_idx

    files = folder_idx.get(ts, {})

    scan_period_s = 0.1
    cfg = {'lidar0': (15, 45), 'lidar1': (45, 90), 'lidar2': (-45, 90),
           'lidar3': (180, 90), 'lidar4': (-180, 90), 'lidar5': (360, 90)}
    order = ['lidar0','lidar1','lidar2','lidar3','lidar5'] + (['lidar4'] if right_lidar_flag else [])

    # estimate yaw rate at this timestamp
    yaw_rate = 0.0
    if times_sorted is not None and states_sorted is not None:
        yaw_rate = _estimate_yaw_rate(times_sorted, states_sorted, _parse_fileprefix_ts(ts))

    chunks = []
    for key in order:
        p = files.get(key)
        if (not p) or (key not in extrinsics):
            continue

        th, off = cfg[key]

        raw = load_lidar_raw(
            p, intensity_threshold=0.0, velocity_mps=speed_mps,
            scan_duration=scan_period_s, theta_deg=th, offset=off,
            yaw_rate_rad_s=yaw_rate
        )
        pts_v = transform_to_vehicle_frame(raw, extrinsics[key])
        chunks.append(pts_v)

    if not chunks:
        return np.empty((0,3), dtype=np.float32)

    cloud = np.vstack(chunks)
    x, y, z = cloud[:,0], cloud[:,1], cloud[:,2]
    m = (~((x >= -1.0) & (x <= 4.0) & (y >= -1) & (y <= 1))) & (z > 0.2)
    out = cloud[m]

    return out


from bisect import bisect_left
from datetime import datetime
from math import atan2, cos, sin

def _parse_iso_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)

def _parse_fileprefix_ts(ts_prefix: str) -> datetime:
    """
    Convert file prefix like '120241028T205421.915256' -> '2024-10-28T20:54:21.915256'
    (drops the leading '1')
    """
    s = ts_prefix
    if s and s[0] == '1':
        s = s[1:]
    date, time = s.split('T', 1)          # '20241028', '205421.915256'
    y, m, d = date[:4], date[4:6], date[6:8]
    hh, mm, rest = time[:2], time[2:4], time[4:]
    iso = f"{y}-{m}-{d}T{hh}:{mm}:{rest}"
    return datetime.fromisoformat(iso)

# def _build_state_index(states_list):
#     """
#     states_list: a LIST of dicts like:
#       { "vehicle_timestamp": "...", "pose": {...}, "velocity": {...}, "speed_mps": ... }
#     Returns (sorted_times, aligned_states)
#     """
#     entries = []
#     for s in states_list:
#         ts = s.get('vehicle_timestamp') or s.get('timestamp')
#         if not ts:
#             continue
#         # strictly ignore 'log_timestamp'
#         entries.append((_parse_iso_ts(ts), s))
#     entries.sort(key=lambda x: x[0])
#     times = [t for t, _ in entries]
#     states = [s for _, s in entries]
#     return times, states

def _build_state_index(states_list):
    # Fast path: reuse if we've seen an identical list object
    if not hasattr(_build_state_index, "_cache"):
        _build_state_index._cache = {}  # {id(states_list): (times, states, size_marker)}
    cache = _build_state_index._cache

    marker = (id(states_list), len(states_list))
    hit = cache.get(marker)
    if hit is not None:
        return hit[0], hit[1]

    # original logic
    entries = []
    for s in states_list:
        ts = s.get('vehicle_timestamp') or s.get('timestamp')
        if not ts:  # strictly ignore 'log_timestamp'
            continue
        entries.append((_parse_iso_ts(ts), s))
    entries.sort(key=lambda x: x[0])
    times = [t for t, _ in entries]
    states = [s for _, s in entries]

    cache[marker] = (times, states, marker[1])
    return times, states

def _nearest_state(times_sorted, states_sorted, query_dt: datetime, max_gap_s=2.0):
    """Pick nearest state by time within ±max_gap_s seconds, else None."""
    if not times_sorted:
        return None
    i = bisect_left(times_sorted, query_dt)
    cand = []
    if i < len(times_sorted):
        cand.append((abs((times_sorted[i] - query_dt).total_seconds()), i))
    if i > 0:
        cand.append((abs((times_sorted[i-1] - query_dt).total_seconds()), i-1))
    if not cand:
        return None
    gap, idx = min(cand, key=lambda x: x[0])
    return states_sorted[idx] if gap <= max_gap_s else None

def _yaw_from_state(state):
    """
    Prefer explicit heading if present; else if moving use velocity; else fallback to pose.rot1/rot2.
    Ensures the returned yaw is in radians.
    """
    import numpy as np

    # 1) explicit heading
    heading = state.get('heading') or state.get('pose', {}).get('heading')
    if heading is not None:
        h = float(heading)
        if abs(h) > 2*np.pi + 1e-3:  # looks like degrees
            h = np.deg2rad(h)
        return h

    # 2) velocity if moving
    vel = state.get('velocity', {})
    vx, vy = vel.get('x'), vel.get('y')
    spd = float(state.get('speed_mps', 0.0) or 0.0)
    if vx is not None and vy is not None and spd > 1.0:
        return np.arctan2(float(vy), float(vx))

    # 3) fallback to pose.rot1/rot2 (forward vector)
    pose = state.get('pose', {})
    r1, r2 = float(pose.get('rot1', 1.0)), float(pose.get('rot2', 0.0))
    return np.arctan2(r2, r1)

def _T_w_from_state(state):
    """World←Vehicle SE(3) from UTM pose + yaw (Z-up; planar yaw)."""
    p = state['pose']
    x, y, z = float(p['x']), float(p['y']), float(p['z'])
    yaw = _yaw_from_state(state)
    cy, sy = cos(yaw), sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = np.array([[ cy, -sy, 0.0],
                         [ sy,  cy, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T

def _o3d_icp_refine(source_np, target_np, init_T,
                    voxel=0.30,        # downsample for speed/stability
                    max_corr=10,     # meters; tune to your scene scale
                    point_to_plane=True,
                    max_iter=50):
    """
    Refine source → target alignment using Open3D ICP.

    Args:
      source_np: (N,3) source points in their *own* frame (NOT pre-transformed)
      target_np: (M,3) target points in ref frame
      init_T:    (4,4) initial transform that maps source→target (e.g., T_Vref_Vn)
      voxel:     downsample size for ICP (meters)
      max_corr:  correspondence distance (meters)
      point_to_plane: use point-to-plane ICP when True; else point-to-point
      max_iter:  ICP iterations

    Returns:
      T: (4,4) refined transform
      fitness, rmse: ICP stats
    """
    if source_np.size == 0 or target_np.size == 0:
        return init_T, 0.0, np.inf

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_np.astype(np.float64)))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_np.astype(np.float64)))

    if voxel and voxel > 0:
        src = src.voxel_down_sample(voxel)
        tgt = tgt.voxel_down_sample(voxel)

    if point_to_plane:
        # normals only needed on target for point-to-plane
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.5*voxel, max_nn=50))
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_corr,
        init_T,
        estimation,
        criteria
    )
    return reg.transformation, reg.fitness, reg.inlier_rmse


def visualize_fused_cloud(states_dict, extrinsics_dict, folder_path,
                          window=0.5, voxel=None, has_right_lidar=True, ref_ts=None):
    # 0) list timestamps
    all_ts = _list_ts_in_folder(folder_path)
    if not all_ts:
        print("[viz] No lidar timestamps in folder.")
        return

    # 1) pick deterministic reference ts
    chosen_ref_ts = ref_ts if ref_ts is not None else all_ts[len(all_ts)//2]

    # === NEW: build state index once; get ref_state + speed ===
    state_times, state_list = _build_state_index(states_dict)            # <<< NEW
    ref_state = _nearest_state(state_times, state_list,                  # <<< NEW
                               _parse_fileprefix_ts(chosen_ref_ts),
                               max_gap_s=2.0)
    ref_speed = float(ref_state.get('speed_mps', 0.0)) if ref_state else 0.0  # <<< NEW

    # >>> if no temporal fusion requested, load ref with speed/yaw-rate aware deskew
    if window is None or window <= 0.0:
        cloud_ref0 = _load_cloud_vehicle_frame(
            folder_path, chosen_ref_ts, extrinsics_dict,
            speed_mps=ref_speed,                                 # <<< CHANGED
            right_lidar_flag=has_right_lidar,
            times_sorted=state_times, states_sorted=state_list    # <<< CHANGED
        )
        if cloud_ref0.size == 0:
            print(f"[viz] Empty ref cloud for {chosen_ref_ts}.")
            return
        if voxel:
            p0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_ref0))
            p0 = p0.voxel_down_sample(voxel_size=voxel)
            cloud_ref0 = np.asarray(p0.points)


        print(f"[viz] ref_ts={chosen_ref_ts} neighbors=0 points={cloud_ref0.shape[0]}")
        print("AABB:", cloud_ref0.min(axis=0), cloud_ref0.max(axis=0))
        print("checksum:", np.round(cloud_ref0[:10].sum(), 6), cloud_ref0.shape)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_ref0))
        o3d.visualization.draw_geometries([pcd])
        return
    # <<< END no-fusion branch

    # 2) load the ref cloud in YOUR vehicle frame (now speed/yaw-rate aware)
    cloud_ref0 = _load_cloud_vehicle_frame(
        folder_path, chosen_ref_ts, extrinsics_dict,
        speed_mps=ref_speed,                                   # <<< CHANGED
        right_lidar_flag=has_right_lidar,
        times_sorted=state_times, states_sorted=state_list      # <<< CHANGED
    )
    if cloud_ref0.size == 0:
        print(f"[viz] Empty ref cloud for {chosen_ref_ts}.")
        return
    if voxel:
        p0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_ref0))
        p0 = p0.voxel_down_sample(voxel_size=voxel)
        cloud_ref0 = np.asarray(p0.points)

    icp_target_np = cloud_ref0.copy() 

    # 3) find neighbors (unchanged)
    MAX_NEIGHBORS = 2
    ref_state = _nearest_state(state_times, state_list, _parse_fileprefix_ts(chosen_ref_ts), max_gap_s=2.0)
    neighbors = []
    if ref_state is not None and window > 0.0:
        t0 = _parse_iso_ts(ref_state['vehicle_timestamp'])
        cand = []
        for ts in all_ts:
            st = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ts), max_gap_s=2.0)
            if st is None:
                continue
            dt = abs((_parse_iso_ts(st['vehicle_timestamp']) - t0).total_seconds())
            if 0.0 < dt <= window:
                cand.append((dt, ts))

        # sort by |dt| and pick min(2, len(cand))
        cand.sort(key=lambda x: x[0])
        print(f"[viz] Found {len(cand)} neighbors within ±{window}s of {chosen_ref_ts}.")
        neighbors = [ts for _, ts in cand[:MAX_NEIGHBORS]]

    neighbors.sort()


    # 4) 0-neighbor early exit (unchanged logging)
    if len(neighbors) == 0:
        print(f"[viz] ref_ts={chosen_ref_ts} neighbors=0 points={cloud_ref0.shape[0]}")
        print("AABB:", cloud_ref0.min(axis=0), cloud_ref0.max(axis=0))
        print("checksum:", np.round(cloud_ref0[:10].sum(), 6), cloud_ref0.shape)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_ref0))
        o3d.visualization.draw_geometries([pcd])
        return

    # 5) fuse neighbors -> SAME ref vehicle frame (keep your colorization if you added it)
    T_WV_ref = _T_w_from_state(ref_state)
    T_VW_ref = np.linalg.inv(T_WV_ref)

    pcd_list = []
    pcd_ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_ref0))
    pcd_ref.paint_uniform_color([0.7, 0.7, 0.7])
    pcd_list.append(pcd_ref)

    cmap = plt.get_cmap('tab10' if len(neighbors) <= 10 else 'hsv')
    for k, ts in enumerate(neighbors):
        st = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ts), max_gap_s=2.0)
        if st is None:
            continue
        spd = float(st.get('speed_mps', 0.0))                     # <<< NEW
        cloud_v_n = _load_cloud_vehicle_frame(
            folder_path, ts, extrinsics_dict,
            speed_mps=spd,                                        # <<< CHANGED
            right_lidar_flag=has_right_lidar,
            times_sorted=state_times, states_sorted=state_list     # <<< CHANGED
        )
        if cloud_v_n.size == 0:
            continue

        T_WV_n = _T_w_from_state(st)
        T_Vref_Vn = T_VW_ref @ T_WV_n

        # --- NEW: ICP refinement (minimal addition) ---
        # Use the raw neighbor cloud in its own vehicle frame (cloud_v_n) + initial T
        T_icp, fit, rmse = _o3d_icp_refine(
            source_np=cloud_v_n,
            target_np=icp_target_np,
            init_T=T_Vref_Vn,
            voxel=0.3,          # try 0.2–0.5 depending on density
            max_corr=10,       # try 1.0–2.0 m depending on motion/overlap
            point_to_plane=True, # usually better on road/planar scenes
            max_iter=100
        )

        # If ICP goes off the rails (very low overlap), fall back to the pose-only T
        if fit < 0.10 or not np.isfinite(rmse):
            T_final = T_Vref_Vn
            # optional: print(f"[ICP] low fitness ({fit:.3f}); using init transform.")
        else:
            T_final = T_icp
            # optional: print(f"[ICP] fitness={fit:.3f} rmse={rmse:.3f}")

        cloud_ref_n = _apply_T(cloud_v_n, T_final)
        # --- END NEW ---


        if voxel:
            ptmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_ref_n))
            ptmp = ptmp.voxel_down_sample(voxel_size=voxel)
            cloud_ref_n = np.asarray(ptmp.points)

        pcd_n = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_ref_n))
        c = cmap(k / max(1, len(neighbors)))[:3]
        pcd_n.paint_uniform_color([float(c[0]), float(c[1]), float(c[2])])
        pcd_list.append(pcd_n)

    total_pts = sum(len(np.asarray(p.points)) for p in pcd_list)
    print(f"[viz] ref_ts={chosen_ref_ts} neighbors={len(neighbors)} points={total_pts}")
    o3d.visualization.draw_geometries(pcd_list)

# def get_fused_cloud_for_ts(folder_path, ref_ts, extrinsics, sensor_list,
#                            vehicle_states_list, window=0.5):
#     """
#     Return a fused LiDAR cloud in the *reference vehicle frame* for the given timestamp.
#     Uses the same neighbor search, pose transforms, and ICP refinement already in this file.
#     """
#     # Build state index once
#     state_times, state_list = _build_state_index(vehicle_states_list)
#     has_right_lidar = ("lidar4" in sensor_list)

#     # Load ref cloud (speed/yaw-rate aware deskew)
#     ref_state = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ref_ts), max_gap_s=2.0)
#     ref_speed = float(ref_state.get('speed_mps', 0.0)) if ref_state else 0.0
#     cloud_ref0 = _load_cloud_vehicle_frame(
#         folder_path, ref_ts, extrinsics,
#         speed_mps=ref_speed,
#         right_lidar_flag=has_right_lidar,
#         times_sorted=state_times, states_sorted=state_list
#     )
#     if cloud_ref0.size == 0:
#         return cloud_ref0

#     if window is None or window <= 0.0 or ref_state is None:
#         return cloud_ref0

#     # Find neighbor timestamps within ±window seconds
#     all_ts = _list_ts_in_folder(folder_path)
#     t0 = _parse_iso_ts(ref_state['vehicle_timestamp'])
#     cand = []
#     for ts in all_ts:
#         st = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ts), max_gap_s=2.0)
#         if st is None:
#             continue
#         dt = abs((_parse_iso_ts(st['vehicle_timestamp']) - t0).total_seconds())
#         if 0.0 < dt <= window:
#             cand.append((dt, ts))
#     cand.sort(key=lambda x: x[0])
#     neighbors = [ts for _, ts in cand[:2]]  # keep the same cap as in visualize_fused_cloud

#     if not neighbors:
#         return cloud_ref0

#     T_WV_ref = _T_w_from_state(ref_state)
#     T_VW_ref = np.linalg.inv(T_WV_ref)

#     # Accumulate ref + warped neighbors
#     fused = [cloud_ref0]
#     icp_target_np = cloud_ref0.copy()
#     for ts in neighbors:
#         st = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ts), max_gap_s=2.0)
#         if st is None:
#             continue
#         spd = float(st.get('speed_mps', 0.0))
#         cloud_v_n = _load_cloud_vehicle_frame(
#             folder_path, ts, extrinsics,
#             speed_mps=spd,
#             right_lidar_flag=has_right_lidar,
#             times_sorted=state_times, states_sorted=state_list
#         )
#         if cloud_v_n.size == 0:
#             continue

#         T_WV_n = _T_w_from_state(st)
#         T_Vref_Vn = T_VW_ref @ T_WV_n

#         # ICP refinement (same routine already defined)
#         T_icp, fit, rmse = _o3d_icp_refine(
#             source_np=cloud_v_n,
#             target_np=icp_target_np,
#             init_T=T_Vref_Vn,
#             voxel=0.3,
#             max_corr=10,
#             point_to_plane=False,
#             max_iter=100
#         )
#         T_final = T_icp if (fit >= 0.10 and np.isfinite(rmse)) else T_Vref_Vn
#         # if fit < 0.10 or not np.isfinite(rmse):
#         #     print(f"AAAAAAAAAAAAAAAAAAAAAAAAAAA[ICP] low fitness ({fit:.3f}); using init transform.")
#         # else:
#         #     print(f"BBBBBBBBBBBBBBBBBBBBBBBBBB[ICP] fitness={fit:.3f} rmse={rmse:.3f}")
#         cloud_ref_n = _apply_T(cloud_v_n, T_final)
#         fused.append(cloud_ref_n)

#     fused_np = np.vstack(fused)
#     return fused_np

# def get_fused_cloud_for_ts(folder_path, ref_ts, extrinsics, sensor_list,
#                            vehicle_states_list, window=0.5):
#     """
#     Instrumented, no-subsampling, always-ICP version.
#     Prints timing (ms) for: state index, ref load, neighbor search, per-neighbor load & ICP, total.
#     """
#     import heapq
#     import numpy as np
#     import time

#     def ms(s):  # seconds -> milliseconds (int)
#         return int(round(s * 1000))

#     t0_total = time.perf_counter()

#     # Build state index once
#     t0 = time.perf_counter()
#     state_times, state_list = _build_state_index(vehicle_states_list)
#     has_right_lidar = ("lidar4" in sensor_list)
#     t_state_index = time.perf_counter() - t0

#     # Load ref cloud (deskew aware)
#     t0 = time.perf_counter()
#     ref_state = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ref_ts), max_gap_s=2.0)
#     ref_speed = float(ref_state.get('speed_mps', 0.0)) if ref_state else 0.0
#     cloud_ref0 = _load_cloud_vehicle_frame(
#         folder_path, ref_ts, extrinsics,
#         speed_mps=ref_speed,
#         right_lidar_flag=has_right_lidar,
#         times_sorted=state_times, states_sorted=state_list
#     )
#     t_ref_load = time.perf_counter() - t0

#     if cloud_ref0.size == 0 or ref_state is None or window is None or window <= 0.0:
#         print(f"[fuse] state_index={ms(t_state_index)}ms  ref_load={ms(t_ref_load)}ms  early_return (empty/no-window)")
#         return cloud_ref0

#     # Neighbor candidates
#     t0 = time.perf_counter()
#     all_ts = _list_ts_in_folder(folder_path)
#     t_list_ts = time.perf_counter() - t0

#     t0 = time.perf_counter()
#     t_ref = _parse_iso_ts(ref_state['vehicle_timestamp'])
#     cand = []
#     for ts in all_ts:
#         st = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ts), max_gap_s=2.0)
#         if st is None:
#             continue
#         dt = abs((_parse_iso_ts(st['vehicle_timestamp']) - t_ref).total_seconds())
#         if 0.0 < dt <= window:
#             cand.append((dt, ts))
#     t_neighbor_search = time.perf_counter() - t0

#     if not cand:
#         print(f"[fuse] state_index={ms(t_state_index)}ms  ref_load={ms(t_ref_load)}ms  list_ts={ms(t_list_ts)}ms  "
#               f"nbr_search={ms(t_neighbor_search)}ms  neighbors=0  return_ref")
#         return cloud_ref0

#     # Keep the same number of neighbors as before (2)
#     t0 = time.perf_counter()
#     MAX_NEI = 2
#     neighbors = [ts for (_, ts) in heapq.nsmallest(MAX_NEI, cand, key=lambda x: x[0])]
#     t_pick_neighbors = time.perf_counter() - t0

#     if not neighbors:
#         print(f"[fuse] state_index={ms(t_state_index)}ms  ref_load={ms(t_ref_load)}ms  list_ts={ms(t_list_ts)}ms  "
#               f"nbr_search={ms(t_neighbor_search)}ms  pick_nbrs={ms(t_pick_neighbors)}ms  neighbors=0  return_ref")
#         return cloud_ref0

#     T_WV_ref = _T_w_from_state(ref_state)
#     T_VW_ref = np.linalg.inv(T_WV_ref)

#     fused = [cloud_ref0]
#     icp_target_np = cloud_ref0  # full cloud

#     # Pre-pull neighbor states and dt
#     neighbor_info = []
#     for ts in neighbors:
#         st = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ts), max_gap_s=2.0)
#         if st is None:
#             continue
#         dt = abs((_parse_iso_ts(st['vehicle_timestamp']) - t_ref).total_seconds())
#         neighbor_info.append((ts, st, dt))

#     # Per-neighbor timings
#     per_nbr_logs = []
#     t_neighbors_total_load = 0.0
#     t_neighbors_total_icp = 0.0

#     for i, (ts, st, dt_s) in enumerate(neighbor_info):
#         # neighbor cloud load
#         t0 = time.perf_counter()
#         spd = float(st.get('speed_mps', 0.0))
#         cloud_v_n = _load_cloud_vehicle_frame(
#             folder_path, ts, extrinsics,
#             speed_mps=spd,
#             right_lidar_flag=has_right_lidar,
#             times_sorted=state_times, states_sorted=state_list
#         )
#         t_load = time.perf_counter() - t0
#         t_neighbors_total_load += t_load

#         if cloud_v_n.size == 0:
#             per_nbr_logs.append(
#                 f"[nbr{i}] ts={ts} load={ms(t_load)}ms  points=0  (skipped ICP)"
#             )
#             continue

#         # initial transform
#         T_WV_n = _T_w_from_state(st)
#         T_Vref_Vn = T_VW_ref @ T_WV_n

#         # ICP (always)
#         t0 = time.perf_counter()
#         T_icp, fit, rmse = _o3d_icp_refine(
#             source_np=cloud_v_n,
#             target_np=icp_target_np,
#             init_T=T_Vref_Vn,
#             voxel=0.3,            # your previous setting
#             max_corr=10.0,        # your previous setting
#             point_to_plane=False, # point-to-point
#             max_iter=50           # your previous setting
#         )
#         t_icp = time.perf_counter() - t0
#         t_neighbors_total_icp += t_icp

#         T_final = T_icp if (np.isfinite(rmse) and fit >= 0.0) else T_Vref_Vn
#         cloud_ref_n = _apply_T(cloud_v_n, T_final)
#         fused.append(cloud_ref_n)

#         per_nbr_logs.append(
#             f"[nbr{i}] ts={ts} dt={dt_s:.3f}s spd={spd:.2f}m/s "
#             f"load={ms(t_load)}ms icp={ms(t_icp)}ms "
#             f"src_pts={cloud_v_n.shape[0]} tgt_pts={icp_target_np.shape[0]} "
#             f"fit={fit:.3f} rmse={rmse:.3f}"
#         )

#     fused_np = np.vstack(fused).astype(np.float32, copy=False)
#     t_total = time.perf_counter() - t0_total

#     # Summary print
#     print(
#         "[fuse] "
#         f"state_index={ms(t_state_index)}ms  "
#         f"ref_load={ms(t_ref_load)}ms  "
#         f"list_ts={ms(t_list_ts)}ms  "
#         f"nbr_search={ms(t_neighbor_search)}ms  "
#         f"pick_nbrs={ms(t_pick_neighbors)}ms  "
#         f"nbr_load_sum={ms(t_neighbors_total_load)}ms  "
#         f"nbr_icp_sum={ms(t_neighbors_total_icp)}ms  "
#         f"total={ms(t_total)}ms  "
#         f"neighbors={len(neighbor_info)}  "
#         f"ref_pts={cloud_ref0.shape[0]}  fused_pts={fused_np.shape[0]}"
#     )
#     for line in per_nbr_logs:
#         print(line)

#     return fused_np

def get_fused_cloud_for_ts(folder_path, ref_ts, extrinsics, sensor_list,
                           vehicle_states_list, window=0.5):
    """
    No-subsampling, always-ICP version.
    Same logic as before, but without prints or timing.
    """
    import heapq
    import numpy as np

    # Build state index once
    state_times, state_list = _build_state_index(vehicle_states_list)
    has_right_lidar = ("lidar4" in sensor_list)

    # Load ref cloud (deskew aware)
    ref_state = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ref_ts), max_gap_s=2.0)
    ref_speed = float(ref_state.get('speed_mps', 0.0)) if ref_state else 0.0
    cloud_ref0 = _load_cloud_vehicle_frame(
        folder_path, ref_ts, extrinsics,
        speed_mps=ref_speed,
        right_lidar_flag=has_right_lidar,
        times_sorted=state_times, states_sorted=state_list
    )

    if cloud_ref0.size == 0 or ref_state is None or window is None or window <= 0.0:
        return cloud_ref0

    # Neighbor candidates
    all_ts = _list_ts_in_folder(folder_path)
    t_ref = _parse_iso_ts(ref_state['vehicle_timestamp'])
    cand = []
    for ts in all_ts:
        st = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ts), max_gap_s=2.0)
        if st is None:
            continue
        dt = abs((_parse_iso_ts(st['vehicle_timestamp']) - t_ref).total_seconds())
        if 0.0 < dt <= window:
            cand.append((dt, ts))

    if not cand:
        return cloud_ref0

    # Keep the same number of neighbors as before (2)
    MAX_NEI = 2
    neighbors = [ts for (_, ts) in heapq.nsmallest(MAX_NEI, cand, key=lambda x: x[0])]

    if not neighbors:
        return cloud_ref0

    T_WV_ref = _T_w_from_state(ref_state)
    T_VW_ref = np.linalg.inv(T_WV_ref)

    fused = [cloud_ref0]
    icp_target_np = cloud_ref0  # full cloud

    # Pre-pull neighbor states and dt
    neighbor_info = []
    for ts in neighbors:
        st = _nearest_state(state_times, state_list, _parse_fileprefix_ts(ts), max_gap_s=2.0)
        if st is None:
            continue
        dt = abs((_parse_iso_ts(st['vehicle_timestamp']) - t_ref).total_seconds())
        neighbor_info.append((ts, st, dt))

    for i, (ts, st, dt_s) in enumerate(neighbor_info):
        # neighbor cloud load
        spd = float(st.get('speed_mps', 0.0))
        cloud_v_n = _load_cloud_vehicle_frame(
            folder_path, ts, extrinsics,
            speed_mps=spd,
            right_lidar_flag=has_right_lidar,
            times_sorted=state_times, states_sorted=state_list
        )

        if cloud_v_n.size == 0:
            continue

        # initial transform
        T_WV_n = _T_w_from_state(st)
        T_Vref_Vn = T_VW_ref @ T_WV_n

        # ICP (always)
        T_icp, fit, rmse = _o3d_icp_refine(
            source_np=cloud_v_n,
            target_np=icp_target_np,
            init_T=T_Vref_Vn,
            voxel=0.3,            # your previous setting
            max_corr=10.0,        # your previous setting
            point_to_plane=False, # point-to-point
            max_iter=50           # your previous setting
        )

        T_final = T_icp if (np.isfinite(rmse) and fit >= 0.0) else T_Vref_Vn
        cloud_ref_n = _apply_T(cloud_v_n, T_final)
        fused.append(cloud_ref_n)

    fused_np = np.vstack(fused).astype(np.float32, copy=False)
    return fused_np



def annotate(base_folder, folder_path, extrinsics, camera_params, sensor_list, vehicle_states_list, data):
    timings = {}
    t_total_start = time.perf_counter()

    t_init_start = time.perf_counter()
    folder_name = os.path.basename(folder_path)

    print(folder_name)
    filtered_keys = [k for k, v in data.items() if folder_name in k]
    all_timestamps = set([os.path.basename(k).split('_')[0] for k in filtered_keys])
    count_annotated = 0
    t_init_end = time.perf_counter()
    timings['initialization'] = t_init_end - t_init_start

    t_sensor_start = time.perf_counter()
    if "FRONT_CAMERA" in sensor_list:
        iterable_indices = [1, 0, 2]
        front_camera_flag = True
    else:
        iterable_indices = [1, 0]
        front_camera_flag = False

    if "lidar4" in sensor_list:
        right_lidar_flag = True
    else:
        right_lidar_flag = False
    t_sensor_end = time.perf_counter()
    timings['sensor_selection'] = t_sensor_end - t_sensor_start

    all_items_in_folder= os.listdir(folder_path)
    
    ts_list = _list_ts_in_folder(folder_path)
    print(len(ts_list), "timestamps found in folder")

    print(f"Annotating {len(all_timestamps)} timestamps in folder {folder_path}...")

    for timestamp in tqdm(all_timestamps):
        try:
            t_timestamp_start = time.perf_counter()
            all_annotated_boxes = []
            all_annotated_classes = []
            all_images = {}

            zo_segments_for_ts = None
            zo_classes_for_ts  = None
            zi_segments_for_ts = None
            zi_classes_for_ts  = None
            fr_segments_for_ts = None
            fr_classes_for_ts  = None

            t_camera_loop_start = time.perf_counter()
            for idx in iterable_indices:  # In order - zoomed in, front, zoomed out
                t_camera_block_start = time.perf_counter()

                relative_path = os.path.relpath(folder_path, base_folder).split('/')[-1]
                keys = [k for k, v in data.items() if (relative_path + '/' + timestamp) in k]
                # keys = filtered_keys
                if len(keys) == 0 or len(keys) < idx + 1:
                    continue
                
                # Try to get segments from YOLO data, but don't fail if not available
                try:
                    segments = data[keys[idx]]["segments"]
                    segment_classes = data[keys[idx]]["classes"]
                except KeyError:
                    # No YOLO data available, create empty segments to continue with LiDAR-only processing
                    segments = []
                    segment_classes = []
                original_image_path = folder_path + '/' + os.path.basename(keys[idx])
                image = cv2.imread(original_image_path)
                
                # Skip if image can't be loaded
                if image is None:
                    print(f"Warning: Could not load image {original_image_path}")
                    continue

                image_with_segments = image.copy()
                if "Front_Camera" in keys[idx]:
                    camera_name = "FRONT_CAMERA"
                elif "Front_Zoomed_In_Camera" in keys[idx]:
                    camera_name = "FRONT_ZOOMED_IN_CAMERA"
                elif "Front_Zoomed_Out_Camera" in keys[idx]:
                    camera_name = "FRONT_ZOOMED_OUT_CAMERA"
                all_images[camera_name] = image.copy()

                cam_to_folder = {
                    "FRONT_ZOOMED_OUT_CAMERA": "image_2",
                    "FRONT_CAMERA": "image_1",
                    "FRONT_ZOOMED_IN_CAMERA": "image_0",
                }

                # seg_out_root = "KITTI_dataset_v13/seg_overlays"
                # # match the frame index you later use for KITTI (same derivation from timestamp)
                # frame_idx_for_img = timestamp.split('.')[1] if '.' in timestamp else timestamp
                # seg_out_path = os.path.join(seg_out_root, cam_to_folder[camera_name], f"{frame_idx_for_img}.png")
                # save_segment_overlay(image, segments, seg_out_path)

                if camera_name == "FRONT_ZOOMED_IN_CAMERA":
                    zi_segments_for_ts = segments
                    zi_classes_for_ts  = segment_classes
                elif camera_name == "FRONT_CAMERA":
                    fr_segments_for_ts = segments
                    fr_classes_for_ts  = segment_classes
                elif camera_name == "FRONT_ZOOMED_OUT_CAMERA":
                    zo_segments_for_ts = segments
                    zo_classes_for_ts  = segment_classes


                # combined_points, raw_combined_points = process_data(
                #      folder_path, timestamp, extrinsics, camera_params, camera_name, right_lidar_flag, all_items_in_folder
                # )
                                    
                _, raw_combined_points = process_data(
                    folder_path, timestamp, extrinsics, camera_params, camera_name, right_lidar_flag, all_items_in_folder
                )
                #Use denser (neighbor-fused) point cloud in the center frame
                if len(vehicle_states_list) == 0:
                    combined_points, raw_combined_points = process_data(
                    folder_path, timestamp, extrinsics, camera_params, camera_name, right_lidar_flag, all_items_in_folder
                ) 
                else:
                    fused_points = get_fused_cloud_for_ts(
                        folder_path=folder_path,
                        ref_ts=timestamp,
                        extrinsics=extrinsics,
                        sensor_list=sensor_list,
                        vehicle_states_list=vehicle_states_list,
                        window=0.5
                    )
                    combined_points = fused_points

                params = camera_params[camera_name]
                intrinsic = np.array([
                    [params['projection'][0], 0, params['projection'][2]],
                    [0, params['projection'][1], params['projection'][3]],
                    [0, 0, 1]
                ])
                rotation = quaternion_to_rotation_matrix(params["rotation"])
                translation = np.array(params["translation"])
                distortion = np.array(params["distortion"])

                t_project_lidar_start = time.perf_counter()
                projected_points, depths, points_in_front_3d = project_lidar_to_camera_v2(combined_points, params)
                t_project_lidar_end = time.perf_counter()
                # print(f"[Timing] project_lidar_to_camera_v2 for {camera_name}: {t_project_lidar_end - t_project_lidar_start:.4f}s")

                projected_points = projected_points.astype(int)
                valid_mask = (
                    (projected_points[:, 0] >= 0)
                    & (projected_points[:, 0] < image.shape[1])
                    & (projected_points[:, 1] >= 0)
                    & (projected_points[:, 1] < image.shape[0])
                )
                projected_points = projected_points[valid_mask]
                depths = depths[valid_mask]
                points_in_front_3d = points_in_front_3d[valid_mask]

                t_mask_start = time.perf_counter()
                mask = create_mask_from_segments(segments, image.shape)
                dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
                t_mask_end = time.perf_counter()
                # print(f"[Timing] create/dilate mask for {camera_name}: {t_mask_end - t_mask_start:.4f}s")

                t_filter_points_start = time.perf_counter()
                valid_indices = []
                for i in range(projected_points.shape[0]):
                    x, y = projected_points[i]
                    if dilated_mask[y, x] > 0:
                        valid_indices.append(i)
                valid_indices = np.array(valid_indices)
                t_filter_points_end = time.perf_counter()
                # print(f"[Timing] filter projected points for {camera_name}: {t_filter_points_end - t_filter_points_start:.4f}s")

                if len(valid_indices) == 0:
                    continue

                filtered_projected_points = projected_points[valid_indices]
                filtered_depths = depths[valid_indices]
                filtered_points_3d = points_in_front_3d[valid_indices]

                t_clustering_start = time.perf_counter()
                clusters = cluster_points(filtered_points_3d)
                
                # Use 2D segments if available, otherwise use LiDAR-only clustering
                if len(segments) > 0:
                    bounding_boxes, all_corners, box_classes = find_cluster_with_smallest_centroid_norm_and_assign_classes(
                        clusters, filtered_points_3d, segments, segment_classes, camera_name, camera_params
                    )
                else:
                    # LiDAR-only clustering: create bboxes for all clusters with default class
                    bounding_boxes, all_corners, box_classes = create_lidar_only_bboxes(
                        clusters, filtered_points_3d, camera_name, camera_params
                    )
                t_clustering_end = time.perf_counter()
                # print(f"[Timing] clustering & bbox for {camera_name}: {t_clustering_end - t_clustering_start:.4f}s")

                if len(all_corners) == 0:
                    continue
                final_corners = np.vstack(all_corners)
                lidar_only_corners = deepcopy(final_corners)
                box_classes = np.vstack(box_classes)

                # inverse_transformed_bounding_boxes = (lidar_only_corners - camera_params[camera_name]['translation']) @ quaternion_to_rotation_matrix(camera_params[camera_name]['rotation'])
                R_cam = quaternion_to_rotation_matrix(camera_params[camera_name]['rotation'])
                t_cam = np.array(camera_params[camera_name]['translation']).reshape(3, 1)
                inverse_transformed_bounding_boxes = (R_cam.T @ (lidar_only_corners.T - t_cam)).T
                all_annotated_boxes.append(inverse_transformed_bounding_boxes)
                all_annotated_classes.append(box_classes)

                t_camera_block_end = time.perf_counter()
                # print(f"[Timing] Full camera block {camera_name}: {t_camera_block_end - t_camera_block_start:.4f}s")

            t_camera_loop_end = time.perf_counter()
            # print(f"[Timing] Camera loop for timestamp {timestamp}: {t_camera_loop_end - t_camera_loop_start:.4f}s")

            t_save_start = time.perf_counter()
            if len(all_annotated_boxes) == 0:
                continue

            all_annotated_classes = np.vstack(all_annotated_classes)
            all_annotated_boxes = np.vstack(all_annotated_boxes)
            all_annotated_boxes = filter_bboxes_proximity(all_annotated_boxes, threshold=1.0)

            if all_annotated_boxes.size > 0:
                if all_annotated_classes.ndim == 1:
                    all_annotated_classes = all_annotated_classes.reshape(-1, 1)

                # Snap first with ZI, then only the leftovers with ZO (minimal change, reuses your snap fn)
                all_annotated_boxes = _snap_priority_ZI_then_ZO(
                    all_annotated_boxes,
                    zi_segments_for_ts, zi_classes_for_ts,
                    zo_segments_for_ts, zo_classes_for_ts,
                    camera_params
                )
            else:
                print("No annotated boxes to snap.")


            kitti_output_dir = "/home/rtml/shounak_files/KITTI_dataset_v13"
            os.makedirs(kitti_output_dir, exist_ok=True)

            velodyne_dir = os.path.join(kitti_output_dir, "velodyne")
            calib_dir = os.path.join(kitti_output_dir, "calib")
            label_dir = os.path.join(kitti_output_dir, "label_2")
            vehstate_dir = os.path.join(kitti_output_dir, "vehicle_state")

            os.makedirs(velodyne_dir, exist_ok=True)
            os.makedirs(calib_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            os.makedirs(vehstate_dir, exist_ok=True)

            frame_idx = timestamp

            all_bboxes_on_zoom_out_cam = project_bbox_to_other_cameras(
                all_annotated_boxes, "FRONT_ZOOMED_OUT_CAMERA", "rerun_all", all_images["FRONT_ZOOMED_OUT_CAMERA"], camera_params
            )

            frame_idx = frame_idx.split('.')[1]

            save_kitti_images(all_images, kitti_output_dir, frame_idx, front_camera_flag)
            save_kitti_velodyne(raw_combined_points, velodyne_dir, frame_idx)
            Tr_velo_to_cam, R0_rect =save_kitti_calib(camera_params, extrinsics, calib_dir, frame_idx, front_camera_flag)
            save_kitti_label(all_bboxes_on_zoom_out_cam, all_annotated_boxes, all_annotated_classes, label_dir, frame_idx, front_camera_flag, Tr_velo_to_cam, R0_rect)
            save_vehicle_states(frame_idx, timestamp, vehicle_states_list, vehstate_dir, folder_name)

            count_annotated += 1
            t_save_end = time.perf_counter()
            # print(f"[Timing] Saving outputs for timestamp {timestamp}: {t_save_end - t_save_start:.4f}s")

            t_timestamp_end = time.perf_counter()
        
        except Exception as e:
            print(f"Error processing timestamp {timestamp}: {e}")
            continue
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            break

    t_total_end = time.perf_counter()
    # print(f"[Timing] Total annotate() time: {t_total_end - t_total_start:.4f}s")
    # print("[Timing] Breakdown:", timings)
    return count_annotated

