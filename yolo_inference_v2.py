import os
import json
import argparse
import torch
from ultralytics import YOLO
import time

# Set up argument parser
parser = argparse.ArgumentParser(description='Process images with YOLOv8 segmentation model.')
parser.add_argument('--text_file', type=str, help='Text file containing image paths', default="all_image_files_wzd.txt")
args = parser.parse_args()

# Configuration
folder_name = "/media/rtml/Expansion/RTML_data/Work_Zone_Dataset/"
batch_size = 384  # Total batch size across both GPUs
output_file = 'yolo_segmentation_results_3class_v2.json'

# Load image paths
with open(args.text_file, 'r') as f:
    image_files = [os.path.join(folder_name, line.strip()) for line in f.readlines()]

# Check available GPUs
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')

# Load model separately on each GPU
model0 = YOLO('yolov8l-seg-best-3class.pt').to(device0)
model1 = YOLO('yolov8l-seg-best-3class.pt').to(device1)

total_batches = (len(image_files) + batch_size - 1) // batch_size

def get_relative_path(full_path):
    return os.path.relpath(full_path, folder_name)

# Initialize JSON file
with open(output_file, 'w') as f:
    f.write('{\n')

start_time = time.time()
first_batch = True

def process_batch(batch_paths):
    """Manually split the batch and process with two GPUs"""
    half = len(batch_paths) // 2
    batch0 = batch_paths[:half]
    batch1 = batch_paths[half:]

    results0 = []
    results1 = []

    try:
        if batch0:
            results0 = model0.predict(batch0, batch=len(batch0), verbose=False, device=device0)
        if batch1:
            results1 = model1.predict(batch1, batch=len(batch1), verbose=False, device=device1)
    except Exception as e:
        print(f"Batch error: {str(e)}")

    all_results = list(zip(batch0, results0)) + list(zip(batch1, results1))
    return all_results

for batch_idx in range(0, len(image_files), batch_size):
    batch_paths = image_files[batch_idx:batch_idx + batch_size]
    batch_entries = []
    
    # Process batch manually across two GPUs
    processed = process_batch(batch_paths)
    
    # Create JSON entries
    for path, result in processed:
        relative_path = get_relative_path(path)
        try:
            entry = {
                'boxes': result.boxes.xyxy.cpu().tolist() if result.boxes else [],
                'classes': result.boxes.cls.cpu().tolist() if result.boxes else [],
                'confidences': result.boxes.conf.cpu().tolist() if result.boxes else [],
                'segments': [seg.tolist() for seg in result.masks.xy] if result.masks else []
            }
        except Exception as e:
            print(f"Error processing {relative_path}: {str(e)}")
            entry = {}

        batch_entries.append(f'{json.dumps(relative_path)}: {json.dumps(entry, separators=(",", ":"))}')

    # Write batch to file
    if batch_entries:
        with open(output_file, 'a') as f:
            if not first_batch:
                f.write(',\n')
            f.write(',\n'.join(batch_entries))
            first_batch = False

    # Progress tracking
    completed = (batch_idx + len(batch_paths)) / len(image_files)
    print(f"Progress: {completed * 100:.1f}% | "
          f"Elapsed: {time.time() - start_time:.1f}s | "
          f"Batch {batch_idx//batch_size + 1}/{total_batches}")

# Close JSON structure
with open(output_file, 'a') as f:
    f.write('\n}')

print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
print(f"Results saved to {output_file}")


# import os
# import json
# import argparse
# import torch
# from ultralytics import YOLO
# import time
# from concurrent.futures import ThreadPoolExecutor

# # Set up argument parser
# parser = argparse.ArgumentParser(description='Process images with YOLOv8 segmentation model.')
# parser.add_argument('--text_file', type=str, help='Text file containing image paths', default="all_image_files_wzd.txt")
# args = parser.parse_args()

# # Configuration
# folder_name = "/media/rtml/Expansion/RTML_data/Work_Zone_Dataset/"
# batch_size = 256  # Adjust based on VRAM
# output_file = 'yolo_segmentation_results_3class_v2.json'

# # Load image paths
# with open(args.text_file, 'r') as f:
#     image_files = [os.path.join(folder_name, line.strip()) for line in f.readlines()]

# # Model setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = YOLO('yolov8l-seg-best-3class.pt').to(device)
# total_batches = (len(image_files) + batch_size - 1) // batch_size

# print(f"Processing {len(image_files)} images in {total_batches} batches...")

# def get_relative_path(full_path):
#     return os.path.relpath(full_path, folder_name)

# # Initialize JSON file
# with open(output_file, 'w') as f:
#     f.write('{\n')

# start_time = time.time()
# first_batch = True

# def process_batch(batch_paths):
#     """Process a batch of images using YOLO's native batching"""
#     try:
#         results = model.predict(batch_paths, batch=batch_size, verbose=False)
#         return list(zip(batch_paths, results))
#     except Exception as e:
#         print(f"Batch error: {str(e)}")
#         return []

# for batch_idx in range(0, len(image_files), batch_size):
#     batch_paths = image_files[batch_idx:batch_idx + batch_size]
#     batch_entries = []
    
#     # Process batch using parallel loading and native batching
#     processed = process_batch(batch_paths)
    
#     # Create JSON entries with error handling
#     for path, result in processed:
#         relative_path = get_relative_path(path)
#         try:
#             entry = {
#                 'boxes': result.boxes.xyxy.cpu().tolist() if result.boxes else [],
#                 'classes': result.boxes.cls.cpu().tolist() if result.boxes else [],
#                 'confidences': result.boxes.conf.cpu().tolist() if result.boxes else [],
#                 'segments': [seg.tolist() for seg in result.masks.xy] if result.masks else []
#             }
#         except Exception as e:
#             print(f"Error processing {relative_path}: {str(e)}")
#             entry = {}

#         batch_entries.append(f'{json.dumps(relative_path)}: {json.dumps(entry, separators=(",", ":"))}')

#     # Write entire batch to file
#     if batch_entries:
#         with open(output_file, 'a') as f:
#             if not first_batch:
#                 f.write(',\n')
#             f.write(',\n'.join(batch_entries))
#             first_batch = False

#     # Progress tracking
#     completed = (batch_idx + len(batch_paths)) / len(image_files)
#     print(f"Progress: {completed * 100:.1f}% | "
#           f"Elapsed: {time.time() - start_time:.1f}s | "
#           f"Batch {batch_idx//batch_size + 1}/{total_batches}")

# # Close JSON structure
# with open(output_file, 'a') as f:
#     f.write('\n}')

# print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
# print(f"Results saved to {output_file}")



# import os
# import json
# import argparse
# import torch
# from ultralytics import YOLO
# import time

# # Set up argument parser
# parser = argparse.ArgumentParser(description='Process images with YOLOv8 segmentation model.')
# parser.add_argument('--text_file', type=str, help='Text file containing image paths', default="all_image_files_wzd.txt")
# args = parser.parse_args()

# # Configuration
# folder_name = "/media/rtml/Expansion/RTML_data/Work_Zone_Dataset/"
# batch_size = 512  # Total batch size across both GPUs
# output_file = 'yolo_segmentation_results_3class_v2.json'

# # Load image paths
# with open(args.text_file, 'r') as f:
#     image_files = [os.path.join(folder_name, line.strip()) for line in f.readlines()]

# # Check available GPUs
# device0 = torch.device('cuda:0')
# device1 = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')

# # Load model separately on each GPU
# model0 = YOLO('yolov8l-seg-best-3class.pt').to(device0)
# model1 = YOLO('yolov8l-seg-best-3class.pt').to(device1)

# total_batches = (len(image_files) + batch_size - 1) // batch_size

# def get_relative_path(full_path):
#     return os.path.relpath(full_path, folder_name)

# # Initialize JSON file
# with open(output_file, 'w') as f:
#     f.write('{\n')

# start_time = time.time()
# first_batch = True

# def process_batch(batch_paths):
#     """Manually split the batch and process with two GPUs"""
#     half = len(batch_paths) // 2
#     batch0 = batch_paths[:half]
#     batch1 = batch_paths[half:]

#     results0 = []
#     results1 = []

#     try:
#         if batch0:
#             results0 = model0.predict(batch0, batch=len(batch0), verbose=False, device=device0)
#         if batch1:
#             results1 = model1.predict(batch1, batch=len(batch1), verbose=False, device=device1)
#     except Exception as e:
#         print(f"Batch error: {str(e)}")

#     all_results = list(zip(batch0, results0)) + list(zip(batch1, results1))
#     return all_results

# for batch_idx in range(0, len(image_files), batch_size):
#     batch_paths = image_files[batch_idx:batch_idx + batch_size]
#     batch_entries = []
    
#     # Process batch manually across two GPUs
#     processed = process_batch(batch_paths)
    
#     # Create JSON entries
#     for path, result in processed:
#         relative_path = get_relative_path(path)
#         try:
#             entry = {
#                 'boxes': result.boxes.xyxy.cpu().tolist() if result.boxes else [],
#                 'classes': result.boxes.cls.cpu().tolist() if result.boxes else [],
#                 'confidences': result.boxes.conf.cpu().tolist() if result.boxes else [],
#                 'segments': [seg.tolist() for seg in result.masks.xy] if result.masks else []
#             }
#         except Exception as e:
#             print(f"Error processing {relative_path}: {str(e)}")
#             entry = {}

#         batch_entries.append(f'{json.dumps(relative_path)}: {json.dumps(entry, separators=(",", ":"))}')

#     # Write batch to file
#     if batch_entries:
#         with open(output_file, 'a') as f:
#             if not first_batch:
#                 f.write(',\n')
#             f.write(',\n'.join(batch_entries))
#             first_batch = False

#     # Progress tracking
#     completed = (batch_idx + len(batch_paths)) / len(image_files)
#     print(f"Progress: {completed * 100:.1f}% | "
#           f"Elapsed: {time.time() - start_time:.1f}s | "
#           f"Batch {batch_idx//batch_size + 1}/{total_batches}")

# # Close JSON structure
# with open(output_file, 'a') as f:
#     f.write('\n}')

# print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
# print(f"Results saved to {output_file}")


# import os
# import json
# import argparse
# import torch
# from ultralytics import YOLO
# import time

# # Set up argument parser
# parser = argparse.ArgumentParser(description='Process images with YOLOv8 segmentation model.')
# parser.add_argument('--text_file', type=str, help='Text file containing image paths', default="all_image_files_wzd.txt")
# args = parser.parse_args()

# # Configuration
# folder_name = "/media/rtml/Expansion/RTML_data/Work_Zone_Dataset/"
# batch_size = 256  # Total batch across both GPUs
# output_file = 'yolo_segmentation_results_3class_v2.json'

# # Load image paths
# with open(args.text_file, 'r') as f:
#     image_files = [os.path.join(folder_name, line.strip()) for line in f.readlines()]

# # Multi-GPU setup
# model = YOLO('yolov8l-seg-best-3class.pt').to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# total_batches = (len(image_files) + batch_size - 1) // batch_size


# def get_relative_path(full_path):
#     return os.path.relpath(full_path, folder_name)

# # Initialize JSON file
# with open(output_file, 'w') as f:
#     f.write('{\n')

# start_time = time.time()
# first_batch = True

# def process_batch(batch_paths):
#     """Process a batch using both GPUs"""
#     try:
#         results = model.predict(batch_paths, batch=batch_size, verbose=False)
#         return list(zip(batch_paths, results))
#     except Exception as e:
#         print(f"Batch error: {str(e)}")
#         return []

# for batch_idx in range(0, len(image_files), batch_size):
#     batch_paths = image_files[batch_idx:batch_idx + batch_size]
#     batch_entries = []
    
#     # Process batch using both GPUs
#     processed = process_batch(batch_paths)
    
#     # Create JSON entries
#     for path, result in processed:
#         relative_path = get_relative_path(path)
#         try:
#             entry = {
#                 'boxes': result.boxes.xyxy.cpu().tolist() if result.boxes else [],
#                 'classes': result.boxes.cls.cpu().tolist() if result.boxes else [],
#                 'confidences': result.boxes.conf.cpu().tolist() if result.boxes else [],
#                 'segments': [seg.tolist() for seg in result.masks.xy] if result.masks else []
#             }
#         except Exception as e:
#             print(f"Error processing {relative_path}: {str(e)}")
#             entry = {}

#         batch_entries.append(f'{json.dumps(relative_path)}: {json.dumps(entry, separators=(",", ":"))}')

#     # Write batch to file
#     if batch_entries:
#         with open(output_file, 'a') as f:
#             if not first_batch:
#                 f.write(',\n')
#             f.write(',\n'.join(batch_entries))
#             first_batch = False

#     # Progress tracking
#     completed = (batch_idx + len(batch_paths)) / len(image_files)
#     print(f"Progress: {completed * 100:.1f}% | "
#           f"Elapsed: {time.time() - start_time:.1f}s | "
#           f"Batch {batch_idx//batch_size + 1}/{total_batches}")

# # Close JSON structure
# with open(output_file, 'a') as f:
#     f.write('\n}')

# print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
# print(f"Results saved to {output_file}")
