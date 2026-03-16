import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_interval=10):
    """
    Extracts 1 frame every `frame_interval` frames from a video.
    Returns a list of extracted frame file paths.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    extracted_frame_paths = []
    frame_idx = 0
    saved_count = 0

    print(f"Extracting frames from {video_name} (Total: {total_frames}, FPS: {fps:.2f})...")
    
    with tqdm(total=total_frames//frame_interval) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                out_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:04d}.jpg")
                cv2.imwrite(out_path, frame)
                extracted_frame_paths.append(out_path)
                saved_count += 1
                pbar.update(1)
                
            frame_idx += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return extracted_frame_paths

if __name__ == "__main__":
    # Test script on a sample video
    test_video = "dataset/video/real/000.mp4" 
    if os.path.exists(test_video):
        frames = extract_frames(test_video, "temp/frames", frame_interval=10)
    else:
        print(f"Test video not found: {test_video}. Adjust the path to an existing video.")
