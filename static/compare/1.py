import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def get_video_files(directory):
    """Get all video files from directory"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    
    for file in os.listdir(directory):
        if file.lower().endswith(video_extensions):
            video_files.append(os.path.join(directory, file))
    
    return sorted(video_files)

def get_subdirectories(directory):
    """Get all subdirectories except 'comparison'"""
    return [d for d in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, d)) 
            and d != 'comparison']

def get_video_label(filename):
    """Get label based on filename"""
    filename = filename.lower()
    if '_gt.mp4' in filename:
        return 'Ground Truth'
    elif '_ours.mp4' in filename:
        return 'Vid2World (Ours)'
    elif '_hq.mp4' in filename:
        return 'DIAMOND-HQ'
    elif '_fast.mp4' in filename:
        return 'DIAMOND-Fast'
    return 'Unknown'

def get_video_order(filename):
    """Get order number for sorting videos"""
    filename = filename.lower()
    if '_gt.mp4' in filename:
        return 0  # Ground Truth first
    elif '_ours.mp4' in filename:
        return 1  # Vid2World (Ours) second
    elif '_fast.mp4' in filename:
        return 2  # DIAMOND-FAST third
    elif '_hq.mp4' in filename:
        return 3  # DIAMOND-HQ fourth
    return 4  # Unknown last

def add_label_to_frame(frame, label):
    """Add label text to the top of the frame using PIL for better quality"""
    # Convert frame to PIL Image
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    
    # Load a high-quality font
    try:
        # Try to use Arial font
        font = ImageFont.truetype("Arial", 30)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Get text size
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate text position (centered)
    text_x = (frame.shape[1] - text_width) // 2
    text_y = (frame.shape[0] - text_height) // 2
    
    # Add text with black color
    draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0))
    
    # Convert back to OpenCV format
    return np.array(pil_image)

def center_crop(frame, target_width=512, target_height=275):
    """Center crop the frame to target dimensions"""
    height, width = frame.shape[:2]
    
    # Calculate crop dimensions
    crop_width = min(width, int(height * target_width / target_height))
    crop_height = min(height, int(width * target_height / target_width))
    
    # Calculate crop coordinates
    x = (width - crop_width) // 2
    y = (height - crop_height) // 2
    
    # Crop the frame
    cropped = frame[y:y+crop_height, x:x+crop_width]
    
    # Resize to target dimensions
    return cv2.resize(cropped, (target_width, target_height))

def create_comparison_video(video_files, output_path):
    """Create a comparison video with videos arranged horizontally"""
    if not video_files:
        print("No video files found")
        return
    
    # Sort video files based on the specified order
    video_files = sorted(video_files, key=lambda x: get_video_order(os.path.basename(x)))
    
    # Get video properties from the first video
    cap = cv2.VideoCapture(video_files[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Set target dimensions
    target_width = 512
    target_height = 275
    
    # Set spacing between videos (in pixels)
    spacing = 30  # Increased spacing
    
    # Set label height
    label_height = 50  # Increased label height for better text display
    
    # Calculate total width for all videos including spacing
    total_width = (target_width * len(video_files)) + (spacing * (len(video_files) - 1))
    total_height = target_height + label_height * 2  # Double the label height for top and bottom margins
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (total_width, total_height))
    
    # Open all video captures
    caps = [cv2.VideoCapture(video) for video in video_files]
    
    try:
        for frame_idx in range(total_frames):
            frames = []
            for i, (cap, video_file) in enumerate(zip(caps, video_files)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Apply center crop to the frame
                    frame = center_crop(frame, target_width, target_height)
                    # Create white backgrounds for top and bottom labels
                    top_label_bg = np.ones((label_height, target_width, 3), dtype=np.uint8) * 255
                    bottom_label_bg = np.ones((label_height, target_width, 3), dtype=np.uint8) * 255
                    # Add label to the top white background
                    label = get_video_label(os.path.basename(video_file))
                    top_label_bg = add_label_to_frame(top_label_bg, label)
                    # Stack the label, frame and bottom margin vertically
                    frame = np.vstack([top_label_bg, frame, bottom_label_bg])
                    frames.append(frame)
                else:
                    # If video is shorter, use last frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                    _, frame = cap.read()
                    frame = center_crop(frame, target_width, target_height)
                    # Create white backgrounds for top and bottom labels
                    top_label_bg = np.ones((label_height, target_width, 3), dtype=np.uint8) * 255
                    bottom_label_bg = np.ones((label_height, target_width, 3), dtype=np.uint8) * 255
                    # Add label to the top white background
                    label = get_video_label(os.path.basename(video_file))
                    top_label_bg = add_label_to_frame(top_label_bg, label)
                    # Stack the label, frame and bottom margin vertically
                    frame = np.vstack([top_label_bg, frame, bottom_label_bg])
                    frames.append(frame)
            
            # Create white spacing frames
            spacing_frames = [np.ones((total_height, spacing, 3), dtype=np.uint8) * 255 for _ in range(len(frames) - 1)]
            
            # Interleave frames and spacing
            final_frames = []
            for i in range(len(frames)):
                final_frames.append(frames[i])
                if i < len(spacing_frames):
                    final_frames.append(spacing_frames[i])
            
            # Concatenate frames horizontally
            comparison = np.hstack(final_frames)
            out.write(comparison)
            
    finally:
        # Release all resources
        for cap in caps:
            cap.release()
        out.release()

def process_directory(directory):
    """Process a single directory"""
    video_files = get_video_files(directory)
    if not video_files:
        print(f"No video files found in {directory}")
        return
    
    output_dir = Path(directory).parent / "comparison"
    output_dir.mkdir(exist_ok=True)
    
    # Create output filename based on directory name
    dir_name = os.path.basename(directory)
    output_path = output_dir / f"{dir_name}_comparison.mp4"
    
    print(f"Processing directory: {directory}")
    create_comparison_video(video_files, output_path)
    print(f"Comparison video saved to {output_path}")

def main():
    # Get the directory containing this script
    current_dir = Path(__file__).parent
    
    # Get all subdirectories
    subdirs = get_subdirectories(current_dir)
    
    if not subdirs:
        print("No subdirectories found")
        return
    
    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(current_dir, subdir)
        process_directory(subdir_path)

if __name__ == "__main__":
    main()
