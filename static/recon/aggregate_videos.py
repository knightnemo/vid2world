import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, clips_array, vfx, TextClip, CompositeVideoClip, ColorClip
import glob

def combine_all_videos(video_pairs, output_path):
    """
    Combine all GT videos in top row and all prediction videos in bottom row
    Args:
        video_pairs: list of (gt_path, pred_path) tuples
        output_path: path to save the combined video
    """
    # Load all videos
    gt_clips = []
    pred_clips = []
    
    for gt_path, pred_path in video_pairs:
        gt_clip = VideoFileClip(gt_path)
        pred_clip = VideoFileClip(pred_path)
        gt_clips.append(gt_clip)
        pred_clips.append(pred_clip)
    
    # Resize all videos to have the same height
    target_height = 320
    target_width = 320  # Adjust for RECON videos (different aspect ratio than CS:GO)
    
    # Resize and center crop videos
    def resize_and_crop(clip):
        # First resize to maintain aspect ratio
        clip = clip.fx(vfx.resize, height=target_height)
        # Then center crop to target width
        if clip.w > target_width:
            x1 = (clip.w - target_width) // 2
            clip = clip.crop(x1=x1, y1=0, x2=x1 + target_width, y2=target_height)
        return clip
    
    gt_clips = [resize_and_crop(clip) for clip in gt_clips]
    pred_clips = [resize_and_crop(clip) for clip in pred_clips]
    
    # Calculate total width needed
    spacing = 40  # pixels between videos
    label_width = 200  # width reserved for labels
    # Calculate actual width needed for videos and spacing
    video_section_width = target_width * len(gt_clips) + spacing * (len(gt_clips) - 1)
    # Total width is label width + video section width + spacing between label and videos
    total_width = label_width + spacing + video_section_width
    
    # Calculate content height (without padding)
    content_height = target_height * 2 + spacing * 3  # 2 rows of videos + spacing + labels
    
    # Add minimal padding instead of forcing 16:9 ratio
    padding_vertical = 20  # Small fixed padding instead of calculated padding
    
    # Final dimensions with minimal padding
    total_height = content_height + padding_vertical * 2
    
    # Create black background
    background = ColorClip(size=(total_width, total_height), color=(0, 0, 0))
    background = background.set_duration(max(max(clip.duration for clip in gt_clips), 
                                           max(clip.duration for clip in pred_clips)))
    
    # Create text labels
    gt_label = TextClip("GT", fontsize=28, color='white', font='Arial-Bold')
    pred_label = TextClip("Vid2World\n(Ours)", fontsize=28, color='white', font='Arial-Bold')
    
    # Position the labels vertically centered in their rows and horizontally centered in label area
    label_center_x = label_width // 2
    # Fix vertical alignment by considering spacing and padding
    gt_label = gt_label.set_position((label_center_x - gt_label.w // 2, 
                                     padding_vertical + spacing + (target_height - gt_label.h) // 2)).set_duration(background.duration)
    pred_label = pred_label.set_position((label_center_x - pred_label.w // 2, 
                                         padding_vertical + target_height + spacing * 2 + (target_height - pred_label.h) // 2)).set_duration(background.duration)
    
    # Position the videos
    clips_to_composite = [background, gt_label, pred_label]
    
    # Position GT videos in top row (with vertical padding)
    x_pos = label_width + spacing  # Add spacing after label area
    for i, clip in enumerate(gt_clips):
        clip = clip.set_position((x_pos, padding_vertical + spacing))
        clips_to_composite.append(clip)
        # Only add spacing if not the last video
        if i < len(gt_clips) - 1:
            x_pos += clip.w + spacing
        else:
            x_pos += clip.w
    
    # Position prediction videos in bottom row (with vertical padding)
    x_pos = label_width + spacing  # Add spacing after label area
    for i, clip in enumerate(pred_clips):
        clip = clip.set_position((x_pos, padding_vertical + target_height + spacing * 2))
        clips_to_composite.append(clip)
        # Only add spacing if not the last video
        if i < len(pred_clips) - 1:
            x_pos += clip.w + spacing
        else:
            x_pos += clip.w
    
    # Combine everything
    final_clip = CompositeVideoClip(clips_to_composite)
    
    # Write the result
    final_clip.write_videofile(output_path, codec='libx264')
    
    # Close all clips
    for clip in gt_clips + pred_clips + [gt_label, pred_label, final_clip]:
        clip.close()

def process_recon_directory(directory):
    """
    Process all video pairs in the RECON directory
    Args:
        directory: directory containing RECON video pairs
    """
    # Create output directory
    output_dir = os.path.join(directory, 'combined')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all GT videos
    gt_videos = glob.glob(os.path.join(directory, '*_gt.mp4'))
    gt_videos.sort()  # Sort for consistent ordering
    
    # Collect video pairs
    video_pairs = []
    
    for gt_path in gt_videos:
        # Generate corresponding prediction video path
        pred_path = gt_path.replace('_gt.mp4', '_pred.mp4')
        
        if os.path.exists(pred_path):
            video_pairs.append((gt_path, pred_path))
            print(f"Found pair: {os.path.basename(gt_path)} <-> {os.path.basename(pred_path)}")
        else:
            print(f"Warning: No corresponding prediction video found for {os.path.basename(gt_path)}")
    
    if video_pairs:
        output_path = os.path.join(output_dir, 'all_combined.mp4')
        print(f'Processing {len(video_pairs)} video pairs...')
        try:
            combine_all_videos(video_pairs, output_path)
            print(f'Combined video saved to {output_path}')
        except Exception as e:
            print(f'Error processing videos: {str(e)}')
    else:
        print('No video pairs found')

def create_grid_layout(video_pairs, output_path, grid_size=(4, 4)):
    """
    Create a grid layout for videos (alternative to side-by-side layout)
    Args:
        video_pairs: list of (gt_path, pred_path) tuples
        output_path: path to save the combined video
        grid_size: tuple of (rows, cols) for the grid
    """
    # Load all videos
    all_clips = []
    
    for gt_path, pred_path in video_pairs:
        gt_clip = VideoFileClip(gt_path)
        pred_clip = VideoFileClip(pred_path)
        all_clips.extend([gt_clip, pred_clip])
    
    # Resize all videos to have the same size
    target_height = 200
    target_width = 300
    
    def resize_and_crop(clip):
        clip = clip.fx(vfx.resize, height=target_height)
        if clip.w > target_width:
            x1 = (clip.w - target_width) // 2
            clip = clip.crop(x1=x1, y1=0, x2=x1 + target_width, y2=target_height)
        return clip
    
    all_clips = [resize_and_crop(clip) for clip in all_clips]
    
    # Create grid
    rows = []
    for i in range(0, len(all_clips), grid_size[1]):
        row_clips = all_clips[i:i + grid_size[1]]
        # Pad row if necessary
        while len(row_clips) < grid_size[1]:
            row_clips.append(ColorClip(size=(target_width, target_height), color=(0, 0, 0)))
        row = clips_array([row_clips])
        rows.append(row)
    
    # Pad rows if necessary
    while len(rows) < grid_size[0]:
        empty_row = clips_array([[ColorClip(size=(target_width, target_height), color=(0, 0, 0)) 
                                 for _ in range(grid_size[1])]])
        rows.append(empty_row)
    
    # Combine all rows
    final_clip = clips_array(rows)
    
    # Write the result
    final_clip.write_videofile(output_path, codec='libx264')
    
    # Close all clips
    for clip in all_clips + [final_clip]:
        clip.close()

if __name__ == '__main__':
    # Specify the RECON directory
    recon_dir = '.'  # Current directory (recon folder)
    process_recon_directory(recon_dir)
    
    # Optional: Create grid layout as well
    # gt_videos = glob.glob(os.path.join(recon_dir, '*_gt.mp4'))
    # video_pairs = [(gt, gt.replace('_gt.mp4', '_pred.mp4')) for gt in gt_videos if os.path.exists(gt.replace('_gt.mp4', '_pred.mp4'))]
    # create_grid_layout(video_pairs, os.path.join(recon_dir, 'combined', 'grid_layout.mp4'))
