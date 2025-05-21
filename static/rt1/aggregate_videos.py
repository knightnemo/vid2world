import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, clips_array, vfx, TextClip, CompositeVideoClip, ColorClip

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
    target_height = max(max(clip.h for clip in gt_clips), max(clip.h for clip in pred_clips))
    target_width = 400  # Center crop width
    
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
    spacing = 50  # pixels between videos (increased from 20 to 50)
    label_width = 250  # width reserved for labels (increased from 150 to 250)
    total_width = target_width * len(gt_clips) + spacing * (len(gt_clips) - 1) + label_width + 50
    
    # Calculate content height (without padding)
    content_height = target_height * 2 + spacing * 3  # 2 rows of videos + spacing + labels
    
    # Calculate padding needed for Twitter's 16:9 aspect ratio
    # For 16:9 ratio: height = width * 9/16
    twitter_height = int(total_width * 9 / 16)
    
    # If content is taller than Twitter height, we'll need to adjust width instead
    if content_height > twitter_height:
        # Recalculate for width based on content height
        twitter_width = int(content_height * 16 / 9)
        # Add padding to width if needed
        if twitter_width > total_width:
            total_width = twitter_width
            padding_vertical = 0
        else:
            padding_vertical = (twitter_height - content_height) // 2
    else:
        # Add padding to height
        padding_vertical = (twitter_height - content_height) // 2
    
    # Final dimensions with padding
    total_height = content_height + padding_vertical * 2
    
    # Create black background
    background = ColorClip(size=(total_width, total_height), color=(0, 0, 0))
    background = background.set_duration(max(max(clip.duration for clip in gt_clips), 
                                           max(clip.duration for clip in pred_clips)))
    
    # Create text labels
    gt_label = TextClip("GT", fontsize=30, color='white', font='Arial-Bold')
    pred_label = TextClip("Vid2World\n(Ours)", fontsize=30, color='white', font='Arial-Bold')
    
    # Position the labels vertically centered in their rows and horizontally centered in label area
    label_center_x = label_width // 2
    # Adjust vertical positions with padding
    gt_label = gt_label.set_position((label_center_x - gt_label.w // 2, 
                                     padding_vertical + (target_height - gt_label.h) // 2)).set_duration(background.duration)
    pred_label = pred_label.set_position((label_center_x - pred_label.w // 2, 
                                         padding_vertical + target_height + spacing * 2 + (target_height - pred_label.h) // 2)).set_duration(background.duration)
    
    # Position the videos
    clips_to_composite = [background, gt_label, pred_label]
    
    # Position GT videos in top row (with vertical padding)
    x_pos = label_width
    for clip in gt_clips:
        clip = clip.set_position((x_pos, padding_vertical + spacing))
        clips_to_composite.append(clip)
        x_pos += clip.w + spacing
    
    # Position prediction videos in bottom row (with vertical padding)
    x_pos = label_width
    for clip in pred_clips:
        clip = clip.set_position((x_pos, padding_vertical + target_height + spacing * 2))
        clips_to_composite.append(clip)
        x_pos += clip.w + spacing
    
    # Combine everything
    final_clip = CompositeVideoClip(clips_to_composite)
    
    # Write the result
    final_clip.write_videofile(output_path, codec='libx264')
    
    # Close all clips
    for clip in gt_clips + pred_clips + [gt_label, pred_label, final_clip]:
        clip.close()

def process_directory(directory):
    """
    Process all video pairs in the directory and its subdirectories
    Args:
        directory: directory containing video pairs
    """
    # Create output directory
    output_dir = os.path.join(directory, 'combined')
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all video pairs
    video_pairs = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        # Skip the combined directory
        if 'combined' in root:
            continue
            
        # Find all prediction videos in current directory
        pred_videos = [f for f in files if f == 'pred_video.mp4']
        
        for pred_video in pred_videos:
            # Get corresponding ground truth video
            gt_video = 'gt_video.mp4'
            
            if gt_video in files:
                pred_path = os.path.join(root, pred_video)
                gt_path = os.path.join(root, gt_video)
                video_pairs.append((gt_path, pred_path))
    
    if video_pairs:
        output_path = os.path.join(output_dir, 'all_combined_2.mp4')
        print(f'Processing {len(video_pairs)} video pairs...')
        try:
            combine_all_videos(video_pairs, output_path)
            print(f'Combined video saved to {output_path}')
        except Exception as e:
            print(f'Error processing videos: {str(e)}')
    else:
        print('No video pairs found')

if __name__ == '__main__':
    # Specify the directory containing the videos
    video_dir = '.'  # Changed to the parent directory
    process_directory(video_dir)
