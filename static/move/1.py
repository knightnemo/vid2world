import cv2
import numpy as np
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip, ColorClip, ImageClip
import os

def center_crop(clip, target_width=512, target_height=275):
    # Get the original dimensions
    w, h = clip.size
    
    # Calculate crop dimensions
    crop_width = min(w, int(h * target_width / target_height))
    crop_height = min(h, int(w * target_height / target_width))
    
    # Calculate crop coordinates
    x1 = (w - crop_width) // 2
    y1 = (h - crop_height) // 2
    
    # Crop the clip
    return clip.crop(x1=x1, y1=y1, width=crop_width, height=crop_height).resize((target_width, target_height))

def create_video_grid():
    # Define the video files and their corresponding actions
    videos = {
        'w': 'Forward',
        'a': 'Left',
        's': 'Backward',
        'd': 'Right',
        'up': 'Look Up',
        'down': 'Look Down',
        'l': 'Look Left',
        'r': 'Look Right'
    }
    
    # Define spacing and padding
    video_width = 512
    video_height = 275
    spacing = 20  # Space between videos
    text_height = 80  # Increased height for text label
    padding = 30  # Padding around the entire grid
    condition_width = 512  # Width of the conditioned frame (same as video width)
    font_size = 40  # Increased font size
    
    # Calculate total size
    total_width = (video_width * 4) + (spacing * 3) + (padding * 2) + condition_width + spacing
    total_height = (video_height * 2) + (spacing * 1) + (padding * 2) + (text_height * 2)
    
    # Load all video clips
    clips = []
    condition_frame = None
    
    for key, action in videos.items():
        video_path = f'pred_video_{key}.mp4'
        if os.path.exists(video_path):
            try:
                clip = VideoFileClip(video_path)
                # Get the first frame for condition if not already set
                if condition_frame is None:
                    first_frame = clip.get_frame(0)
                    # Create a clip from the first frame and apply the same cropping
                    temp_clip = ImageClip(first_frame)
                    condition_frame = center_crop(temp_clip, target_width=condition_width, target_height=video_height)
                    condition_frame = condition_frame.set_duration(clip.duration)
                
                # Center crop the clip to 275x512
                clip = center_crop(clip, target_width=video_width, target_height=video_height)
                
                # Create white background for the video and text
                bg_clip = ColorClip(size=(video_width, video_height + text_height), 
                                  color=(255, 255, 255))
                bg_clip = bg_clip.set_duration(clip.duration)
                
                # Add text label with Arial-Bold font
                txt_clip = TextClip(action, fontsize=font_size, color='black', bg_color='white', font='Arial-Bold')
                txt_clip = txt_clip.set_position(('center', video_height + 20)).set_duration(clip.duration)
                
                # Combine video and text on white background
                final_clip = CompositeVideoClip([bg_clip, clip, txt_clip])
                clips.append(final_clip)
                print(f"Successfully loaded {video_path}")
            except Exception as e:
                print(f"Error loading {video_path}: {str(e)}")
        else:
            print(f"Warning: {video_path} does not exist")
    
    if len(clips) != 8:
        print(f"Warning: Expected 8 clips but got {len(clips)}")
        return
    
    try:
        # Create white background for the entire grid
        bg_clip = ColorClip(size=(total_width, total_height), color=(255, 255, 255))
        bg_clip = bg_clip.set_duration(clips[0].duration)
        
        # Create white background for condition frame and label
        condition_bg = ColorClip(size=(condition_width, video_height + text_height), color=(255, 255, 255))
        condition_bg = condition_bg.set_duration(clips[0].duration)
        
        # Add condition label
        condition_label = TextClip("Conditioned Frame", fontsize=font_size, color='black', bg_color='white', font='Arial-Bold')
        condition_label = condition_label.set_position(('center', video_height + 20)).set_duration(clips[0].duration)
        
        # Position the condition frame
        condition_frame = condition_frame.set_position((0, 0))
        
        # Combine condition frame with its background and label
        condition_composite = CompositeVideoClip([
            condition_bg,
            condition_frame,
            condition_label
        ]).set_position((padding, padding))
        
        # Calculate positions for each clip
        positions = []
        for row in range(2):
            for col in range(4):
                x = padding + condition_width + spacing + col * (video_width + spacing)
                y = padding + row * (video_height + text_height + spacing)
                positions.append((x, y))
        
        # Create the final composition
        final_clips = [condition_composite]
        for clip, pos in zip(clips, positions):
            clip = clip.set_position(pos)
            final_clips.append(clip)
        
        # Combine all clips
        final_video = CompositeVideoClip([bg_clip] + final_clips)
        
        # Write the final video
        final_video.write_videofile('combined_video.mp4', fps=30)
    except Exception as e:
        print(f"Error creating video grid: {str(e)}")

if __name__ == "__main__":
    create_video_grid()
