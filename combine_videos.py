import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm  # 用于显示进度条

def center_crop(frame, target_height, target_width):
    """Center crop the frame to target size"""
    h, w = frame.shape[:2]
    start_x = (w - target_width) // 2
    start_y = (h - target_height) // 2
    return frame[start_y:start_y+target_height, start_x:start_x+target_width]

def process_frame(frame, is_csgo):
    """Process frame based on video type and adjust to 3:4 aspect ratio"""
    if is_csgo:
        # For CS:GO videos: crop to 275x512 (HxW)
        frame = center_crop(frame, 275, 512)
    else:
        # For RT-1 videos: crop to 320x400 (HxW)
        frame = center_crop(frame, 320, 400)
    
    # 将所有帧调整为统一的3:4尺寸（宽:高 = 3:4）
    target_height = 320  # 使用固定的高度
    target_width = int(target_height * 4 / 3)  # 根据3:4比例计算宽度
    frame = cv2.resize(frame, (target_width, target_height))
    return frame

def get_video_info(video_path):
    """Get video information including width, height, fps and total frames"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, total_frames

def create_video_grid_with_empty_slots(video_paths, output_path, grid_size=(4, 4)):
    """
    Create a grid of videos with some empty slots and save as a single video
    
    Args:
        video_paths: List of paths to input videos (empty string for empty slots)
        output_path: Path to save the output video
        grid_size: Tuple of (rows, cols) for the grid
    """
    # 获取一个有效视频的信息
    first_valid_path = next(path for path in video_paths if path)
    _, _, fps, total_frames = get_video_info(first_valid_path)
    
    # 处理一帧来获取尺寸
    cap = cv2.VideoCapture(first_valid_path)
    ret, frame = cap.read()
    cap.release()
    
    # 处理帧并设置统一尺寸
    is_csgo = "csgo" in first_valid_path
    processed_frame = process_frame(frame, is_csgo)
    height, width = processed_frame.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                         (width * grid_size[1], height * grid_size[0]))
    
    # 创建视频捕获对象和类型列表
    caps = []
    is_csgo_list = []
    for path in video_paths:
        if path:
            caps.append(cv2.VideoCapture(path))
            is_csgo_list.append("csgo" in path)
        else:
            caps.append(None)
            is_csgo_list.append(False)
    
    # 创建进度条
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    for _ in range(total_frames):
        frames = []
        for cap, is_csgo, path in zip(caps, is_csgo_list, video_paths):
            if not path:  # 空位置
                empty_frame = np.zeros((height, width, 3), dtype=np.uint8)
                frames.append(empty_frame)
            else:
                ret, frame = cap.read()
                if not ret:
                    # 视频结束，从头开始
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                # 根据视频类型处理帧
                frame = process_frame(frame, is_csgo)
                frames.append(frame)
        
        # 创建网格
        rows = []
        for i in range(0, len(frames), grid_size[1]):
            row_frames = frames[i:i + grid_size[1]]
            # 如果行不够，填充
            while len(row_frames) < grid_size[1]:
                row_frames.append(np.zeros((height, width, 3), dtype=np.uint8))
            row = np.hstack(row_frames)
            rows.append(row)
        
        # 如果行数不够，填充
        while len(rows) < grid_size[0]:
            rows.append(np.zeros((height, width * grid_size[1], 3), dtype=np.uint8))
        
        # 合并行
        grid = np.vstack(rows)
        
        # 写入帧
        out.write(grid)
        
        # 更新进度条
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    # 释放资源
    for cap in caps:
        if cap is not None:
            cap.release()
    out.release()

def main():
    # 获取视频路径
    csgo_videos = []
    appendix_videos = []
    
    # 获取CS:GO视频
    csgo_dir = Path("static/csgo")
    for video_path in csgo_dir.glob("hdf5_dm_july2021_*/pred_video.mp4"):
        csgo_videos.append(str(video_path))
    
    # 获取appendix视频
    appendix_dir = Path("static/appendix")
    for video_path in appendix_dir.glob("*/pred_video.mp4"):
        appendix_videos.append(str(video_path))
    
    # 排序确保顺序一致
    csgo_videos.sort()
    appendix_videos.sort()
    
    # 只保留前4个CS:GO视频
    csgo_videos = csgo_videos[:4]
    
    # 创建特定布局的视频路径列表
    video_paths = []
    
    # 第1行：4个appendix视频
    video_paths.extend(appendix_videos[:4])
    
    # 第2行：第一个和最后一个位置放CS:GO视频，中间2个空出来
    video_paths.append(csgo_videos[0])  # 位置(1,0)
    video_paths.append("")               # 空位置(1,1)
    video_paths.append("")               # 空位置(1,2)
    video_paths.append(csgo_videos[1])   # 位置(1,3)
    
    # 第3行：第一个和最后一个位置放CS:GO视频，中间2个空出来
    video_paths.append(csgo_videos[2])   # 位置(2,0)
    video_paths.append("")               # 空位置(2,1)
    video_paths.append("")               # 空位置(2,2)
    video_paths.append(csgo_videos[3])   # 位置(2,3)
    
    # 第4行：4个appendix视频
    remaining_appendix = appendix_videos[4:]
    while len(remaining_appendix) < 4:
        remaining_appendix.append("")  # 如果视频不够，用空位置填充
    video_paths.extend(remaining_appendix)
    
    # 打印找到的视频列表以进行调试
    print(f"找到{len(video_paths)}个位置，其中{len([p for p in video_paths if p])}个有视频:")
    for i, path in enumerate(video_paths):
        row = i // 4 + 1
        col = i % 4 + 1
        status = "有视频" if path else "空位置"
        print(f"位置({row},{col}): {status} {path}")
    
    # 创建输出目录
    output_dir = Path("static/combined")
    output_dir.mkdir(exist_ok=True)
    
    # 修改create_video_grid函数调用
    create_video_grid_with_empty_slots(
        video_paths=video_paths,
        output_path=str(output_dir / "grid_video.mp4"),
        grid_size=(4, 4)  # 4x4 grid
    )

if __name__ == "__main__":
    main() 