import os
import sys
import cv2

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def cut_clip(input_video_path, output_clip_path, start_frame, end_frame):
    """
    Extract a clip from start_frame to end_frame and save as MP4.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
        
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)
    num_frames = end_frame - start_frame
    
    print(f"\n[VideoCutter] Extracting '{os.path.basename(output_clip_path)}'...")
    print(f" -> Frames: {start_frame} to {end_frame} ({num_frames} frames, {num_frames/fps:.1f}s)")
    
    os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_clip_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for f in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written += 1
        if written % 100 == 0 or written == num_frames:
            print(f"   Progress: {written}/{num_frames} frames ({(written/num_frames)*100:.0f}%)", end='\r')
            
    cap.release()
    out.release()
    size_mb = os.path.getsize(output_clip_path) / (1024 * 1024)
    print(f"\n -> Done! Saved to: {output_clip_path} ({size_mb:.2f} MB)")
    return output_clip_path

def main():
    raw_video = r"D:\FPT\DAT301m\tennis_analysis-main\input_videos\new_input\YTSave_YouTube_Media_zYHXez1M75k_Những-pha-bóng-tennis-hay-nhất-tại-Australian-Open-2016-DancoSport-com_001_1080p.mp4"
    output_dir = r"D:\FPT\DAT301m\tennis_analysis-main\input_videos\new_input\clips"
    
    # Pre-defined high-quality rallies from Australian Open 2016
    clips_config = [
        {
            "name": "clip_01_nadal_vs_verdasco_fast.mp4",
            "start": 95,
            "end": 490,
            "desc": "Rally 1 Full Point (395 frames ~15.8s) - Trọn vẹn tình huống từ giao bóng, cứu bóng ngoài sân đến ghi điểm kết thúc"
        },
        {
            "name": "clip_01_nadal_vs_verdasco_full.mp4",
            "start": 85,
            "end": 495,
            "desc": "Rally 1 Full (410 frames ~16.4s) - Trọn vẹn pha bóng đỉnh cao Nadal vs Verdasco"
        },
        {
            "name": "clip_02_zverev_vs_murray.mp4",
            "start": 620,
            "end": 955,
            "desc": "Rally 2 Full (335 frames ~13.4s) - Zverev giao bóng lên lưới & Murray phản công"
        },
        {
            "name": "clip_03_duckworth_vs_hewitt.mp4",
            "start": 1060,
            "end": 1375,
            "desc": "Rally 3 Full (315 frames ~12.6s) - Duckworth vs Lleyton Hewitt"
        },
        {
            "name": "clip_04_monfils_rally.mp4",
            "start": 2880,
            "end": 3240,
            "desc": "Rally 4 Full (360 frames ~14.4s) - Pha bóng tốc độ cao của Gael Monfils"
        }
    ]
    
    print("=== TENNIS VIDEO CLIP EXTRACTOR ===")
    print(f"Source: {raw_video}")
    print(f"Target directory: {output_dir}")
    
    for item in clips_config:
        out_path = os.path.join(output_dir, item["name"])
        print(f"\nProcessing: {item['desc']}")
        cut_clip(raw_video, out_path, item["start"], item["end"])
        
    print("\n=== ALL CLIPS SUCCESSFULLY EXTRACTED ===")

if __name__ == "__main__":
    main()
