import os
import sys
import re
import argparse
import cv2

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def format_seconds_to_mmss(total_seconds):
    """Format total seconds to MM:SS or HH:MM:SS string."""
    total_seconds = max(0, float(total_seconds))
    hrs = int(total_seconds // 3600)
    mins = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:04.1f}"
    return f"{mins:02d}:{secs:04.1f}"

def parse_time_str(time_str):
    """
    Parse a time string into total seconds.
    Supports formats:
      - 'MM:SS' (e.g., '1:25', '01:30.5')
      - 'HH:MM:SS' (e.g., '0:01:25', '01:10:00')
      - Pure number in seconds (e.g., '85', '85.5')
    """
    time_str = str(time_str).strip()
    if not time_str:
        return None

    if ':' in time_str:
        parts = time_str.split(':')
        try:
            if len(parts) == 2:
                mins = float(parts[0])
                secs = float(parts[1])
                return mins * 60.0 + secs
            elif len(parts) == 3:
                hrs = float(parts[0])
                mins = float(parts[1])
                secs = float(parts[2])
                return hrs * 3600.0 + mins * 60.0 + secs
        except ValueError:
            return None
    else:
        try:
            return float(time_str)
        except ValueError:
            return None

def prompt_time(label):
    """
    Prompt the user to enter time.
    Supports entering 'Phút:Giây' (e.g. 1:25) directly,
    or entering minute first, then second.
    """
    while True:
        raw = input(f"👉 Nhập thời gian {label} (Ví dụ '1:25' hoặc chỉ nhập số phút): ").strip()
        if not raw:
            print("   ⚠️ Vui lòng nhập thời gian!")
            continue

        if ':' in raw:
            val = parse_time_str(raw)
            if val is not None and val >= 0:
                return val
            print("   ⚠️ Định dạng không hợp lệ! Vui lòng nhập dạng Phút:Giây (ví dụ 1:30).")
        else:
            try:
                mins = float(raw)
                sec_raw = input(f"   ↳ Nhập số giây {label} (0 - 59, mặc định 0): ").strip()
                secs = float(sec_raw) if sec_raw else 0.0
                return mins * 60.0 + secs
            except ValueError:
                print("   ⚠️ Vui lòng nhập số hợp lệ!")

def find_default_source_video():
    """
    Automatically search for the most suitable raw video in input_videos/.
    Prioritizes full match videos in new_input/ or input_videos/.
    """
    candidates = []
    base_dirs = ["input_videos/new_input", "input_videos"]
    for b_dir in base_dirs:
        if os.path.exists(b_dir):
            for f in os.listdir(b_dir):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    full_p = os.path.join(b_dir, f)
                    if os.path.isfile(full_p):
                        size_mb = os.path.getsize(full_p) / (1024 * 1024)
                        # Avoid pre-extracted small clips if larger full matches exist
                        candidates.append((full_p, size_mb))

    if not candidates:
        return None
    # Pick largest file (source video usually > 50MB)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def cut_clip(input_video_path, output_clip_path, start_seconds, end_seconds):
    """
    Extract video segment between start_seconds and end_seconds, saving to output_clip_path.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Không tìm thấy video nguồn: {input_video_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Không thể mở video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps

    start_seconds = max(0.0, float(start_seconds))
    end_seconds = min(total_duration, float(end_seconds))

    if start_seconds >= end_seconds:
        cap.release()
        raise ValueError(f"Thời gian bắt đầu ({start_seconds:.1f}s) phải nhỏ hơn thời gian kết thúc ({end_seconds:.1f}s)!")

    start_frame = int(round(start_seconds * fps))
    end_frame = min(total_frames, int(round(end_seconds * fps)))
    num_frames = end_frame - start_frame
    duration_sec = num_frames / fps

    out_dir = os.path.dirname(output_clip_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_clip_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    bar_width = 35

    print(f"\n🎬 Đang trích xuất clip: {os.path.basename(output_clip_path)}")
    print(f"   - Từ: [{format_seconds_to_mmss(start_seconds)}] đến [{format_seconds_to_mmss(end_seconds)}]")
    print(f"   - Thời lượng clip: {duration_sec:.1f}s ({num_frames} frames @ {fps:.1f} FPS)")
    print(f"   - Độ phân giải: {width}x{height}")

    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written += 1

        pct = (written / max(1, num_frames))
        filled = int(bar_width * pct)
        bar = "=" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r   Tiến độ: [{bar}] {pct*100:.0f}% ({written}/{num_frames} frames)")
        sys.stdout.flush()

    cap.release()
    out.release()

    size_mb = os.path.getsize(output_clip_path) / (1024 * 1024) if os.path.exists(output_clip_path) else 0.0
    print(f"\n   ✅ Hoàn thành 100%! Đã lưu tại: {output_clip_path} ({size_mb:.2f} MB)\n")
    return output_clip_path

def update_config_yaml_input_path(clip_path):
    """Optionally update config.yaml with the newly extracted clip path."""
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        return False

    norm_path = os.path.normpath(clip_path).replace('\\', '/')
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        replaced = False
        for line in lines:
            if re.match(r'^\s*input_path\s*:', line) and not replaced:
                new_lines.append(f'  input_path: "{norm_path}"\n')
                replaced = True
            else:
                new_lines.append(line)

        with open(config_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    except Exception as e:
        print(f"⚠️ Không thể cập nhật config.yaml tự động: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Công cụ cắt video clip Tennis theo Phút:Giây và đặt tên file dễ dàng."
    )
    parser.add_argument("--source", "-i", type=str, default=None, help="Đường dẫn file video gốc (MP4)")
    parser.add_argument("--start", "-s", type=str, default=None, help="Thời gian bắt đầu dạng 'MM:SS' (Ví dụ: '01:25') hoặc giây")
    parser.add_argument("--end", "-e", type=str, default=None, help="Thời gian kết thúc dạng 'MM:SS' (Ví dụ: '01:45') hoặc giây")
    parser.add_argument("--start_min", type=float, default=None, help="Số phút bắt đầu (Ví dụ: 1)")
    parser.add_argument("--start_sec", type=float, default=None, help="Số giây bắt đầu (Ví dụ: 25)")
    parser.add_argument("--end_min", type=float, default=None, help="Số phút kết thúc (Ví dụ: 1)")
    parser.add_argument("--end_sec", type=float, default=None, help="Số giây kết thúc (Ví dụ: 45)")
    parser.add_argument("--name", "-n", type=str, default=None, help="Tên file clip xuất ra (Ví dụ: 'my_clip.mp4')")
    parser.add_argument("--outdir", "-o", type=str, default="input_videos/new_input/clips", help="Thư mục lưu clip")
    parser.add_argument("--update_config", action="store_true", help="Tự động cập nhật file config.yaml với clip mới")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎾 CÔNG CỤ CẮT VIDEO CLIP TENNIS TỰ ĐỘNG (TENNIS VIDEO CLIPPER)")
    print("=" * 70)

    # 1. Xác định file video nguồn
    source_video = args.source
    if not source_video:
        default_video = find_default_source_video()
        has_cli_times = (
            args.start is not None or args.end is not None or
            args.start_min is not None or args.start_sec is not None or
            args.end_min is not None or args.end_sec is not None
        )
        if default_video and (has_cli_times or not sys.stdin.isatty()):
            source_video = default_video
            print(f"📁 Video nguồn tự động chọn: '{default_video}'")
        elif default_video:
            print(f"📁 Video nguồn tự động nhận diện: '{default_video}'")
            try:
                user_src = input("👉 Nhấn Enter để dùng video này (hoặc dán đường dẫn video khác): ").strip()
                source_video = user_src if user_src else default_video
            except (EOFError, KeyboardInterrupt):
                source_video = default_video
        else:
            source_video = input("👉 Nhập đường dẫn file video gốc: ").strip()

    source_video = source_video.strip('"\'')
    if not os.path.exists(source_video):
        print(f"❌ Lỗi: File video không tồn tại: '{source_video}'")
        return

    # Lấy thông tin video nguồn
    cap = cv2.VideoCapture(source_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_s = total_f / fps if fps > 0 else 0
    cap.release()
    print(f"ℹ️  Thông tin video: {format_seconds_to_mmss(duration_s)} ({total_f:.0f} frames @ {fps:.1f} FPS)")

    # 2. Xác định thời gian bắt đầu
    if args.start_min is not None or args.start_sec is not None:
        start_sec = (args.start_min or 0.0) * 60.0 + (args.start_sec or 0.0)
    elif args.start is not None:
        start_sec = parse_time_str(args.start)
    else:
        print("\n--- [1/3] THỜI GIAN BẮT ĐẦU CLIP ---")
        start_sec = prompt_time("BẮT ĐẦU")

    # 3. Xác định thời gian kết thúc
    if args.end_min is not None or args.end_sec is not None:
        end_sec = (args.end_min or 0.0) * 60.0 + (args.end_sec or 0.0)
    elif args.end is not None:
        end_sec = parse_time_str(args.end)
    else:
        print("\n--- [2/3] THỜI GIAN KẾT THÚC CLIP ---")
        end_sec = prompt_time("KẾT THÚC")

    if start_sec >= end_sec:
        print(f"❌ Lỗi: Thời gian bắt đầu ({format_seconds_to_mmss(start_sec)}) lớn hơn hoặc bằng thời gian kết thúc ({format_seconds_to_mmss(end_sec)})!")
        return

    # 4. Xác định tên file clip
    if args.name is not None:
        clip_name = args.name.strip()
    else:
        print("\n--- [3/3] ĐẶT TÊN CHO FILE CLIP ---")
        clip_name = input("👉 Nhập tên file clip (ví dụ: clip_05_doubles): ").strip()
        if not clip_name:
            clip_name = f"clip_{int(start_sec)}s_to_{int(end_sec)}s"

    if not clip_name.lower().endswith(('.mp4', '.avi', '.mov')):
        clip_name += ".mp4"

    out_clip_path = os.path.join(args.outdir, clip_name)

    # 5. Tiến hành cắt clip
    print("\n" + "-" * 70)
    saved_path = cut_clip(source_video, out_clip_path, start_sec, end_sec)

    # 6. Hỏi người dùng có muốn tự động cập nhật vào config.yaml không
    auto_update = args.update_config
    if not auto_update and sys.stdin.isatty():
        choice = input("👉 Bạn có muốn đặt clip này làm video phân tích mặc định trong 'config.yaml' không? (Y/n): ").strip().lower()
        if choice in ('', 'y', 'yes'):
            auto_update = True

    if auto_update:
        if update_config_yaml_input_path(saved_path):
            print(f"✨ Đã tự động cập nhật 'config.yaml' -> input_path: '{saved_path.replace(chr(92), '/')}'")
            print("🚀 Bây giờ bạn chỉ cần chạy: python main.py")

    print("\n🎉 Chúc bạn thực hiện phân tích video vui vẻ!\n")

if __name__ == "__main__":
    main()
