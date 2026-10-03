import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import argparse
from copy import deepcopy
from itertools import islice
import cv2
import pandas as pd
import numpy as np
import torch

from utils import (
    iter_video_frames,
    save_video,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters,
    load_config,
    print_config_summary,
    RefereeSystem
)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import (
    CourtLineDetector,
    track_court_keypoints_cpv
)
from court_line_detector.court_segments import detect_court_segments, merge_court_frames
from mini_court import MiniCourt

def cache_is_current(stub_path, video_path, model_path):
    if not os.path.exists(stub_path):
        return False
    newest_input = os.path.getmtime(video_path)
    if model_path and os.path.exists(model_path):
        newest_input = max(newest_input, os.path.getmtime(model_path))
    return os.path.getmtime(stub_path) >= newest_input

def video_output_path(cfg, video_stem):
    output_dir = cfg["video"].get("output_dir", "output_videos")
    output_filename = cfg["video"].get("output_filename", "auto")
    if output_filename and output_filename != "auto":
        if os.path.isabs(output_filename) or os.path.dirname(output_filename):
            output_path = output_filename
        else:
            output_path = os.path.join(output_dir, output_filename)
    elif video_stem == "input_video":
        output_path = os.path.join(output_dir, "output_video.mp4")
    else:
        output_path = os.path.join(output_dir, f"{video_stem}_analysis.mp4")

    return output_path


def main(run_args=None, auto_segment=True, frame_range=None, return_frames=False):
    parser = argparse.ArgumentParser(description="Tennis Analysis Pipeline")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="Path to YAML configuration file (default: config.yaml)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Override input video path specified in config.yaml")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Override output video path or filename")
    parser.add_argument("--ball_detector", "-b", type=str, default=None, choices=["tracknet", "yolo"],
                        help="Ball detector engine: 'tracknet' (TrackNetV4 Keras) or 'yolo' (YOLO26 PyTorch)")
    parser.add_argument("--court_mode", "-m", type=str, default="cpv",
                        help="Court tracking mode: 'cpv' (Pure CPV Optical Flow Tracking)")
    parser.add_argument("--device", "-d", type=str, default=None, choices=["auto", "cuda", "cpu"],
                        help="Processing device: 'cuda', 'cpu', or 'auto' (default: from config.yaml)")
    parser.add_argument("--read_stub", action="store_true", default=None,
                        help="Force reading detections from stub cache")
    parser.add_argument("--no_stub", action="store_true",
                        help="Force live detection without reading from stub cache")
    args = parser.parse_args() if run_args is None else run_args

    # 1. Load Configuration
    cfg = load_config(args.config)
    overrides = {}

    if args.input is not None:
        cfg["video"]["input_path"] = args.input
        overrides["input"] = args.input
    if args.output is not None:
        cfg["video"]["output_filename"] = args.output
        overrides["output"] = args.output
    if args.ball_detector is not None:
        cfg["tracking"]["ball_detector"] = args.ball_detector
        overrides["ball_detector"] = args.ball_detector
    if args.court_mode is not None:
        cfg["tracking"]["court_mode"] = args.court_mode
        overrides["court_mode"] = args.court_mode
    if args.device is not None:
        cfg["tracking"]["device"] = args.device
        overrides["device"] = args.device
    if args.no_stub:
        cfg["tracking"]["force_live"] = True
        cfg["tracking"]["use_stubs"] = False
        overrides["force_live"] = True
    elif args.read_stub:
        cfg["tracking"]["use_stubs"] = True
        cfg["tracking"]["force_live"] = False
        overrides["use_stubs"] = True

    active_detector = str(cfg.get("tracking", {}).get("ball_detector", "hybrid")).lower()
    if active_detector == "yolo":
        cfg["models"]["ball_model"] = cfg.get("models", {}).get("ball_model_yolo", "models/ball_detector_yolo26_best.pt")
    else:
        cfg["models"]["ball_model"] = cfg.get("models", {}).get("ball_model_tracknet", "models/tracknet_weights.pth")

    if frame_range is None:
        print_config_summary(cfg, overrides=overrides if overrides else None)

    # 2. Read Video
    input_video_path = cfg["video"].get("input_path", "input_videos/input_video.mp4")
    if not os.path.exists(input_video_path):
        print(f"[ERROR] Input video not found at '{input_video_path}'")
        print(" [HINT] Check 'input_path' in config.yaml or provide a valid path via --input")
        return
        
    cap = cv2.VideoCapture(input_video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    range_start, range_end = frame_range if frame_range is not None else (0, total_frames)
    if range_start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, range_start)
    has_frame, first_frame = cap.read()
    cap.release()
    if not has_frame:
        raise ValueError(f"Could not read frames from '{input_video_path}'")
    if not np.isfinite(video_fps) or video_fps <= 0:
        video_fps = 24.0
        print("[Warning] Could not determine input FPS; falling back to 24 FPS")

    def video_frames():
        return iter_video_frames(input_video_path, start_frame=range_start, end_frame=range_end)

    # 3. Determine Stubs and Caching
    video_stem = os.path.splitext(os.path.basename(input_video_path))[0]
    cache_stem = video_stem if frame_range is None else f"{video_stem}_{range_start}_{range_end}"
    stubs_dir = cfg["tracking"].get("stubs_dir", "tracker_stubs")
    os.makedirs(stubs_dir, exist_ok=True)

    ball_detector_type = str(cfg.get("tracking", {}).get("ball_detector", "hybrid")).lower()
    if ball_detector_type not in ("tracknet", "yolo", "hybrid"):
        ball_detector_type = "hybrid"

    if cache_stem == "input_video":
        player_stub = os.path.join(stubs_dir, "player_detections.pkl")
        ball_stub = os.path.join(stubs_dir, f"{ball_detector_type}_ball_detections.pkl")
        court_stub = os.path.join(stubs_dir, "court_keypoints.pkl")
    else:
        player_stub = os.path.join(stubs_dir, f"{cache_stem}_player_detections.pkl")
        ball_stub = os.path.join(stubs_dir, f"{cache_stem}_{ball_detector_type}_ball_detections.pkl")
        court_stub = os.path.join(stubs_dir, f"{cache_stem}_court_keypoints.pkl")

    # 4. Configure Models & Hardware Device
    configured_device = str(cfg.get("tracking", {}).get("device", "auto")).lower()
    if configured_device == "cuda":
        if not torch.cuda.is_available():
            print("[Warning] CUDA requested in config but torch.cuda.is_available() is False. Falling back to CPU.")
            device_str = "cpu"
        else:
            device_str = "cuda"
    elif configured_device == "cpu":
        device_str = "cpu"
    else:  # "auto"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)
    models_cfg = cfg.get("models", {})

    # Player Tracker: YOLO26 Small or TF SavedModel
    player_model_path = models_cfg.get("player_model", "models/yolo26s.pt")
    if not os.path.exists(player_model_path) and os.path.exists(os.path.basename(player_model_path)):
        player_model_path = os.path.basename(player_model_path)

    # Ball Tracker: TrackNet PyTorch / Keras (with optional YOLO Hybrid) or YOLO26 PyTorch
    if ball_detector_type in ("tracknet", "hybrid"):
        ball_model_path = models_cfg.get("ball_model_tracknet", "models/tracknet_weights.pth")
        if not os.path.exists(ball_model_path):
            for cand in ["models/tracknet_weights.pth", "models/tracknet.pt", "models/tracknet_v4_ball_detector_best.keras"]:
                if os.path.exists(cand):
                    ball_model_path = cand
                    break
    else:
        ball_model_path = models_cfg.get("ball_model_yolo", "models/ball_detector_yolo26_best.pt")
        if not os.path.exists(ball_model_path) and os.path.exists("models/ball_detector_yolo26_best.pt"):
            ball_model_path = "models/ball_detector_yolo26_best.pt"

    if not os.path.exists(ball_model_path):
        generic_ball = models_cfg.get("ball_model")
        if generic_ball and os.path.exists(generic_ball):
            ball_model_path = generic_ball
            if str(ball_model_path).endswith('.keras') or 'tracknet' in str(ball_model_path).lower():
                ball_detector_type = "hybrid" if ball_detector_type == "hybrid" else "tracknet"
            else:
                ball_detector_type = "yolo"
        else:
            custom_pt = "models/ball_detector_yolo26_best.pt"
            custom_tracknet = "models/tracknet_weights.pth"
            if ball_detector_type in ("tracknet", "hybrid") and os.path.exists(custom_tracknet):
                ball_model_path = custom_tracknet
            elif os.path.exists(custom_pt):
                ball_model_path = custom_pt
            else:
                ball_model_path = player_model_path

    # Court Line Detector: TrackNet-style PyTorch heatmaps
    court_model_path = models_cfg.get("court_model", "models/model_tennis_court_det.pt")
    segment_detector = None
    if auto_segment:
        segment_detector = CourtLineDetector(court_model_path, device=device)
        segments = detect_court_segments(input_video_path, segment_detector)
        if segments != [(0, total_frames)]:
            del segment_detector
            if not segments:
                print("[Court Segments] No full-court view found; keeping all frames without overlays")

            def analyze_segment(start, end):
                print(f"[Court Segments] Analyzing frames [{start}, {end}), "
                      f"{start / video_fps:.2f}-{end / video_fps:.2f}s")
                return main(args, auto_segment=False, frame_range=(start, end), return_frames=True)

            output_path = video_output_path(cfg, video_stem)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            saved_path = save_video(
                merge_court_frames(iter_video_frames(input_video_path), segments, analyze_segment),
                output_path, fps=video_fps,
            )
            print(f"\n=== Successfully generated tennis analysis video at {saved_path} ===")
            return

    use_stubs = cfg["tracking"].get("use_stubs", True) and not cfg["tracking"].get("force_live", False)
    player_cache = use_stubs and cache_is_current(player_stub, input_video_path, player_model_path)
    ball_cache = use_stubs and cache_is_current(ball_stub, input_video_path, ball_model_path)
    court_tracker_source = os.path.join("court_line_detector", "cpv_court_tracker.py")
    court_cache = (
        use_stubs
        and cache_is_current(court_stub, input_video_path, court_model_path)
        and cache_is_current(court_stub, input_video_path, court_tracker_source)
    )
    print(f"[Pipeline] Cache for '{cache_stem}': players={player_cache}, ball={ball_cache}, court={court_cache}")
    device_display = f"GPU ({torch.cuda.get_device_name(0)})" if device_str == "cuda" else "CPU"
    print(f"[Pipeline] Active device: {device_display} (configured as '{configured_device}')")

    aux_yolo_model = models_cfg.get("ball_model_yolo", "models/ball_detector_yolo26_best.pt")
    player_tracker = PlayerTracker(model_path=player_model_path, load_model=not player_cache, device=device_str)
    ball_tracker = BallTracker(
        model_path=ball_model_path,
        model_type=ball_detector_type,
        load_model=not ball_cache,
        device=device_str,
        aux_yolo_path=aux_yolo_model,
        enable_hybrid=(ball_detector_type == "hybrid")
    )
    court_line_detector = segment_detector or CourtLineDetector(court_model_path, load_model=not court_cache, device=device)

    # 5. Detect Players and Ball
    player_detections = player_tracker.detect_frames(
        video_frames(),
        read_from_stub=player_cache,
        stub_path=player_stub
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames(),
        read_from_stub=ball_cache,
        stub_path=ball_stub,
        batch_size=cfg["tracking"].get("ball_batch_size", 4)
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # 6. Court Keypoint Extraction (TrackNet-corrected CPV Optical Flow)
    print("[CourtLineDetector] TrackNet Heatmaps + Homography Init, then Lucas-Kanade + RANSAC...")
    court_keypoints = track_court_keypoints_cpv(
        video_frames(),
        detector=court_line_detector,
        stub_path=court_stub,
        read_from_stub=court_cache
    )
    frame_count = len(player_detections)
    if frame_count != len(ball_detections) or frame_count != len(court_keypoints):
        raise ValueError("Detection caches do not match the video frame count; rerun with --no_stub")
    print(f" Loaded {frame_count} frames from '{input_video_path}'")

    # 7. Choose Players & Interpolate
    match_mode = cfg.get("tracking", {}).get("match_mode", "auto")
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections, match_mode=match_mode)
    player_detections = player_tracker.interpolate_player_positions(player_detections)

    # 8. MiniCourt Homography & Projection
    mini_court = MiniCourt(first_frame)

    # 9. Detect Ball Shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections, player_positions=player_detections)

    # Convert positions to mini court coordinates with Physics-Informed 3D Ground Projection
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, 
        ball_detections,
        court_keypoints,
        ball_shot_frames=ball_shot_frames
    )

    initial_stats = {'frame_num': 0}
    for p in (1, 2, 3, 4):
        initial_stats[f'player_{p}_number_of_shots'] = 0
        initial_stats[f'player_{p}_total_shot_speed'] = 0
        initial_stats[f'player_{p}_last_shot_speed'] = 0
        initial_stats[f'player_{p}_total_player_speed'] = 0
        initial_stats[f'player_{p}_last_player_speed'] = 0
    player_stats_data = [initial_stats]
    
    shot_frames_extended = list(ball_shot_frames)
    if shot_frames_extended and (frame_count - 1 - shot_frames_extended[-1] >= 8):
        shot_frames_extended.append(frame_count - 1)

    for ball_shot_ind in range(len(shot_frames_extended) - 1):
        start_frame = shot_frames_extended[ball_shot_ind]
        end_frame = shot_frames_extended[ball_shot_ind + 1]
        flight_duration_frames = min(video_fps, max(1, end_frame - start_frame))
        ball_shot_time_in_seconds = flight_duration_frames / video_fps

        # Safely find start and end positions of ball during the shot
        ball_start_pos = None
        for f in range(start_frame, end_frame + 1):
            if 1 in ball_mini_court_detections[f]:
                ball_start_pos = ball_mini_court_detections[f][1]
                break

        ball_end_pos = None
        for f in range(end_frame, start_frame - 1, -1):
            if 1 in ball_mini_court_detections[f]:
                ball_end_pos = ball_mini_court_detections[f][1]
                break

        if ball_start_pos is not None and ball_end_pos is not None:
            distance_covered_by_ball_pixels = measure_distance(ball_start_pos, ball_end_pos)
        else:
            distance_covered_by_ball_pixels = 0

        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = (distance_covered_by_ball_meters / max(1e-4, ball_shot_time_in_seconds)) * 3.6

        # Determine which player hit the ball
        player_positions = player_mini_court_detections[start_frame]
        ball_ref_pos = ball_start_pos if ball_start_pos is not None else (0, 0)
        if player_positions:
            player_shot_ball = min(
                player_positions.keys(),
                key=lambda p_id: measure_distance(player_positions[p_id], ball_ref_pos)
            )
        else:
            player_shot_ball = 1

        # Opponent player speed
        hitter_team = 1 if (player_shot_ball % 2 == 1) else 2
        opponent_team = 2 if hitter_team == 1 else 1
        opp_candidates = [pid for pid in player_positions if (pid % 2 == opponent_team % 2)]
        opponent_player_id = opp_candidates[0] if opp_candidates else (2 if hitter_team == 1 else 1)
        if opponent_player_id in player_mini_court_detections[start_frame] and opponent_player_id in player_mini_court_detections[end_frame]:
            distance_covered_by_opponent_pixels = measure_distance(
                player_mini_court_detections[start_frame][opponent_player_id],
                player_mini_court_detections[end_frame][opponent_player_id]
            )
            distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
                distance_covered_by_opponent_pixels,
                constants.DOUBLE_LINE_WIDTH,
                mini_court.get_width_of_mini_court()
            ) 
            speed_of_opponent = (distance_covered_by_opponent_meters / max(1e-4, ball_shot_time_in_seconds)) * 3.6
        else:
            speed_of_opponent = 0.0

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(frame_count))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    for p in (1, 2, 3, 4):
        opp = 2 if p in (1, 3) else 1
        player_stats_data_df[f'player_{p}_average_shot_speed'] = (
            player_stats_data_df[f'player_{p}_total_shot_speed'] / player_stats_data_df[f'player_{p}_number_of_shots'].replace(0, np.nan)
        ).fillna(0)
        player_stats_data_df[f'player_{p}_average_player_speed'] = (
            player_stats_data_df[f'player_{p}_total_player_speed'] / player_stats_data_df[f'player_{opp}_number_of_shots'].replace(0, np.nan)
        ).fillna(0)

    # 9.5 Referee Hawk-Eye ELC Decision Analysis (CatBoost Bounce Detection)
    bounce_model_path = cfg.get("models", {}).get("bounce_model", "models/bounce_model.cbm")
    referee_system = RefereeSystem(mini_court=mini_court, bounce_model_path=bounce_model_path)
    decision_info = referee_system.evaluate_point_decision(
        ball_shot_frames=ball_shot_frames,
        player_mini_court_detections=player_mini_court_detections,
        ball_mini_court_detections=ball_mini_court_detections,
        ball_detections=ball_detections,
        court_keypoints=court_keypoints,
        mini_court=mini_court
    )

    # Re-synchronize mini court and ball detections with referee decision info and all detected bounces
    if decision_info is not None:
        mini_court.set_referee_decision(decision_info, all_bounces=referee_system.all_bounces)
        _, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections,
            ball_detections,
            court_keypoints,
            ball_shot_frames=ball_shot_frames
        )

    # 10. Render Outputs
    print("Drawing detections and visualizations...")
    vis_cfg = cfg.get("visualization", {})
    ball_history = []
    if vis_cfg.get("draw_ball", True) and vis_cfg.get("ball_effect", "tracer") == "tracer":
        for ball_dict in ball_detections:
            box = ball_dict.get(1, [])
            if len(box) == 4 and not np.isnan(box[0]):
                ball_history.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
            else:
                ball_history.append(None)

    def render_frames():
        frame_iter = video_frames()
        for start in range(0, frame_count, 16):
            end = min(start + 16, frame_count)
            output_video_frames = list(islice(frame_iter, end - start))
            if len(output_video_frames) != end - start:
                raise ValueError("Input video ended before its detection data")

            if vis_cfg.get("draw_players", True):
                player_mode = vis_cfg.get("player_draw_mode", "ellipse")
                output_video_frames = player_tracker.draw_bboxes(
                    output_video_frames, player_detections[start:end], draw_mode=player_mode)

            if vis_cfg.get("draw_ball", True):
                ball_mode = vis_cfg.get("ball_effect", "tracer")
                output_video_frames = ball_tracker.draw_bboxes(
                    output_video_frames, ball_detections[start:end], draw_mode=ball_mode,
                    start_frame=start, ball_history=ball_history if ball_history else None)

            if vis_cfg.get("draw_court_keypoints", True):
                output_video_frames = court_line_detector.draw_keypoints_on_video(
                    output_video_frames, court_keypoints[start:end])

            if vis_cfg.get("draw_mini_court", True):
                output_video_frames = mini_court.draw_mini_court(output_video_frames, start_frame=start)
                output_video_frames = mini_court.draw_points_on_mini_court(
                    output_video_frames, player_mini_court_detections[start:end])
                output_video_frames = mini_court.draw_points_on_mini_court(
                    output_video_frames, ball_mini_court_detections[start:end],
                    color=(0, 255, 255), start_frame=start, ball_history=ball_mini_court_detections)

            if vis_cfg.get("draw_stats", True):
                output_video_frames = draw_player_stats(
                    output_video_frames, player_stats_data_df, start_frame=start)

            if vis_cfg.get("draw_referee_decision", True) and decision_info is not None:
                output_video_frames = referee_system.draw_referee_overlay(
                    output_video_frames, decision_info, start_frame=start)

            if vis_cfg.get("draw_frame_number", True):
                for i, frame in enumerate(output_video_frames, start=start):
                    cv2.putText(frame, f"Frame: {i + range_start}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            yield from output_video_frames
        if next(frame_iter, None) is not None:
            raise ValueError("Input video has more frames than its detection data")

    # 11. Determine Output Destination
    if return_frames:
        return render_frames()
    output_path = video_output_path(cfg, video_stem)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    saved_path = save_video(render_frames(), output_path, fps=video_fps)
    print(f"\n=== Successfully generated tennis analysis video at {saved_path} ===")

if __name__ == "__main__":
    main()
