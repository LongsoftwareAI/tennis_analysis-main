import os
import argparse
from copy import deepcopy
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

from utils import (
    read_video, 
    save_video,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters,
    load_config,
    print_config_summary
)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import (
    CourtLineDetector,
    detect_court_keypoints_dynamically,
    track_court_keypoints_cpv
)
from mini_court import MiniCourt

def main():
    parser = argparse.ArgumentParser(description="Tennis Analysis Pipeline")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="Path to YAML configuration file (default: config.yaml)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Override input video path specified in config.yaml")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Override output video path or filename")
    parser.add_argument("--court_mode", "-m", type=str, default=None, choices=["cpv", "dynamic", "static"],
                        help="Override court tracking mode: 'cpv' (Hybrid AI+Pure CPV), 'dynamic', or 'static'")
    parser.add_argument("--static_court", action="store_true",
                        help="Shortcut for --court_mode static")
    parser.add_argument("--read_stub", action="store_true", default=None,
                        help="Force reading detections from stub cache")
    parser.add_argument("--no_stub", action="store_true",
                        help="Force live detection without reading from stub cache")
    args = parser.parse_args()

    # 1. Load Configuration
    cfg = load_config(args.config)
    overrides = {}

    if args.input is not None:
        cfg["video"]["input_path"] = args.input
        overrides["input"] = args.input
    if args.output is not None:
        cfg["video"]["output_filename"] = args.output
        overrides["output"] = args.output
    if args.court_mode is not None:
        cfg["tracking"]["court_mode"] = args.court_mode
        overrides["court_mode"] = args.court_mode
    elif args.static_court:
        cfg["tracking"]["court_mode"] = "static"
        overrides["court_mode"] = "static"
    if args.no_stub:
        cfg["tracking"]["force_live"] = True
        cfg["tracking"]["use_stubs"] = False
        overrides["force_live"] = True
    elif args.read_stub:
        cfg["tracking"]["use_stubs"] = True
        cfg["tracking"]["force_live"] = False
        overrides["use_stubs"] = True

    print_config_summary(cfg, overrides=overrides if overrides else None)

    # 2. Read Video
    input_video_path = cfg["video"].get("input_path", "input_videos/input_video.mp4")
    if not os.path.exists(input_video_path):
        print(f"[ERROR] Input video not found at '{input_video_path}'")
        print(" [HINT] Check 'input_path' in config.yaml or provide a valid path via --input")
        return
        
    video_frames = read_video(input_video_path)
    print(f" Loaded {len(video_frames)} frames from '{input_video_path}'")

    # 3. Determine Stubs and Caching
    video_stem = os.path.splitext(os.path.basename(input_video_path))[0]
    stubs_dir = cfg["tracking"].get("stubs_dir", "tracker_stubs")
    os.makedirs(stubs_dir, exist_ok=True)

    if video_stem == "input_video":
        player_stub = os.path.join(stubs_dir, "player_detections.pkl")
        ball_stub = os.path.join(stubs_dir, "ball_detections.pkl")
        court_stub = os.path.join(stubs_dir, "court_keypoints.pkl")
    else:
        player_stub = os.path.join(stubs_dir, f"{video_stem}_player_detections.pkl")
        ball_stub = os.path.join(stubs_dir, f"{video_stem}_ball_detections.pkl")
        court_stub = os.path.join(stubs_dir, f"{video_stem}_court_keypoints.pkl")

    if cfg["tracking"].get("force_live", False):
        use_stub = False
    elif not cfg["tracking"].get("use_stubs", True):
        use_stub = False
    else:
        use_stub = os.path.exists(player_stub) and os.path.exists(ball_stub)

    if use_stub:
        print(f"[Pipeline] Using cached tracker stubs for '{video_stem}'")
    else:
        print(f"[Pipeline] Running live detections for '{video_stem}' (will cache to {player_stub})")

    # 4. Configure Models
    models_cfg = cfg.get("models", {})

    # Player Tracker: YOLO26 Small or TF SavedModel
    player_model_path = models_cfg.get("player_model", "models/yolo26s.pt")
    if not os.path.exists(player_model_path) and os.path.exists(os.path.basename(player_model_path)):
        player_model_path = os.path.basename(player_model_path)
    player_tracker = PlayerTracker(model_path=player_model_path)

    # Ball Tracker: custom trained YOLO26 or TF SavedModel/TFLite
    ball_model_path = models_cfg.get("ball_model", "models/ball_detector_yolo26_best.pt")
    if not os.path.exists(ball_model_path):
        custom_pt = "models/ball_detector_yolo26_best.pt"
        tf_saved = "models/ball_detector_tf_saved_model"
        if os.path.exists(custom_pt):
            ball_model_path = custom_pt
        elif os.path.exists(tf_saved):
            ball_model_path = tf_saved
        else:
            ball_model_path = player_model_path
    ball_tracker = BallTracker(model_path=ball_model_path)

    # Court Line Detector: ResNet50 Keras/H5
    court_model_path = models_cfg.get("court_model", "models/keypoints_model.keras")
    if not os.path.exists(court_model_path):
        court_h5 = "models/keypoints_model.h5"
        if os.path.exists(court_h5):
            court_model_path = court_h5
        else:
            court_model_path = None
    court_line_detector = CourtLineDetector(court_model_path)

    # 5. Detect Players and Ball
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=use_stub,
        stub_path=player_stub
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=use_stub,
        stub_path=ball_stub
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # 6. Court Keypoint Extraction
    mode = cfg["tracking"].get("court_mode", "cpv")
    if mode == "static":
        print("[CourtLineDetector] Mode: Static Single-Frame (Frame 0)")
        court_keypoints = court_line_detector.predict(video_frames[0])
    elif mode == "dynamic":
        print("[CourtLineDetector] Mode: Dynamic AI Per-Frame (Cách 2)")
        court_keypoints = detect_court_keypoints_dynamically(
            court_line_detector,
            video_frames,
            sample_interval=2,
            smooth_window=5,
            stub_path=court_stub if use_stub else None
        )
    else:  # mode == "cpv"
        print("[CourtLineDetector] Mode: Hybrid AI Init + Pure CPV Optical Flow Tracking (Cách 3 - Chuẩn công nghiệp)")
        court_keypoints = track_court_keypoints_cpv(
            video_frames,
            detector=court_line_detector,
            stub_path=court_stub if use_stub else None
        )

    # 7. Choose Players & Interpolate
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    player_detections = player_tracker.interpolate_player_positions(player_detections)

    # 8. MiniCourt Homography & Projection
    mini_court = MiniCourt(video_frames[0]) 

    # 9. Detect Ball Shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections, player_positions=player_detections)

    # Convert positions to mini court coordinates with Physics-Informed 3D Ground Projection
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, 
        ball_detections,
        court_keypoints,
        ball_shot_frames=ball_shot_frames
    )

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]
    
    shot_frames_extended = list(ball_shot_frames)
    if shot_frames_extended and (len(video_frames) - 1 - shot_frames_extended[-1] >= 8):
        shot_frames_extended.append(len(video_frames) - 1)

    for ball_shot_ind in range(len(shot_frames_extended) - 1):
        start_frame = shot_frames_extended[ball_shot_ind]
        end_frame = shot_frames_extended[ball_shot_ind + 1]
        flight_duration_frames = min(24, max(1, end_frame - start_frame))
        ball_shot_time_in_seconds = flight_duration_frames / 24.0  # 24 fps

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
        opponent_player_id = 1 if player_shot_ball == 2 else 2
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
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = (
        player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots'].replace(0, np.nan)
    ).fillna(0)
    player_stats_data_df['player_2_average_shot_speed'] = (
        player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots'].replace(0, np.nan)
    ).fillna(0)
    player_stats_data_df['player_1_average_player_speed'] = (
        player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots'].replace(0, np.nan)
    ).fillna(0)
    player_stats_data_df['player_2_average_player_speed'] = (
        player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots'].replace(0, np.nan)
    ).fillna(0)

    # 10. Render Outputs
    print("Drawing detections and visualizations...")
    vis_cfg = cfg.get("visualization", {})
    output_video_frames = video_frames

    if vis_cfg.get("draw_players", True):
        output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections)

    if vis_cfg.get("draw_ball", True):
        output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Court keypoints
    if vis_cfg.get("draw_court_keypoints", True):
        output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Mini court
    if vis_cfg.get("draw_mini_court", True):
        output_video_frames = mini_court.draw_mini_court(output_video_frames)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))    

    # Player stats
    if vis_cfg.get("draw_stats", True):
        output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Draw frame number
    if vis_cfg.get("draw_frame_number", True):
        for i, frame in enumerate(output_video_frames):
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 11. Determine Output Destination
    output_dir = cfg["video"].get("output_dir", "output_videos")
    output_filename = cfg["video"].get("output_filename", "auto")

    if output_filename and output_filename != "auto":
        if os.path.isabs(output_filename) or os.path.dirname(output_filename) != "":
            output_path = output_filename
        else:
            output_path = os.path.join(output_dir, output_filename)
    elif video_stem == "input_video":
        output_path = os.path.join(output_dir, "output_video.avi")
    else:
        output_path = os.path.join(output_dir, f"{video_stem}_analysis.avi")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    saved_path = save_video(output_video_frames, output_path)
    print(f"\n=== Successfully generated tennis analysis video at {saved_path} ===")

if __name__ == "__main__":
    main()