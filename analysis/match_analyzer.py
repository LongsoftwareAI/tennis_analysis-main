import os
import sys

# Ensure workspace root and package dir are in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .court_geometry import CourtGeometry
    from .player_metrics import PlayerMetricsAnalyzer
    from .ball_shot_metrics import BallShotMetricsAnalyzer
    from .data_exporter import DataExporter
    from .visualizer import MatchVisualizer
except ImportError:
    from court_geometry import CourtGeometry
    from player_metrics import PlayerMetricsAnalyzer
    from ball_shot_metrics import BallShotMetricsAnalyzer
    from data_exporter import DataExporter
    from visualizer import MatchVisualizer

class MatchAnalyzer:
    """
    Advanced Post-Match Tennis Analytics & Multi-Modal Exporter Orchestrator (Facade).
    Điều phối toàn bộ quy trình trích xuất chỉ số và đồ họa sau trận:
      1. Tọa độ & hình học sân (CourtGeometry)
      2. Thông số di chuyển & thể lực cầu thủ (PlayerMetricsAnalyzer)
      3. Thông số bóng & từng cú đánh trong rally (BallShotMetricsAnalyzer)
      4. Xuất dữ liệu có cấu trúc ra đĩa JSON, CSV (DataExporter)
      5. Tạo bộ ảnh biểu đồ đồ họa cao cấp 300 DPI (MatchVisualizer)
    """
    def __init__(
        self,
        mini_court=None,
        video_fps=24.0,
        video_stem="match",
        export_dir="match_analytics",
        generate_visuals=True,
        dpi=200
    ):
        self.mini_court = mini_court
        self.video_fps = float(video_fps) if video_fps and video_fps > 0 else 24.0
        self.video_stem = str(video_stem)
        self.export_base_dir = export_dir
        self.generate_visuals = generate_visuals
        self.dpi = dpi

        # Destination directory for this match
        self.match_dir = os.path.join(self.export_base_dir, self.video_stem)
        os.makedirs(self.match_dir, exist_ok=True)

        # Initialize modular components
        self.geometry = CourtGeometry()
        self.player_analyzer = PlayerMetricsAnalyzer(self.geometry)
        self.ball_analyzer = BallShotMetricsAnalyzer(self.geometry)
        self.exporter = DataExporter()
        self.visualizer = MatchVisualizer(self.geometry)

    # Backward-compatibility delegates
    def mini_px_to_court_meters(self, px, py):
        return self.geometry.mini_px_to_court_meters(self.mini_court, px, py)

    def court_meters_to_mini_px(self, x_m, y_m):
        return self.geometry.court_meters_to_mini_px(self.mini_court, x_m, y_m)

    def analyze_and_export(
        self,
        player_detections,
        player_mini_court_detections,
        ball_detections,
        ball_mini_court_detections,
        court_keypoints,
        ball_shot_frames,
        player_stats_data_df=None,
        decision_info=None,
        all_bounces=None
    ):
        """Thực hiện toàn bộ quy trình phân tích và xuất dữ liệu ra thư mục match_analytics/<video_stem>/"""
        num_frames = len(player_mini_court_detections)
        print(f"\n[MatchAnalyzer] Starting post-match analytics for '{self.video_stem}' ({num_frames} frames)...")

        # 1. Trích xuất thông số người chơi (Quãng đường, Tốc độ, 3 Vùng sân)
        player_metrics, df_player_tracking = self.player_analyzer.compute_player_metrics(
            player_mini_court_detections=player_mini_court_detections,
            num_frames=num_frames,
            video_fps=self.video_fps,
            mini_court=self.mini_court
        )

        # 2. Trích xuất thông số bóng và nảy sân
        df_ball_tracking, bounces_list = self.ball_analyzer.compute_ball_tracking(
            ball_detections=ball_detections,
            ball_mini_court_detections=ball_mini_court_detections,
            all_bounces=all_bounces,
            decision_info=decision_info,
            num_frames=num_frames,
            video_fps=self.video_fps,
            mini_court=self.mini_court
        )

        # 3. Phân tích chi tiết từng cú đánh trong rally
        shots_df = self.ball_analyzer.compute_shot_details(
            ball_shot_frames=ball_shot_frames,
            player_mini_court_detections=player_mini_court_detections,
            ball_mini_court_detections=ball_mini_court_detections,
            bounces_list=bounces_list,
            player_stats_data_df=player_stats_data_df,
            decision_info=decision_info,
            num_frames=num_frames,
            video_fps=self.video_fps,
            mini_court=self.mini_court
        )

        # 4. Tổng hợp KPIs cấp cao cho trận đấu
        summary_kpis = self.ball_analyzer.compile_summary_kpis(
            match_name=self.video_stem,
            player_metrics=player_metrics,
            shots_df=shots_df,
            decision_info=decision_info,
            bounces_list=bounces_list,
            num_frames=num_frames,
            video_fps=self.video_fps
        )

        # 5. Lưu toàn bộ file dữ liệu (JSON, CSV)
        self.exporter.export_data_files(
            match_dir=self.match_dir,
            summary_kpis=summary_kpis,
            shots_df=shots_df,
            df_player_tracking=df_player_tracking,
            df_ball_tracking=df_ball_tracking,
            bounces_list=bounces_list
        )

        # 6. Tạo và lưu các biểu đồ đồ họa chuyên nghiệp (PNG)
        if self.generate_visuals:
            print("[MatchAnalyzer] Generating high-resolution analytical visual charts...")
            self.visualizer.generate_heatmaps(self.match_dir, df_player_tracking, dpi=self.dpi)
            self.visualizer.generate_zone_distribution_chart(self.match_dir, player_metrics, dpi=self.dpi)
            self.visualizer.generate_bounce_map(self.match_dir, bounces_list, decision_info, dpi=self.dpi)
            self.visualizer.generate_trajectory_map(self.match_dir, shots_df, bounces_list, dpi=self.dpi)
            self.visualizer.generate_speed_timeline(self.match_dir, shots_df, df_player_tracking, self.video_fps, dpi=self.dpi)
            self.visualizer.generate_summary_dashboard(self.match_dir, summary_kpis, player_metrics, shots_df, decision_info, dpi=self.dpi)

        print(f"[MatchAnalyzer] Successfully exported all analytics to: {self.match_dir}\n")
        return {
            "summary_kpis": summary_kpis,
            "export_dir": self.match_dir,
            "total_shots": len(shots_df),
            "files_generated": os.listdir(self.match_dir)
        }


# -----------------------------------------------------------------------------
# Standalone CLI entrypoint for fast extraction without re-rendering MP4
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pickle
    import numpy as np
    from mini_court import MiniCourt

    parser = argparse.ArgumentParser(description="Standalone Tennis Post-Match Analytics Exporter")
    parser.add_argument("--video", "-v", type=str, default="input_video_2", help="Video stem name (e.g. input_video_2)")
    parser.add_argument("--stubs_dir", type=str, default="tracker_stubs", help="Stubs cache directory")
    parser.add_argument("--export_dir", type=str, default="match_analytics", help="Output export folder")
    args = parser.parse_args()

    v_stem = args.video
    stubs = args.stubs_dir
    p_stub = os.path.join(stubs, f"{v_stem}_player_detections.pkl")
    c_stub = os.path.join(stubs, f"{v_stem}_court_keypoints.pkl")
    b_stub = os.path.join(stubs, f"{v_stem}_hybrid_ball_detections.pkl")
    if not os.path.exists(b_stub):
        b_stub = os.path.join(stubs, f"{v_stem}_tracknet_ball_detections.pkl")

    if not os.path.exists(p_stub) or not os.path.exists(c_stub) or not os.path.exists(b_stub):
        print(f"[Error] Required stub files not found for '{v_stem}' in '{stubs}'")
        exit(1)

    print(f"[Standalone] Loading detection caches for '{v_stem}'...")
    with open(p_stub, "rb") as f:
        player_detections = pickle.load(f)
    with open(c_stub, "rb") as f:
        court_keypoints = pickle.load(f)
    with open(b_stub, "rb") as f:
        ball_detections = pickle.load(f)

    # Ball interpolation and shot detection
    from trackers.ball_tracker import BallTracker
    b_tracker = BallTracker("models/tracknet_weights.pth", load_model=False)
    ball_detections = b_tracker.interpolate_ball_positions(ball_detections, player_positions=player_detections)
    ball_shot_frames = b_tracker.get_ball_shot_frames(ball_detections, player_positions=player_detections)

    # Mini court initialization
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mini_court = MiniCourt(dummy_frame)
    p_mini, b_mini = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints, ball_shot_frames=ball_shot_frames
    )

    # Referee Hawk-Eye ELC decision
    from utils import RefereeSystem
    referee = RefereeSystem(mini_court=mini_court, bounce_model_path="models/bounce_model.cbm")
    decision_info = referee.evaluate_point_decision(
        ball_shot_frames=ball_shot_frames,
        player_mini_court_detections=p_mini,
        ball_mini_court_detections=b_mini,
        ball_detections=ball_detections,
        court_keypoints=court_keypoints,
        mini_court=mini_court
    )

    analyzer = MatchAnalyzer(
        mini_court=mini_court,
        video_fps=24.0,
        video_stem=v_stem,
        export_dir=args.export_dir,
        generate_visuals=True
    )
    results = analyzer.analyze_and_export(
        player_detections=player_detections,
        player_mini_court_detections=p_mini,
        ball_detections=ball_detections,
        ball_mini_court_detections=b_mini,
        court_keypoints=court_keypoints,
        ball_shot_frames=ball_shot_frames,
        decision_info=decision_info,
        all_bounces=referee.all_bounces
    )
    print("Standalone analysis completed successfully!")
