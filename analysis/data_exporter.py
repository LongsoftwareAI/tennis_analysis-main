import os
import json
import pandas as pd

class DataExporter:
    """
    Quản lý lưu trữ và xuất toàn bộ dữ liệu có cấu trúc ra đĩa (JSON, CSV):
    - summary_kpis.json
    - shots_detail.csv
    - player_tracking.csv
    - ball_tracking.csv
    - bounces.json
    """
    def export_data_files(
        self,
        match_dir,
        summary_kpis,
        shots_df,
        df_player_tracking,
        df_ball_tracking,
        bounces_list
    ):
        os.makedirs(match_dir, exist_ok=True)

        # 1. summary_kpis.json
        kpi_path = os.path.join(match_dir, "summary_kpis.json")
        with open(kpi_path, "w", encoding="utf-8") as f:
            json.dump(summary_kpis, f, indent=2, ensure_ascii=False)

        # 2. shots_detail.csv
        if not shots_df.empty:
            shots_path = os.path.join(match_dir, "shots_detail.csv")
            shots_df.to_csv(shots_path, index=False, encoding="utf-8-sig")

        # 3. player_tracking.csv
        if not df_player_tracking.empty:
            player_track_path = os.path.join(match_dir, "player_tracking.csv")
            df_player_tracking.to_csv(player_track_path, index=False, encoding="utf-8-sig")

        # 4. ball_tracking.csv
        if not df_ball_tracking.empty:
            ball_track_path = os.path.join(match_dir, "ball_tracking.csv")
            df_ball_tracking.to_csv(ball_track_path, index=False, encoding="utf-8-sig")

        # 5. bounces.json
        bounces_path = os.path.join(match_dir, "bounces.json")
        with open(bounces_path, "w", encoding="utf-8") as f:
            json.dump(bounces_list, f, indent=2, ensure_ascii=False)
