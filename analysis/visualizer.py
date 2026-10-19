import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter

class MatchVisualizer:
    """
    Hệ thống vẽ và xuất các biểu đồ đồ họa chuyên nghiệp (300 DPI):
    - Bản đồ nhiệt di chuyển cầu thủ (Player Movement Heatmaps 2D)
    - Biểu đồ phân bổ % thời gian tại 3 vùng sân chiến thuật
    - Bản đồ phân bố điểm bóng nảy sân Hawk-Eye ELC (Bounce Dispersion)
    - Vector quỹ đạo đường bóng bay đa chiều (2D Shot Trajectories)
    - Biểu đồ tiến trình vận tốc bóng và tốc độ chạy nước rút (Timeline)
    - Thẻ Infographic tổng quan truyền hình (Broadcast Match Summary Card)
    """
    def __init__(self, geometry):
        self.geom = geometry

    def draw_court_canvas(self, ax, title="Tennis Court 2D Analytics", show_zones=False):
        """Vẽ bản đồ sân tennis 2D chuyên nghiệp chuẩn ATP với tọa độ mét thực tế."""
        ax.set_facecolor("#0b0f19")

        # Court Surround Boundary
        surround_rect = patches.Rectangle(
            (-6.8, -14.2), 13.6, 28.4,
            facecolor="#0f172a", edgecolor="#1e293b", linewidth=1.5, zorder=1
        )
        ax.add_patch(surround_rect)

        # Main Court Surface (US Open Midnight Blue)
        court_surface = patches.Rectangle(
            (-self.geom.court_width_doubles / 2.0, -self.geom.court_length / 2.0),
            self.geom.court_width_doubles, self.geom.court_length,
            facecolor="#172554", edgecolor="#ffffff", linewidth=2.5, zorder=2
        )
        ax.add_patch(court_surface)

        # Singles Court Surface (Lighter Royal Blue)
        singles_surface = patches.Rectangle(
            (-self.geom.court_width_singles / 2.0, -self.geom.court_length / 2.0),
            self.geom.court_width_singles, self.geom.court_length,
            facecolor="#1e3a8a", edgecolor="#ffffff", linewidth=2.0, zorder=3
        )
        ax.add_patch(singles_surface)

        # Service Boxes
        sw = self.geom.court_width_singles / 2.0
        # Service Lines (-6.4m and +6.4m)
        ax.plot([-sw, sw], [-self.geom.service_line_dist, -self.geom.service_line_dist], color="#ffffff", lw=2.0, zorder=4)
        ax.plot([-sw, sw], [self.geom.service_line_dist, self.geom.service_line_dist], color="#ffffff", lw=2.0, zorder=4)

        # Center Service Line
        ax.plot([0, 0], [-self.geom.service_line_dist, self.geom.service_line_dist], color="#ffffff", lw=2.0, zorder=4)

        # Center Baseline Marks (0.35m length)
        ax.plot([0, 0], [-self.geom.baseline_dist, -self.geom.baseline_dist + 0.35], color="#ffffff", lw=2.5, zorder=4)
        ax.plot([0, 0], [self.geom.baseline_dist, self.geom.baseline_dist - 0.35], color="#ffffff", lw=2.5, zorder=4)

        # NET Line (at Y=0, spanning across alley)
        net_w = self.geom.court_width_doubles / 2.0 + 0.7
        ax.plot([-net_w, net_w], [0, 0], color="#94a3b8", lw=3.5, linestyle="--", zorder=5)
        ax.plot([-net_w, -net_w], [-0.2, 0.2], color="#cbd5e1", lw=4.0, zorder=5)
        ax.plot([net_w, net_w], [-0.2, 0.2], color="#cbd5e1", lw=4.0, zorder=5)

        # Optional Tactical Zone Delineations
        if show_zones:
            ax.plot([-sw, sw], [-self.geom.deep_zone_dist, -self.geom.deep_zone_dist], color="#f59e0b", lw=1.2, linestyle=":", zorder=4, alpha=0.7)
            ax.plot([-sw, sw], [self.geom.deep_zone_dist, self.geom.deep_zone_dist], color="#06b6d4", lw=1.2, linestyle=":", zorder=4, alpha=0.7)

        # Limits & Labels
        ax.set_xlim(-7.2, 7.2)
        ax.set_ylim(-14.8, 14.8)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight="bold", color="#f8fafc", pad=15)
        ax.tick_params(colors="#64748b", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        ax.text(0, -13.6, "FAR COURT (PLAYER 2)", color="#94a3b8", fontsize=9, ha="center", fontweight="semibold")
        ax.text(0, 13.6, "NEAR COURT (PLAYER 1)", color="#94a3b8", fontsize=9, ha="center", fontweight="semibold")
        ax.text(net_w + 0.2, 0, "NET", color="#cbd5e1", fontsize=8, va="center")

    def generate_heatmaps(self, match_dir, df_player_tracking, dpi=200):
        """Tạo 3 bản đồ nhiệt: Player 1, Player 2 và Combined Heatmap."""
        grid_x = np.linspace(-6.5, 6.5, 130)
        grid_y = np.linspace(-14.0, 14.0, 280)

        # 1. Player 1 Heatmap (Cyan Theme)
        fig, ax = plt.subplots(figsize=(7, 12), facecolor="#0b0f19")
        self.draw_court_canvas(ax, title="Player 1 - Court Coverage Heatmap", show_zones=True)

        p1_x = df_player_tracking["p1_x_m"].values
        p1_y = df_player_tracking["p1_y_m"].values
        p1_hist, _, _ = np.histogram2d(p1_x, p1_y, bins=[grid_x, grid_y])
        p1_smooth = gaussian_filter(p1_hist, sigma=3.2)
        p1_smooth /= max(1e-5, p1_smooth.max())

        ax.contourf(
            grid_x[:-1], grid_y[:-1], p1_smooth.T,
            levels=np.linspace(0.12, 1.0, 14),
            cmap="YlGnBu", alpha=0.72, zorder=6
        )
        ax.scatter(p1_x[::3], p1_y[::3], color="#22d3ee", s=8, alpha=0.35, zorder=7)

        fig.tight_layout()
        fig.savefig(os.path.join(match_dir, "heatmap_player_1.png"), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        # 2. Player 2 Heatmap (Amber/Orange Theme)
        fig, ax = plt.subplots(figsize=(7, 12), facecolor="#0b0f19")
        self.draw_court_canvas(ax, title="Player 2 - Court Coverage Heatmap", show_zones=True)

        p2_x = df_player_tracking["p2_x_m"].values
        p2_y = df_player_tracking["p2_y_m"].values
        p2_hist, _, _ = np.histogram2d(p2_x, p2_y, bins=[grid_x, grid_y])
        p2_smooth = gaussian_filter(p2_hist, sigma=3.2)
        p2_smooth /= max(1e-5, p2_smooth.max())

        ax.contourf(
            grid_x[:-1], grid_y[:-1], p2_smooth.T,
            levels=np.linspace(0.12, 1.0, 14),
            cmap="YlOrRd", alpha=0.72, zorder=6
        )
        ax.scatter(p2_x[::3], p2_y[::3], color="#fbbf24", s=8, alpha=0.35, zorder=7)

        fig.tight_layout()
        fig.savefig(os.path.join(match_dir, "heatmap_player_2.png"), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        # 3. Combined Dual Heatmap
        fig, ax = plt.subplots(figsize=(7, 12), facecolor="#0b0f19")
        self.draw_court_canvas(ax, title="Dual Court Coverage Heatmap (P1 vs P2)", show_zones=True)

        ax.contourf(
            grid_x[:-1], grid_y[:-1], p1_smooth.T,
            levels=np.linspace(0.15, 1.0, 10),
            cmap="YlGnBu", alpha=0.65, zorder=6
        )
        ax.contourf(
            grid_x[:-1], grid_y[:-1], p2_smooth.T,
            levels=np.linspace(0.15, 1.0, 10),
            cmap="YlOrRd", alpha=0.65, zorder=6
        )

        ax.plot([], [], color="#06b6d4", lw=6, label="Player 1 (Near Court)")
        ax.plot([], [], color="#f59e0b", lw=6, label="Player 2 (Far Court)")
        legend = ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", fontsize=9)
        for text in legend.get_texts():
            text.set_color("#f8fafc")

        fig.tight_layout()
        fig.savefig(os.path.join(match_dir, "heatmap_combined.png"), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    def generate_zone_distribution_chart(self, match_dir, player_metrics, dpi=200):
        """Vẽ biểu đồ phân bổ % thời gian tại 3 vùng sân chiến thuật."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0b0f19")

        categories = ["Baseline Zone\n(> 1.5m Day san)", "No-Man's Land\n(Vung cam)", "Attack Zone\n(Tan cong luoi)"]
        p1_vals = [
            player_metrics[1]["zone_baseline_pct"],
            player_metrics[1]["zone_nomans_land_pct"],
            player_metrics[1]["zone_attack_pct"]
        ]
        p2_vals = [
            player_metrics[2]["zone_baseline_pct"],
            player_metrics[2]["zone_nomans_land_pct"],
            player_metrics[2]["zone_attack_pct"]
        ]

        # Player 1 Bar
        colors_p1 = ["#0284c7", "#0ea5e9", "#38bdf8"]
        bars1 = ax1.bar(categories, p1_vals, color=colors_p1, width=0.55, edgecolor="#f8fafc", linewidth=1.2)
        ax1.set_facecolor("#131926")
        ax1.set_title("Player 1 - Tactical Court Zones", fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("% Rally Duration", color="#94a3b8", fontsize=10)
        ax1.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax1.spines.values():
            spine.set_color("#334155")
        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", color="#f8fafc", fontweight="bold", fontsize=10)

        # Player 2 Bar
        colors_p2 = ["#d97706", "#f59e0b", "#fbbf24"]
        bars2 = ax2.bar(categories, p2_vals, color=colors_p2, width=0.55, edgecolor="#f8fafc", linewidth=1.2)
        ax2.set_facecolor("#131926")
        ax2.set_title("Player 2 - Tactical Court Zones", fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax2.spines.values():
            spine.set_color("#334155")
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", color="#f8fafc", fontweight="bold", fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(match_dir, "court_zones_distribution.png"), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    def generate_bounce_map(self, match_dir, bounces_list, decision_info, dpi=200):
        """Vẽ bản đồ phân bố các điểm bóng nảy sân chuẩn Hawk-Eye (IN xanh, OUT đỏ)."""
        fig, ax = plt.subplots(figsize=(7, 12), facecolor="#0b0f19")
        self.draw_court_canvas(ax, title="Hawk-Eye Ball Bounce Dispersion Map", show_zones=True)

        for b in bounces_list:
            bx = b["x_m"]
            by = b["y_m"]
            is_in = b["is_in"]
            is_final = b["is_final"]
            s_idx = b["shot_index"]
            margin = b["margin_cm"]

            color = "#22c55e" if is_in else "#ef4444"
            edge = "#ffffff" if is_final else "#1e293b"
            size = 140 if is_final else 80

            ax.scatter(bx, by, s=size, color=color, edgecolors=edge, linewidth=2.0, zorder=8)
            ax.scatter(bx, by, s=size * 2.5, color=color, alpha=0.25, zorder=7)

            label_txt = f"#{s_idx} ({'+' if is_in else '-'}{margin:.0f}cm)"
            y_offset = 0.55 if by > 0 else -0.55
            ax.text(bx, by + y_offset, label_txt, color="#f8fafc", fontsize=8, ha="center",
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e293b", edgecolor=color, alpha=0.85), zorder=9)

        ax.scatter([], [], color="#22c55e", s=80, label="IN (Bong trong san)")
        ax.scatter([], [], color="#ef4444", s=80, label="OUT (Bong ngoai san)")
        legend = ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", fontsize=9)
        for text in legend.get_texts():
            text.set_color("#f8fafc")

        fig.tight_layout()
        fig.savefig(os.path.join(match_dir, "ball_bounce_dispersion.png"), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    def generate_trajectory_map(self, match_dir, shots_df, bounces_list, dpi=200):
        """Vẽ toàn bộ vector quỹ đạo đường bay 2D của các cú đánh."""
        fig, ax = plt.subplots(figsize=(7, 12), facecolor="#0b0f19")
        self.draw_court_canvas(ax, title="2D Ball Trajectory Shot Vectors", show_zones=False)

        if not shots_df.empty:
            for _, shot in shots_df.iterrows():
                hx = shot["hit_x_m"]
                hy = shot["hit_y_m"]
                bx = shot["bounce_x_m"]
                by = shot["bounce_y_m"]
                speed = shot["speed_kmh"]
                s_idx = shot["shot_index"]

                if speed >= 120:
                    line_color = "#ef4444"
                elif speed >= 95:
                    line_color = "#f59e0b"
                elif speed >= 75:
                    line_color = "#22c55e"
                else:
                    line_color = "#06b6d4"

                ax.annotate(
                    "", xy=(bx, by), xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="->", color=line_color, lw=2.2, mutation_scale=16),
                    zorder=7
                )
                ax.scatter(hx, hy, color="#ffffff", s=35, edgecolors=line_color, linewidth=1.5, zorder=8)
                ax.text((hx + bx) / 2.0, (hy + by) / 2.0, f"#{s_idx} ({speed:.0f} km/h)",
                        color="#f8fafc", fontsize=7.5, ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#0f172a", edgecolor=line_color, alpha=0.85), zorder=9)

        ax.plot([], [], color="#ef4444", lw=2.5, label="High Speed (>= 120 km/h)")
        ax.plot([], [], color="#f59e0b", lw=2.5, label="Fast Drive (95-120 km/h)")
        ax.plot([], [], color="#22c55e", lw=2.5, label="Medium Rally (75-95 km/h)")
        ax.plot([], [], color="#06b6d4", lw=2.5, label="Lob / Touch (< 75 km/h)")
        legend = ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", fontsize=8.5)
        for text in legend.get_texts():
            text.set_color("#f8fafc")

        fig.tight_layout()
        fig.savefig(os.path.join(match_dir, "ball_trajectories_2d.png"), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    def generate_speed_timeline(self, match_dir, shots_df, df_player_tracking, video_fps, dpi=200):
        """Vẽ biểu đồ tiến trình vận tốc bóng và tốc độ chạy nước rút của 2 cầu thủ."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), facecolor="#0b0f19", sharex=True)

        time_s = df_player_tracking["time_s"].values

        # Subplot 1: Ball Shot Speeds
        ax1.set_facecolor("#131926")
        ax1.set_title("Ball Shot Velocity Timeline", fontsize=12, fontweight="bold", color="#f8fafc", pad=10)
        if not shots_df.empty:
            shot_times = [s["hit_frame"] / video_fps for _, s in shots_df.iterrows()]
            shot_speeds = shots_df["speed_kmh"].values
            colors = ["#06b6d4" if h == "Player 1" else "#f59e0b" for h in shots_df["hitter"]]

            markerline, stemlines, baseline = ax1.stem(shot_times, shot_speeds, linefmt="#64748b", markerfmt="o", basefmt=" ")
            plt.setp(markerline, color="#ffffff", markersize=8, markeredgewidth=2)
            for st, ss, c, idx in zip(shot_times, shot_speeds, colors, shots_df["shot_index"]):
                ax1.scatter([st], [ss], color=c, s=120, edgecolors="#ffffff", zorder=5)
                ax1.text(st, ss + 5, f"#{idx}: {ss:.0f}", color="#f8fafc", fontsize=8.5, ha="center", fontweight="bold")

        ax1.set_ylabel("Speed (km/h)", color="#94a3b8", fontsize=10)
        ax1.set_ylim(40, 180)
        ax1.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax1.spines.values():
            spine.set_color("#334155")

        # Subplot 2: Player Sprint Velocities
        ax2.set_facecolor("#131926")
        ax2.set_title("Player Movement Velocities (Running Speed)", fontsize=12, fontweight="bold", color="#f8fafc", pad=10)
        ax2.plot(time_s, df_player_tracking["p1_speed_kmh"], color="#06b6d4", lw=2.2, label="Player 1 Speed")
        ax2.plot(time_s, df_player_tracking["p2_speed_kmh"], color="#f59e0b", lw=2.2, label="Player 2 Speed")

        ax2.set_xlabel("Rally Time (seconds)", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Sprint Speed (km/h)", color="#94a3b8", fontsize=10)
        ax2.set_ylim(0, 35)
        ax2.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax2.spines.values():
            spine.set_color("#334155")

        legend = ax2.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", fontsize=9)
        for text in legend.get_texts():
            text.set_color("#f8fafc")

        fig.tight_layout()
        fig.savefig(os.path.join(match_dir, "speed_and_distance_timeline.png"), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    def generate_summary_dashboard(self, match_dir, summary_kpis, player_metrics, shots_df, decision_info, dpi=200):
        """Tạo thẻ tóm tắt tổng quan pha bóng Broadcast Infographic Card."""
        fig = plt.figure(figsize=(14, 8), facecolor="#090d16")
        gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.3)

        # 1. Left Court Mini-Map
        ax_court = fig.add_subplot(gs[:, 0])
        self.draw_court_canvas(ax_court, title="Tactical Rally Mini-Map", show_zones=True)
        if not shots_df.empty:
            for _, s in shots_df.iterrows():
                ax_court.plot([s["hit_x_m"], s["bounce_x_m"]], [s["hit_y_m"], s["bounce_y_m"]],
                              color="#f59e0b" if s["hitter_id"] == 2 else "#06b6d4", lw=1.8, alpha=0.75)
                ax_court.scatter([s["bounce_x_m"]], [s["bounce_y_m"]],
                                color="#22c55e" if s["status"] == "IN" else "#ef4444", s=45, zorder=6)

        # 2. Top-Center: Outcome Card
        ax_winner = fig.add_subplot(gs[0, 1])
        ax_winner.set_facecolor("#111827")
        ax_winner.axis("off")
        winner_name = summary_kpis["point_winner"]
        dec_type = summary_kpis["decision"]
        margin_cm = summary_kpis["margin_cm"]

        ax_winner.text(0.5, 0.88, "POINT OUTCOME", color="#94a3b8", fontsize=11, ha="center", fontweight="bold")
        badge_color = "#22c55e" if dec_type in ("IN", "WINNER") else "#ef4444"
        ax_winner.text(0.5, 0.65, f"WINNER: {winner_name.upper()}", color=badge_color, fontsize=15, ha="center", fontweight="black")
        ax_winner.text(0.5, 0.45, f"Hawk-Eye ELC: {dec_type} ({margin_cm:+.1f} cm)", color="#f8fafc", fontsize=12, ha="center", fontweight="semibold")
        ax_winner.text(0.5, 0.22, f"Total Shots: {summary_kpis['rally_length_shots']}  |  Duration: {summary_kpis['duration_seconds']}s",
                       color="#cbd5e1", fontsize=10, ha="center")

        # 3. Top-Right: Speed Card
        ax_speed = fig.add_subplot(gs[0, 2])
        ax_speed.set_facecolor("#111827")
        ax_speed.axis("off")
        ax_speed.text(0.5, 0.88, "SHOT SPEED STATS", color="#94a3b8", fontsize=11, ha="center", fontweight="bold")
        ax_speed.text(0.5, 0.65, f"Fastest Shot: {summary_kpis['ball_speed_kpis']['max_speed_kmh']} km/h",
                      color="#f59e0b", fontsize=14, ha="center", fontweight="bold")
        ax_speed.text(0.5, 0.45, f"By: {summary_kpis['ball_speed_kpis']['fastest_hitter']}",
                      color="#f8fafc", fontsize=11, ha="center")
        ax_speed.text(0.5, 0.22, f"Rally Average Speed: {summary_kpis['ball_speed_kpis']['avg_speed_kmh']} km/h",
                      color="#cbd5e1", fontsize=10.5, ha="center")

        # 4. Bottom-Center: Player 1 Workload
        ax_p1 = fig.add_subplot(gs[1, 1])
        ax_p1.set_facecolor("#111827")
        ax_p1.axis("off")
        p1 = summary_kpis["player_1"]
        ax_p1.text(0.5, 0.88, "PLAYER 1 WORKLOAD", color="#06b6d4", fontsize=12, ha="center", fontweight="bold")
        ax_p1.text(0.5, 0.65, f"Distance Covered: {p1['total_distance_m']} m", color="#f8fafc", fontsize=12, ha="center", fontweight="bold")
        ax_p1.text(0.5, 0.45, f"Max Sprint: {p1['max_speed_kmh']} km/h  |  Avg: {p1['avg_speed_kmh']} km/h", color="#cbd5e1", fontsize=10, ha="center")
        ax_p1.text(0.5, 0.22, f"Shots Hit: {p1['shots_hit']}  |  Deep Ball: {p1['deep_ball_ratio_pct']}%", color="#94a3b8", fontsize=9.5, ha="center")

        # 5. Bottom-Right: Player 2 Workload
        ax_p2 = fig.add_subplot(gs[1, 2])
        ax_p2.set_facecolor("#111827")
        ax_p2.axis("off")
        p2 = summary_kpis["player_2"]
        ax_p2.text(0.5, 0.88, "PLAYER 2 WORKLOAD", color="#f59e0b", fontsize=12, ha="center", fontweight="bold")
        ax_p2.text(0.5, 0.65, f"Distance Covered: {p2['total_distance_m']} m", color="#f8fafc", fontsize=12, ha="center", fontweight="bold")
        ax_p2.text(0.5, 0.45, f"Max Sprint: {p2['max_speed_kmh']} km/h  |  Avg: {p2['avg_speed_kmh']} km/h", color="#cbd5e1", fontsize=10, ha="center")
        ax_p2.text(0.5, 0.22, f"Shots Hit: {p2['shots_hit']}  |  Deep Ball: {p2['deep_ball_ratio_pct']}%", color="#94a3b8", fontsize=9.5, ha="center")

        fig.savefig(os.path.join(match_dir, "match_summary_dashboard.png"), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
