import numpy as np
import pandas as pd
import cv2

def draw_player_stats(output_video_frames, player_stats, start_frame=0):
    is_doubles = False
    if 'player_3_number_of_shots' in player_stats.columns or 'player_4_number_of_shots' in player_stats.columns:
        total_p3 = player_stats.get('player_3_number_of_shots', pd.Series([0])).max()
        total_p4 = player_stats.get('player_4_number_of_shots', pd.Series([0])).max()
        if total_p3 > 0 or total_p4 > 0:
            is_doubles = True

    for index, row in player_stats.iloc[start_frame:start_frame + len(output_video_frames)].iterrows():
        frame_index = index - start_frame
        if is_doubles:
            p1_shot_speed = max(row.get('player_1_last_shot_speed', 0.0), row.get('player_3_last_shot_speed', 0.0))
            p2_shot_speed = max(row.get('player_2_last_shot_speed', 0.0), row.get('player_4_last_shot_speed', 0.0))
            p1_speed = max(row.get('player_1_last_player_speed', 0.0), row.get('player_3_last_player_speed', 0.0))
            p2_speed = max(row.get('player_2_last_player_speed', 0.0), row.get('player_4_last_player_speed', 0.0))
            avg_p1_shot_speed = max(row.get('player_1_average_shot_speed', 0.0), row.get('player_3_average_shot_speed', 0.0))
            avg_p2_shot_speed = max(row.get('player_2_average_shot_speed', 0.0), row.get('player_4_average_shot_speed', 0.0))
            avg_p1_speed = max(row.get('player_1_average_player_speed', 0.0), row.get('player_3_average_player_speed', 0.0))
            avg_p2_speed = max(row.get('player_2_average_player_speed', 0.0), row.get('player_4_average_player_speed', 0.0))
            header_text = "  Team 1 (P1/P3)  Team 2 (P2/P4)"
        else:
            p1_shot_speed = row.get('player_1_last_shot_speed', 0.0)
            p2_shot_speed = row.get('player_2_last_shot_speed', 0.0)
            p1_speed = row.get('player_1_last_player_speed', 0.0)
            p2_speed = row.get('player_2_last_player_speed', 0.0)
            avg_p1_shot_speed = row.get('player_1_average_shot_speed', 0.0)
            avg_p2_shot_speed = row.get('player_2_average_shot_speed', 0.0)
            avg_p1_speed = row.get('player_1_average_player_speed', 0.0)
            avg_p2_speed = row.get('player_2_average_player_speed', 0.0)
            header_text = "     Player 1     Player 2"

        frame = output_video_frames[frame_index]
        width = 360
        height = 230

        start_x = frame.shape[1] - 410
        start_y = frame.shape[0] - 500
        end_x = start_x + width
        end_y = start_y + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.55
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[frame_index] = frame

        cv2.putText(output_video_frames[frame_index], header_text, (start_x + 35, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(output_video_frames[frame_index], "Shot Speed", (start_x + 10, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        text = f"{p1_shot_speed:.1f} km/h    {p2_shot_speed:.1f} km/h"
        cv2.putText(output_video_frames[frame_index], text, (start_x + 130, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(output_video_frames[frame_index], "Player Speed", (start_x + 10, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        text = f"{p1_speed:.1f} km/h    {p2_speed:.1f} km/h"
        cv2.putText(output_video_frames[frame_index], text, (start_x + 130, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(output_video_frames[frame_index], "avg. S. Speed", (start_x + 10, start_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        text = f"{avg_p1_shot_speed:.1f} km/h    {avg_p2_shot_speed:.1f} km/h"
        cv2.putText(output_video_frames[frame_index], text, (start_x + 130, start_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(output_video_frames[frame_index], "avg. P. Speed", (start_x + 10, start_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        text = f"{avg_p1_speed:.1f} km/h    {avg_p2_speed:.1f} km/h"
        cv2.putText(output_video_frames[frame_index], text, (start_x + 130, start_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return output_video_frames
