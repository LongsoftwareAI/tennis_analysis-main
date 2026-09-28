"""
Convenience entrypoint for cutting tennis video clips.
Usage:
  python cut_video_clips.py
  python cut_video_clips.py --start 01:25 --end 01:45 --name my_doubles_clip.mp4
"""
import sys
import os

if __name__ == "__main__":
    from utils.cut_video_clips import main
    main()
