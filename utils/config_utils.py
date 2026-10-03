import os
import yaml
from typing import Dict, Any, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "video": {
        "input_path": "input_videos/new_input/clips/clip_01_nadal_vs_verdasco_fast.mp4",
        "output_dir": "output_videos",
        "output_filename": "auto",  # 'auto' generates <video_stem>_analysis.avi
    },
    "tracking": {
        "device": "auto",          # 'auto', 'cuda', or 'cpu'
        "court_mode": "cpv",       # 'cpv' (recommended), 'dynamic', or 'static'
        "use_stubs": True,         # Load detection cache from stubs if available
        "force_live": False,       # Force live detection even if stubs exist
        "stubs_dir": "tracker_stubs",
        "ball_batch_size": 4,
    },
    "models": {
        "player_model": "models/yolo26s.pt",
        "ball_model": "models/ball_detector_yolo26_best.pt",
        "court_model": "models/model_tennis_court_det.pt",
    },
    "visualization": {
        "draw_players": True,
        "draw_ball": True,
        "draw_court_keypoints": True,
        "draw_mini_court": True,
        "draw_stats": True,
        "draw_frame_number": True,
    }
}

def deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update dict into base dict."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    If file doesn't exist, returns default configuration.
    Any missing keys in user file are automatically filled with default values.
    """
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
                if isinstance(user_config, dict):
                    config = deep_merge(DEFAULT_CONFIG, user_config)
                    config["_loaded_from"] = config_path
                else:
                    print(f"[Config] Warning: {config_path} is not a valid dict. Using defaults.")
        except Exception as e:
            print(f"[Config] Error reading {config_path}: {e}. Using defaults.")
    else:
        config["_loaded_from"] = "DEFAULT_CONFIG (config.yaml not found)"

    return config

def print_config_summary(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None):
    """Print an aesthetic, clean summary of active configuration without unicode emojis."""
    v = config.get("video", {})
    t = config.get("tracking", {})
    m = config.get("models", {})
    source = config.get("_loaded_from", "config.yaml")

    print("\n" + "=" * 65)
    print(">>> TENNIS ANALYSIS PIPELINE - ACTIVE CONFIGURATION <<<")
    print("=" * 65)
    print(f" [Config] Source  : {source}")
    print(f" [Video]  Input   : {v.get('input_path')}")
    print(f" [Video]  Output  : {v.get('output_dir')} (File: {v.get('output_filename', 'auto')})")
    print(f" [Track]  Court   : {t.get('court_mode', 'cpv').upper()}")
    print(f" [Track]  Stubs   : Use={t.get('use_stubs', True)}, ForceLive={t.get('force_live', False)}")
    print(f" [Device] Hardware: {t.get('device', 'auto').upper()}")
    print(f" [Model]  Player  : {m.get('player_model')}")
    print(f" [Model]  Ball    : {m.get('ball_model')}")
    print(f" [Model]  Court   : {m.get('court_model')}")
    if overrides:
        print(" [CLI]    Overrides: " + ", ".join(f"{k}={v}" for k, v in overrides.items() if v is not None))
    print("=" * 65 + "\n")
