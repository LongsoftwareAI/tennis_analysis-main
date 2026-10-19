import os
import yaml
from copy import deepcopy
from typing import Dict, Any, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "video": {
        "input_path": "input_videos/new_input/clips/clip_01_nadal_vs_verdasco_fast.mp4",
        "output_dir": "output_videos",
        "output_filename": "auto",  # 'auto' generates <video_stem>_analysis.mp4
    },
    "players": {
        "player1_name": "Player 1",
        "player2_name": "Player 2",
    },
    "tracking": {
        "device": "auto",           # 'auto', 'cuda', or 'cpu'
        "court_mode": "cpv",        # 'cpv' (recommended), 'dynamic', or 'static'
        "ball_detector": "hybrid",  # 'hybrid', 'tracknet', or 'yolo'
        "match_mode": "auto",       # 'auto', 'singles', or 'doubles'
        "use_stubs": True,          # Load detection cache from stubs if available
        "force_live": False,        # Force live detection even if stubs exist
        "stubs_dir": "tracker_stubs",
        "ball_batch_size": 4,
    },
    "models": {
        "player_model": "models/yolo26s.pt",
        "ball_model": "models/ball_detector_yolo26_best.pt",
        "ball_model_tracknet": "models/tracknet_weights.pth",
        "ball_model_yolo": "models/ball_detector_yolo26_best.pt",
        "court_model": "models/model_tennis_court_det.pt",
        "bounce_model": "models/bounce_model.cbm",
    },
    "visualization": {
        "draw_players": True,
        "player_draw_mode": "ellipse",   # 'ellipse' or 'box'
        "draw_ball": True,
        "ball_effect": "tracer",         # 'tracer' or 'box'
        "draw_court_keypoints": True,
        "draw_mini_court": True,
        "draw_stats": True,
        "draw_referee_decision": True,
        "draw_frame_number": True,
    },
    "analytics": {
        "enabled": True,
        "export_dir": "match_analytics",
        "generate_visuals": True,
        "dpi": 200,
    },
}


def deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update dict into base dict."""
    result = deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def user_override_path(config_path: str = "config.yaml") -> str:
    """config.yaml -> config.user.yaml (cùng thư mục)."""
    base, ext = os.path.splitext(config_path)
    return f"{base}.user{ext or '.yaml'}"


def _read_yaml(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
        print(f"[Config] Warning: {path} is not a valid dict. Ignoring.")
    except Exception as e:
        print(f"[Config] Error reading {path}: {e}. Ignoring.")
    return None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Thứ tự ưu tiên: DEFAULT_CONFIG < config.yaml < config.user.yaml.
    Key thiếu sẽ tự lấy giá trị mặc định.
    """
    config = deepcopy(DEFAULT_CONFIG)
    sources = []

    base_cfg = _read_yaml(config_path)
    if base_cfg is not None:
        config = deep_merge(config, base_cfg)
        sources.append(config_path)

    user_path = user_override_path(config_path)
    user_cfg = _read_yaml(user_path)
    if user_cfg is not None:
        config = deep_merge(config, user_cfg)
        sources.append(user_path)

    config["_loaded_from"] = " + ".join(sources) if sources else "DEFAULT_CONFIG (config.yaml not found)"
    return config


def save_user_config(settings: Dict[str, Any], config_path: str = "config.yaml") -> str:
    """Lưu các cài đặt người dùng vào config.user.yaml (ghi đè lên config.yaml khi load)."""
    clean = {k: v for k, v in settings.items() if not str(k).startswith("_")}
    path = user_override_path(config_path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, allow_unicode=True, sort_keys=False)
    return path


def reset_user_config(config_path: str = "config.yaml") -> bool:
    """Xoá config.user.yaml. Trả về True nếu có file để xoá."""
    path = user_override_path(config_path)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def print_config_summary(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None):
    """Print a clean summary of active configuration."""
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
    print(f" [Track]  Ball Det: {t.get('ball_detector', 'hybrid').upper()}")
    print(f" [Track]  Match   : {t.get('match_mode', 'auto').upper()}")
    print(f" [Track]  Stubs   : Use={t.get('use_stubs', True)}, ForceLive={t.get('force_live', False)}")
    print(f" [Device] Hardware: {t.get('device', 'auto').upper()}")
    print(f" [Model]  Player  : {m.get('player_model')}")
    print(f" [Model]  Ball    : {m.get('ball_model')}")
    print(f" [Model]  Court   : {m.get('court_model')}")
    if overrides:
        print(" [CLI]    Overrides: " + ", ".join(f"{k}={v}" for k, v in overrides.items() if v is not None))
    print("=" * 65 + "\n")