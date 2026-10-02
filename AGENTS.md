# Repository Guidelines

## Project Structure & Module Organization

`main.py` orchestrates video analysis. `trackers/` handles players and balls; `court_line_detector/` detects and tracks court keypoints; `mini_court/` projects and draws the 2D court. Shared helpers belong in `utils/`, dimensions in `constants/`. `training/` contains training scripts and notebooks; `analysis/` contains exploratory notebooks. Input media lives in `input_videos/`, demos in `docs/assets/`, and training plots in `reports/`.

## Build, Test, and Development Commands

Use Python 3.10+ and run commands from the repository root:

```powershell
conda activate dat302m
```

The `dat302m` environment is already prepared. Do not install or upgrade packages unless a dependency change is essential to the requested task; explain why and obtain user approval first. No separate build step is configured.

- `python main.py --input input_videos/input_video.mp4`: analyze the included video; output defaults to `output_videos/`.
- `python main.py --config config.yaml --no_stub`: run configured analysis with fresh detections.
- `python cut_video_clips.py`: interactively extract rally clips.
- `python training/train_yolo26_ball_detector.py --dataset merged --epochs 50 --batch 8`: train the ball detector after preparing datasets.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions, variables, and filenames, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Match surrounding code and document non-obvious geometry or coordinate assumptions. No formatter or linter is configured. Keep changes focused; avoid unrelated refactoring.

## Testing Guidelines

No automated test suite or coverage threshold is configured. Check syntax with `python -m compileall -q main.py cut_video_clips.py trackers court_line_detector mini_court utils constants training`. Smoke-test a short clip with `--no_stub` after detection changes; inspect player IDs, court alignment, ball trajectories, statistics, and output playback. Name future permanent tests `tests/test_<module>.py` and adjust `.gitignore`, which currently ignores test filenames. Remove temporary verification scripts after checks pass.

## Commit & Pull Request Guidelines

History mixes `feat:` and `fix:` prefixes with unprefixed updates. Prefer concise, descriptive messages such as `fix: correct ball interpolation`. Keep commits scoped. PRs should explain behavior changes, link relevant issues, report validation commands and configuration, and include screenshots or short clips for visual changes.

## Configuration & Assets

Set video and model paths in `config.yaml`; its default clip is absent from this checkout. Download weights into `models/` using README instructions. Follow `DATASET_STRUCTURE.md` for training data. Keep credentials, checkpoints, datasets, generated videos, and detection caches out of commits; `.env.example` documents `ROBOFLOW_API_KEY`.

