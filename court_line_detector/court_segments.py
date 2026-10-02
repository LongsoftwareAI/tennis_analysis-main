import math

import cv2
import numpy as np


def detect_court_segments(video_path, detector, sample_seconds=0.5, min_seconds=2.0):
    """Return half-open frame ranges where the full court can be detected."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video '{video_path}'")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Could not determine frame count or FPS for '{video_path}'")

        def has_court(index):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            readable, frame = capture.read()
            if not readable:
                return False
            try:
                points = np.asarray(detector.predict(frame), dtype=np.float32).reshape((14, 2))
            except RuntimeError as exc:
                if "could not locate keypoints" in str(exc):
                    return False
                raise
            height, width = frame.shape[:2]
            return bool(
                np.isfinite(points).all()
                and np.all((0 <= points[:, 0]) & (points[:, 0] < width))
                and np.all((0 <= points[:, 1]) & (points[:, 1] < height))
            )

        step = max(1, round(fps * sample_seconds))
        sample_indices = list(range(0, frame_count, step))
        if sample_indices[-1] != frame_count - 1:
            sample_indices.append(frame_count - 1)
        valid = [has_court(index) for index in sample_indices]

        segments = []
        first = 0
        minimum_frames = math.ceil(fps * min_seconds)
        while first < len(valid):
            if not valid[first]:
                first += 1
                continue
            last = first
            while last + 1 < len(valid) and valid[last + 1]:
                last += 1

            start = sample_indices[first]
            previous_sample = sample_indices[first - 1] if first else -1
            for index in range(start - 1, previous_sample, -1):
                if not has_court(index):
                    break
                start = index

            end = sample_indices[last] + 1
            next_sample = sample_indices[last + 1] if last + 1 < len(valid) else frame_count
            for index in range(end, next_sample):
                if not has_court(index):
                    break
                end = index + 1

            if end - start >= minimum_frames:
                segments.append((start, end))
            first = last + 1

        return segments
    finally:
        capture.release()


def merge_court_frames(source_frames, segments, analyze_segment):
    """Keep original frames outside court views and render only valid ranges."""
    segment_number = 0
    analyzed_frames = None
    for frame_index, original_frame in enumerate(source_frames):
        if segment_number == len(segments):
            yield original_frame
            continue

        start, end = segments[segment_number]
        if frame_index < start:
            yield original_frame
            continue
        if frame_index == start:
            analyzed_frames = iter(analyze_segment(start, end))

        try:
            analyzed_frame = next(analyzed_frames)
        except StopIteration as exc:
            raise ValueError("Analyzed segment ended before its source range") from exc
        yield analyzed_frame

        if frame_index + 1 == end:
            if next(analyzed_frames, None) is not None:
                raise ValueError("Analyzed segment has more frames than its source range")
            segment_number += 1

    if segment_number != len(segments):
        raise ValueError("Source video ended before all court segments")
