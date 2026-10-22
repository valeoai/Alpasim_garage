#!/usr/bin/env python
"""Join the rollout videos of one AlpaSim run into a time-ordered montage.

    python tools/make_montage.py runs/<RUN> [--slow 2] [--name montage]

Clips are ordered by their position in the source nuPlan log (trajdata
start_idx). Each clip gets a title bar (clip k/N, log time, gap to the
previous clip, score, failure) and the legend strip of annotate_video.py.
Every clip is a separate closed-loop rollout that restarts from the logged
pose, so the montage is not one continuous drive; the title bar says so.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_video import ffmpeg, legend_png, video_size  # noqa: E402

NUPLAN_ROOT = os.environ["ALPASIM_NUPLAN_ROOT"]
LOG_HZ = 20  # nuPlan lidar frame rate: start_idx/end_idx are in these frames


def scene_span(scene_id):
    with open(f"{NUPLAN_ROOT}/nuplan_test/{scene_id}/scene_metadata_dt0.10.dill", "rb") as f:
        info = dill.load(f).data_access_info
    return info["start_idx"], info["end_idx"]


def title_png(path, width_px, lines, dpi=100):
    fig = plt.figure(figsize=(width_px / dpi, 0.62), dpi=dpi)
    fig.patch.set_facecolor("#1f2933")
    fig.text(0.012, 0.66, lines[0], fontsize=11, color="white", weight="bold", va="center")
    fig.text(0.012, 0.24, lines[1], fontsize=8.5, color="#d0d7de", va="center")
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--slow", type=int, default=1)
    parser.add_argument("--name", default="montage")
    args = parser.parse_args()

    rollouts = json.load(open(os.path.join(args.run_dir, "aggregate", "results-summary.json")))["rollouts"]
    clips = []
    for r in rollouts:
        scene = r["clipgt_id"]
        videos = glob.glob(os.path.join(args.run_dir, "rollouts", scene, "*", "*_CAM_F0_default.mp4"))
        if not videos:
            continue
        start, end = scene_span(scene)
        clips.append((start, end, scene, r, videos[0]))
    clips.sort()
    if not clips:
        sys.exit("no rollout videos found")

    width, _ = video_size(clips[0][4])
    fps = f"{2 / args.slow:g}"
    out = os.path.join(args.run_dir, args.name + (f"_x1-{args.slow}" if args.slow > 1 else ""))
    log = clips[0][2].rsplit("-", 1)[0]

    with tempfile.TemporaryDirectory() as tmp:
        legend = os.path.join(tmp, "legend.png")
        legend_png(legend, width)
        parts = []
        prev_end = None
        for k, (start, end, scene, r, video) in enumerate(clips, 1):
            gap = "" if prev_end is None else f"   ·   {(start - prev_end) / LOG_HZ:+.1f} s after previous clip"
            prev_end = end
            score = "n/a" if r["score"] is None else f"{r['score']:.2f}"
            verdict = r["status"] + ("" if not r["failure_reason"] else f" ({r['failure_reason']})")
            title = os.path.join(tmp, f"title_{k}.png")
            title_png(title, width, [
                f"Clip {k}/{len(clips)}   ·   log time {start / LOG_HZ:.1f}–{end / LOG_HZ:.1f} s{gap}",
                f"{log}-{scene.rsplit('-', 1)[1]}   ·   score {score}, {verdict}   ·   "
                "separate rollout, restarts from the logged pose",
            ])
            part = os.path.join(tmp, f"part_{k}.mp4")
            ffmpeg("-i", video, "-loop", "1", "-framerate", "2", "-i", title,
                   "-loop", "1", "-framerate", "2", "-i", legend, "-filter_complex",
                   f"[1:v]scale={width}:-2[t];[2:v]scale={width}:-2[l];"
                   f"[t][0:v][l]vstack=inputs=3:shortest=1,format=yuv420p,setpts={args.slow}*PTS",
                   "-r", fps, "-c:v", "libx264", "-crf", "20", part)
            parts.append(part)

        concat = os.path.join(tmp, "concat.txt")
        with open(concat, "w") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        ffmpeg("-f", "concat", "-safe", "0", "-i", concat, "-c", "copy", out + ".mp4")
    ffmpeg("-i", out + ".mp4", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", out + ".webm")
    ffmpeg("-i", out + ".mp4", "-vf",
           "scale=900:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", out + ".gif")
    for start, end, scene, r, _ in clips:
        print(f"{start / LOG_HZ:6.1f}-{end / LOG_HZ:6.1f}s  {scene}  score={r['score']}  {r['status']}")
    print(f"{out}.{{mp4,webm,gif}}")


if __name__ == "__main__":
    main()
