#!/usr/bin/env python
"""Add a legend strip under an AlpaSim DEFAULT-layout rollout video.

    python tools/annotate_video.py <video.mp4> [--out-dir DIR] [--slow 2]

Writes <name>_annotated.{mp4,webm,gif} (and, with --slow N, also an N-times
slower copy <name>_annotated_x1-N.*). Colours and line styles are those of the
AlpaSim eval renderer (src/eval/src/eval/{video,data}.py).
"""
import argparse
import os
import re
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def video_size(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path], text=True)
    w, h = out.strip().split(",")
    return int(w), int(h)


def video_fps(path):
    """Frames per second: 2 for nuPlan rollouts (one frame per decision), 10 for PAI."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", path], text=True)
    num, den = out.strip().split("/")
    return float(num) / float(den)


def camera_name(path):
    """The camera in an AlpaSim video name, <scene>_<rollout>_<camera>_<layout>.mp4."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(CAM_[A-Z0-9]+|camera_\w+?_\d+fov)_(default|reasoning_overlay)$", stem)
    return m.group(1) if m else "front camera"


def legend_png(path, width_px, camera, fps, dpi=100):
    handles = [
        Line2D([], [], color="orange", lw=3, label="Driver's current plan (8 poses, 4 s)"),
        Patch(facecolor=(0.196, 0.804, 0.196, 0.3), edgecolor="k",
              label="Ego car (map centred on it, always pointing up)"),
        Patch(facecolor=(0.196, 0.804, 0.196, 0.12), edgecolor=(0, 0, 0, 0.3),
              label="Human driver's car at the same time (log)"),
        Line2D([], [], color="g", lw=1, alpha=0.7, label="Human driver's recorded path"),
        Line2D([], [], color="g", ls="--", alpha=0.6,
               label="Pointer to the start of the route given to the driver (~40 m ahead)"),
        Patch(facecolor=(0, 0, 0, 0.1), edgecolor="k",
              label="Other vehicles (log replay, non-reactive)"),
        Line2D([], [], color="blue", lw=1, label="Lane boundaries"),
        Line2D([], [], color="blue", lw=1, ls="--", alpha=0.5, label="Lane centrelines"),
    ]
    height_px = 190
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
               fontsize=8.5, handlelength=2.6, columnspacing=1.2,
               bbox_to_anchor=(0.01, 0.97))
    fig.text(0.015, 0.06,
             f"Bottom: rendered camera ({camera}).  Right: metrics "
             "(Agg = over the rollout so far, Per-Ts = this step).  Time in µs.  "
             f"{fps:g} frames per simulated second.",
             fontsize=7.5, color="#444444")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--slow", type=int, default=0,
                        help="also write a copy slowed down by this factor")
    args = parser.parse_args()

    width, _ = video_size(args.video)
    fps = video_fps(args.video)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.video))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(args.video))[0])

    with tempfile.TemporaryDirectory() as tmp:
        legend = os.path.join(tmp, "legend.png")
        legend_png(legend, width, camera_name(args.video), fps)
        # shortest=1: the looped legend image is infinite, so end with the video
        # (-shortest does not stop a looped input that goes through a filter graph)
        stack = "[1:v]scale={w}:-2[l];[0:v][l]vstack=inputs=2:shortest=1,format=yuv420p".format(w=width)
        variants = [("_annotated", 1)] + ([(f"_annotated_x1-{args.slow}", args.slow)] if args.slow > 1 else [])
        for suffix, slow in variants:
            vf = stack + (f",setpts={slow}*PTS" if slow > 1 else "")
            ffmpeg("-i", args.video, "-loop", "1", "-framerate", f"{fps:g}", "-i", legend,
                   "-filter_complex", vf, "-r", f"{fps / slow:g}", "-c:v", "libx264", "-crf", "20",
                   f"{stem}{suffix}.mp4")
            ffmpeg("-i", f"{stem}{suffix}.mp4", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30",
                   f"{stem}{suffix}.webm")
            ffmpeg("-i", f"{stem}{suffix}.mp4", "-vf",
                   "scale=900:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                   f"{stem}{suffix}.gif")
            print(f"{stem}{suffix}.{{mp4,webm,gif}}")


if __name__ == "__main__":
    main()
