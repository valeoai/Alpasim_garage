#!/usr/bin/env python
"""Side-by-side rollout videos of two drivers on the same scenes.

    python tools/compare_videos.py runs/<RUN_A> runs/<RUN_B> --labels A B \
        [--slow 2] [--out DIR]

Both runs need rollout videos (VIDEO=1). For every scene present in both, writes
<out>/<token>.{mp4,webm,gif}: the two videos next to each other, each under a
title bar with the driver, its scene score, progress, failure and whether it was
rear-ended, plus the legend strip of annotate_video.py underneath. All scenes are
also joined into <out>/all_scenes.{mp4,webm,gif}. If one rollout ends earlier,
its last frame is held.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_video import ffmpeg, legend_png, video_size  # noqa: E402
from make_montage import title_png  # noqa: E402


def frame_count(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path], text=True)
    return int(out.strip())


def rollouts(run_dir):
    """scene -> (first rollout's result, its CAM_F0 video) for rollouts that have a video."""
    with open(os.path.join(run_dir, "aggregate", "results-summary.json")) as f:
        results = json.load(f)["rollouts"]
    found = {}
    for r in sorted(results, key=lambda r: str(r["rollout_id"])):
        videos = glob.glob(os.path.join(run_dir, "rollouts", r["clipgt_id"], str(r["rollout_id"]),
                                        "*_CAM_F0_default.mp4"))
        if videos and r["clipgt_id"] not in found:
            found[r["clipgt_id"]] = (r, videos[0])
    return found


def describe(label, r):
    m = r["metrics"] or {}
    score = "n/a" if r["score"] is None else f"{r['score']:.2f}"
    verdict = r["status"] + (f" ({r['failure_reason']})" if r["failure_reason"] else "")
    line2 = f"progress {m.get('progress_clipped_rel', float('nan')):.2f}"
    line2 += f"   ·   drove {m.get('dist_traveled_m', float('nan')):.0f} m of the human's {m.get('gt_dist_traveled_m', float('nan')):.0f} m"
    if m.get("collision_rear"):
        line2 += "   ·   REAR-ENDED by replayed traffic"
    return [f"{label}   ·   scene score {score}, {verdict}", line2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--labels", nargs=2, default=["A", "B"])
    parser.add_argument("--slow", type=int, default=2, help="slow-down factor (2 fps source)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    a, b = rollouts(args.run_a), rollouts(args.run_b)
    scenes = [s for s in a if s in b]
    if not scenes:
        sys.exit("no scene has a video in both runs (rerun with VIDEO=1)")
    out_dir = args.out or os.path.join(args.run_a, "compare_" + "_vs_".join(args.labels))
    os.makedirs(out_dir, exist_ok=True)
    width, _ = video_size(a[scenes[0]][1])
    fps = f"{2 / args.slow:g}"

    with tempfile.TemporaryDirectory() as tmp:
        legend = os.path.join(tmp, "legend.png")
        legend_png(legend, 2 * width)
        parts = []
        for scene in scenes:
            (ra, va), (rb, vb) = a[scene], b[scene]
            ta, tb = os.path.join(tmp, "ta.png"), os.path.join(tmp, "tb.png")
            title_png(ta, width, describe(args.labels[0], ra))
            title_png(tb, width, describe(args.labels[1], rb))
            n = max(frame_count(va), frame_count(vb))
            stem = os.path.join(out_dir, scene.rsplit("-", 1)[1])
            ffmpeg("-i", va, "-i", vb,
                   "-loop", "1", "-framerate", "2", "-i", ta,
                   "-loop", "1", "-framerate", "2", "-i", tb,
                   "-loop", "1", "-framerate", "2", "-i", legend,
                   "-filter_complex",
                   f"[0:v]tpad=stop_mode=clone:stop=20[a];[1:v]tpad=stop_mode=clone:stop=20[b];"
                   f"[2:v]scale={width}:-2[ta];[3:v]scale={width}:-2[tb];[4:v]scale={2 * width}:-2[l];"
                   f"[ta][a]vstack=inputs=2:shortest=1[left];[tb][b]vstack=inputs=2:shortest=1[right];"
                   f"[left][right]hstack=inputs=2[top];[top][l]vstack=inputs=2:shortest=1,"
                   f"format=yuv420p,setpts={args.slow}*PTS",
                   "-frames:v", str(n), "-r", fps, "-c:v", "libx264", "-crf", "22", stem + ".mp4")
            parts.append(stem + ".mp4")
            print(f"{stem}.mp4   {args.labels[0]}: {describe(args.labels[0], ra)[0]}   |   "
                  f"{args.labels[1]}: {describe(args.labels[1], rb)[0]}")

        concat = os.path.join(tmp, "concat.txt")
        with open(concat, "w") as f:
            f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
        ffmpeg("-f", "concat", "-safe", "0", "-i", concat, "-c", "copy",
               os.path.join(out_dir, "all_scenes.mp4"))

    for mp4 in parts + [os.path.join(out_dir, "all_scenes.mp4")]:
        stem = mp4[:-4]
        ffmpeg("-i", mp4, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", stem + ".webm")
        ffmpeg("-i", mp4, "-vf",
               "scale=1400:-1:flags=lanczos,split[x][y];[x]palettegen[p];[y][p]paletteuse",
               stem + ".gif")
    print(f"{out_dir}/{{<token>,all_scenes}}.{{mp4,webm,gif}}")


if __name__ == "__main__":
    main()
