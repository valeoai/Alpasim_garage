#!/usr/bin/env python
"""Summarize a run folder: a single run (runs/<RUN>/aggregate/) or a sharded
SLURM run (runs/<RUN>/shard_*/aggregate/).

    python tools/summarize_run.py runs/<RUN> [--scenes navtest_full]

Prints the AlpaSim challenge metrics (scene score, km per infraction) and
sub-metrics (at-fault collision, off-road, corridor exit, progress), plus
failure reasons and the driver's per-session counters (fallbacks, inference
errors) when its log has them. With several rollouts per scene, each metric is
also computed per rollout repetition (the k-th rollout of every scene) and
reported as mean, variance and standard deviation over repetitions.

Saves the same numbers to runs/<RUN>/metrics.json and metrics.txt. Sharded
runs also get runs/<RUN>/merged-results-summary.json with all rollouts, in the
schema of AlpaSim's results-summary.json. See the README's metrics section for
the definitions.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import statistics

import yaml

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("run_dir")
parser.add_argument("--scenes", default=None,
                    help="scene list to check coverage against (e.g. navtest_full)")
args = parser.parse_args()

run_dir = os.path.abspath(args.run_dir)
parts = sorted(glob.glob(os.path.join(run_dir, "shard_*/")))
sharded = bool(parts)
if not sharded:
    parts = [run_dir]

rollouts, missing = [], []
counters = collections.Counter()
for part in parts:
    summary = os.path.join(part, "aggregate", "results-summary.json")
    if not os.path.exists(summary):
        missing.append(os.path.basename(part.rstrip("/")))
        continue
    with open(summary) as f:
        rollouts += json.load(f)["rollouts"]
    # driver counters: last "session=<id> key=value ..." line per session
    log = os.path.join(part, "driver.log")
    if os.path.exists(log):
        last = {}
        with open(log, errors="replace") as f:
            for line in f:
                m = re.search(r"session=(\S+) (.*)", line)
                if m and not m.group(1).startswith("probe-"):
                    last[m.group(1)] = m.group(2)
        for values in last.values():
            for key, value in re.findall(r"(\w+)=(\d+)", values):
                counters[key] += int(value)

if sharded:
    with open(os.path.join(run_dir, "merged-results-summary.json"), "w") as f:
        json.dump({"schema_version": 1, "rollouts": rollouts}, f)

out_lines = []


def emit(line=""):
    print(line)
    out_lines.append(line)


if sharded:
    emit(f"shards:    {len(parts) - len(missing)}/{len(parts)} finished"
         + (f"  (missing: {', '.join(missing)})" if missing else ""))
elif missing:
    print(f"no aggregate/results-summary.json in {run_dir}: the run did not finish")

by_scene = collections.defaultdict(list)
for r in rollouts:
    by_scene[r["clipgt_id"]].append(r)
for group in by_scene.values():
    group.sort(key=lambda r: str(r["rollout_id"]))  # fixes which rollout is repetition k
n_rep = max((len(g) for g in by_scene.values()), default=0)

line = f"scenes:    {len(by_scene)}"
if args.scenes:
    repo = os.environ["ALPASIM_REPO"]
    path = args.scenes if args.scenes.endswith(".yaml") else os.path.join(
        repo, "src/wizard/configs/nuplan_scenes", f"{args.scenes}.yaml")
    with open(path) as f:
        expected = set(yaml.safe_load(f)["scenes"]["scene_ids"])
    line += f"/{len(expected)} of {os.path.basename(args.scenes)}"
emit(f"{line}   rollouts: {len(rollouts)} ({n_rep} per scene)")
if not rollouts:
    raise SystemExit(1 if missing else 0)


def ran(r):
    """Rollouts that crashed (e.g. route errors) have no metrics."""
    return bool(r["metrics"]) and r["metrics"].get("dist_traveled_m") is not None


def table(groups):
    """Challenge metrics over scene groups (lists of rollouts), as in OPTED Eq. 5."""
    rs = [r for g in groups for r in g]
    ok = [r for r in rs if ran(r)]
    km = sum(r["metrics"]["dist_traveled_m"] for r in ok) / 1000
    infractions = sum(1 for r in ok
                      if r["metrics"]["collision_at_fault"] or r["metrics"]["offroad"])

    def pct(metric):
        return 100 * statistics.mean(float(r["metrics"][metric] > 0) for r in ok) if ok else math.nan

    return {
        "scene_score": 100 * statistics.mean(
            statistics.mean(r["score"] or 0.0 for r in g) for g in groups),
        "km_per_infraction": km / infractions if infractions else math.inf,
        "collision_at_fault": pct("collision_at_fault"),
        "offroad": pct("offroad"),
        "corridor_exit": pct("left_corridor_laterally"),
        "progress": 100 * statistics.mean(
            min(max(r["metrics"]["progress_clipped_rel"], 0.0), 1.0) for r in ok) if ok else math.nan,
        "pass_rate": 100 * statistics.mean(r["status"] == "pass" for r in rs),
        "collision_rear": pct("collision_rear"),
    }


LABELS = {  # name: (label, better)
    "scene_score": ("Scene score (%)", "↑"),
    "km_per_infraction": ("km / infr.", "↑"),
    "collision_at_fault": ("Coll. at-fault (%)", "↓"),
    "offroad": ("Off-road (%)", "↓"),
    "corridor_exit": ("Corridor (%)", "↓"),
    "progress": ("Progress (%)", "↑"),
    "pass_rate": ("Pass (%)", "↑"),
    "collision_rear": ("Rear-ended (%)", "·"),
}

groups = list(by_scene.values())
overall = table(groups)
result = {"run_dir": run_dir, "scenes": len(by_scene), "rollouts": len(rollouts),
          "rollouts_per_scene": n_rep, "crashed_rollouts": sum(not ran(r) for r in rollouts)}
emit()
if n_rep > 1:
    # repetition k = the k-th rollout of every scene, scored like a separate run
    reps = [table([[g[k]] for g in groups if len(g) > k]) for k in range(n_rep)]
    header = (f"{'Metric':22}{'mean':>9}{'var':>10}{'std':>9}"
              + "".join(f"{'rollout ' + str(k + 1):>11}" for k in range(n_rep)))
    emit(header)
    emit("─" * len(header))
    result["metrics"] = {}
    for name, (label, better) in LABELS.items():
        values = [rep[name] for rep in reps]
        var = statistics.variance(values)
        result["metrics"][name] = {"mean": statistics.mean(values), "var": var,
                                   "std": math.sqrt(var), "per_rollout": values,
                                   "all_rollouts": overall[name]}
        d = 3 if name == "km_per_infraction" else 2
        emit(f"{label + ' ' + better:22}{statistics.mean(values):9.{d}f}{var:10.4g}"
             f"{math.sqrt(var):9.{d}f}" + "".join(f"{v:11.{d}f}" for v in values))
    emit(f"(mean/var/std over the {n_rep} rollout repetitions; var is the sample variance)")
else:
    result["metrics"] = overall
    emit(f"{'Metric':22}{'value':>9}")
    emit("─" * 31)
    for name, (label, better) in LABELS.items():
        d = 3 if name == "km_per_infraction" else 2
        emit(f"{label + ' ' + better:22}{overall[name]:9.{d}f}")

# crashed rollouts carry the error text as failure_reason; group them by its first clause
failures = collections.Counter(
    re.sub(r"\d+", "N", (r["failure_reason"] or "none").split(":")[0]) for r in rollouts)
result["failures"] = dict(failures.most_common())
emit()
emit("failures:  " + ", ".join(f"{k}={v}" for k, v in failures.most_common()))
if result["crashed_rollouts"]:
    emit(f"crashed:   {result['crashed_rollouts']} rollouts without metrics "
         "(scored 0; excluded from the other metrics)")
if counters:
    result["driver_counters"] = dict(sorted(counters.items()))
    emit("driver:    " + ", ".join(f"{k}={v}" for k, v in sorted(counters.items())))



def finite(x):
    """inf (no infractions) and nan (no rollouts ran) become null in the JSON."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [finite(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump(finite(result), f, indent=1)
with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
    f.write("\n".join(out_lines) + "\n")
print(f"\nsaved {os.path.join(run_dir, 'metrics.json')} and metrics.txt")
