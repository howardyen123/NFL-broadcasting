"""
NFL Big Data Bowl — Animated Play Viewer + Scoring-Impact Analysis
==================================================================

A single-file, paste-and-run script for VS Code.

WHAT IT DOES
  1. Loads your tracking_2021*.csv file(s) from the current folder.
  2. (Optional) Joins plays.csv / players.csv for names + EPA (scoring impact).
  3. Picks a play, prints its scoring analysis (EPA, result), and animates it.
  4. Saves the animation as play.gif that you can share.

HOW TO USE
  - Put this file in the SAME folder as your tracking_2021*.csv files.
  - Install libraries once (in the VS Code terminal):
        pip install pandas numpy matplotlib pillow
  - Run it (top-right ▷ Run button, or in the terminal):
        python test.py
  - To list the plays available instead of animating, set LIST_PLAYS = True below.
  - To animate a specific play, set GAME_ID and PLAY_ID below.

Everything is configured in the CONFIG block right under here.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIG  — edit these, then run.
# ============================================================================
DATA_DIR = "."                 # folder with the CSVs ('.' = same folder as this file)
TRACKING_GLOB = "tracking_2021*.csv"

LIST_PLAYS = False             # True = just print available plays and exit
FIRST_PLAY = True              # True = animate the first play found
GAME_ID = None                 # or set an int, e.g. 2021090900
PLAY_ID = None                 # or set an int, e.g. 97

MAX_FILES = 1                  # how many tracking files to load (keep small = fast)
SAVE_GIF = "play.gif"          # output filename; set to None to skip saving
SHOW_WINDOW = True             # pop up an interactive window after saving
# ============================================================================

# Field geometry (matches the tracking coordinate system).
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
ENDZONE = 10.0
HASH_FROM_SIDELINE = 23.58
TRAIL_FRAMES = 6
FPS = 10

# ----------------------------------------------------------------------------
# 1. FIELD
# ----------------------------------------------------------------------------
def draw_field(ax=None):
    """Draw a to-scale NFL field; return the Axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(13.3, 6.0))

    ax.add_patch(Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH,
                           facecolor="#3f9b4e", zorder=0))
    ax.add_patch(Rectangle((0, 0), ENDZONE, FIELD_WIDTH,
                           facecolor="#2f7a3c", zorder=0.5))
    ax.add_patch(Rectangle((FIELD_LENGTH - ENDZONE, 0), ENDZONE, FIELD_WIDTH,
                           facecolor="#2f7a3c", zorder=0.5))

    for x in range(10, 111, 5):
        ax.plot([x, x], [0, FIELD_WIDTH], color="white", lw=1, zorder=1)

    for x in range(20, 101, 10):
        yardage = x - 10 if x <= 60 else 110 - x
        for y, rot in ((5, 0), (FIELD_WIDTH - 5, 180)):
            ax.text(x, y, str(yardage), color="white", fontsize=14,
                    ha="center", va="center", rotation=rot,
                    fontweight="bold", zorder=2)

    for x in range(11, 110):
        for y in (HASH_FROM_SIDELINE, FIELD_WIDTH - HASH_FROM_SIDELINE):
            ax.plot([x, x], [y - 0.4, y + 0.4], color="white", lw=1, zorder=1)
        ax.plot([x, x], [0, 0.6], color="white", lw=1, zorder=1)
        ax.plot([x, x], [FIELD_WIDTH - 0.6, FIELD_WIDTH], color="white", lw=1, zorder=1)

    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax

# ----------------------------------------------------------------------------
# 2. LOAD
# ----------------------------------------------------------------------------
def normalize_direction(df):
    """Flip left-direction plays so everything runs left-to-right."""
    if "playDirection" not in df.columns:
        return df
    left = df["playDirection"].astype(str).str.lower() == "left"
    df.loc[left, "x"] = FIELD_LENGTH - df.loc[left, "x"]
    df.loc[left, "y"] = FIELD_WIDTH - df.loc[left, "y"]
    for ang in ("o", "dir"):
        if ang in df.columns:
            df.loc[left, ang] = (df.loc[left, ang] + 180.0) % 360.0
    return df


def load_data():
    """Load tracking (+ optional plays/players) from DATA_DIR."""
    pattern = os.path.join(DATA_DIR, TRACKING_GLOB)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No tracking files matched '{pattern}'. "
            f"Put this script in the same folder as your tracking_2021*.csv files."
        )
    files = files[:MAX_FILES]
    print(f"Loading {len(files)} tracking file(s):")
    for f in files:
        print(f"   - {os.path.basename(f)}")

    tracking = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    for col in ("gameId", "playId", "frameId"):
        if col in tracking.columns:
            tracking[col] = pd.to_numeric(tracking[col], errors="coerce").astype("Int64")
    tracking = normalize_direction(tracking)

    plays = _maybe_csv("plays.csv")
    players = _maybe_csv("players.csv")

    if players is not None and "nflId" in players.columns:
        name_col = next((c for c in ("displayName", "name") if c in players.columns), None)
        if name_col:
            tracking = tracking.merge(
                players[["nflId", name_col]].rename(columns={name_col: "displayName"}),
                on="nflId", how="left",
            )
    return tracking, plays, players


def _maybe_csv(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        print(f"   + found {name}")
        return pd.read_csv(path)
    return None

# ----------------------------------------------------------------------------
# 3. SCORING ANALYSIS
# ----------------------------------------------------------------------------
def scoring_analysis(plays, game_id, play_id):
    """Print the EPA / result for a play. Returns a short title string."""
    if plays is None:
        print("\n[scoring] plays.csv not found — skipping EPA analysis.")
        print("          (Download plays.csv from the same Big Data Bowl dataset")
        print("           to unlock Expected Points Added / scoring impact.)")
        return f"game {game_id} · play {play_id}"

    row = plays[(plays["gameId"] == game_id) & (plays["playId"] == play_id)]
    if row.empty:
        return f"game {game_id} · play {play_id}"
    r = row.iloc[0]

    desc = str(r["playDescription"]) if "playDescription" in r and pd.notna(r.get("playDescription")) else ""
    epa = r.get("epa", np.nan)
    result = r.get("playResult", np.nan)
    pass_res = r.get("passResult", "")

    print("\n" + "=" * 70)
    print("SCORING IMPACT ANALYSIS")
    print("=" * 70)
    if desc:
        print(f"Play       : {desc}")
    if "down" in r and pd.notna(r.get("down")):
        print(f"Situation  : {int(r['down'])} & {int(r.get('yardsToGo', 0))}"
              f"   ({r.get('possessionTeam', '?')} ball)")
    if pd.notna(result):
        print(f"Yards      : {int(result)}")
    if pass_res:
        meaning = {"C": "Complete", "I": "Incomplete", "S": "Sack",
                   "IN": "Interception", "R": "Scramble"}.get(str(pass_res), pass_res)
        print(f"Pass result: {meaning}")
    if pd.notna(epa):
        verdict = ("strongly helped" if epa > 1 else "helped" if epa > 0
                   else "hurt" if epa > -1 else "strongly hurt")
        print(f"EPA        : {epa:+.2f}  ->  this play {verdict} the team's scoring chances")
        print("             (EPA = Expected Points Added: how much the play changed")
        print("              the team's expected points. Positive = better scoring odds.")
    print("=" * 70 + "\n")

    short = desc[:70] + ("..." if len(desc) > 70 else "") if desc else f"play {play_id}"
    if pd.notna(epa):
        short += f"   |   EPA {epa:+.2f}"
    return short

# ----------------------------------------------------------------------------
# 4. ANIMATE
# ----------------------------------------------------------------------------
def _is_ball(series):
    return series.astype(str).str.lower().isin(["football", "ball"])


def animate_play(tracking, plays, game_id, play_id):
    play = tracking[(tracking["gameId"] == game_id) &
                    (tracking["playId"] == play_id)].copy()
    if play.empty:
        raise ValueError(f"No tracking rows for game {game_id}, play {play_id}.")
    play = play.sort_values("frameId").reset_index(drop=True)
    frames = sorted(play["frameId"].dropna().unique())

    title = scoring_analysis(plays, game_id, play_id)

    teams = [t for t in play["team"].dropna().unique()]
    palette = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c", "#ff7f0e"]
    color_map, ci = {}, 0
    for t in teams:
        if str(t).lower() in ("football", "ball"):
            color_map[t] = "#8B4513"
        else:
            color_map[t] = palette[ci % len(palette)]
            ci += 1

    fig, ax = plt.subplots(figsize=(13.3, 6.0))
    draw_field(ax)
    fig.suptitle(title, fontsize=10, y=0.97, wrap=True)
    event_text = ax.text(1, FIELD_WIDTH + 1.5, "", fontsize=11, fontweight="bold")

    scatters = {}
    for team, color in color_map.items():
        is_ball = str(team).lower() in ("football", "ball")
        (s,) = ax.plot([], [], "o", color=color,
                       markersize=8 if is_ball else 13,
                       markeredgecolor="black", markeredgewidth=0.7,
                       zorder=5 if is_ball else 4,
                       label=None if is_ball else str(team))
        scatters[team] = s
    (trail,) = ax.plot([], [], "o", color="white", markersize=4, alpha=0.25, zorder=3)
    annotations = []

    if [t for t in teams if str(t).lower() not in ("football", "ball")]:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    def init():
        for s in scatters.values():
            s.set_data([], [])
        trail.set_data([], [])
        event_text.set_text("")
        return list(scatters.values()) + [trail, event_text]

    def update(i):
        fid = frames[i]
        fr = play[play["frameId"] == fid]
        for team, s in scatters.items():
            sub = fr[fr["team"] == team]
            s.set_data(sub["x"].to_numpy(), sub["y"].to_numpy())

        lo = max(0, i - TRAIL_FRAMES)
        tf = frames[lo:i]
        if tf:
            tp = play[play["frameId"].isin(tf) & ~_is_ball(play["team"])]
            trail.set_data(tp["x"].to_numpy(), tp["y"].to_numpy())
        else:
            trail.set_data([], [])

        for a in annotations:
            a.remove()
        annotations.clear()
        if "jerseyNumber" in fr.columns:
            for _, p in fr[~_is_ball(fr["team"])].iterrows():
                if pd.notna(p.get("jerseyNumber")):
                    annotations.append(ax.text(
                        p["x"], p["y"], str(int(p["jerseyNumber"])),
                        color="white", fontsize=6.5, ha="center", va="center",
                        zorder=6, fontweight="bold"))

        ev = ""
        if "event" in fr.columns:
            evs = fr["event"].dropna()
            evs = evs[evs.astype(str).str.lower() != "none"]
            if not evs.empty:
                ev = f"event: {evs.iloc[0]}"
        event_text.set_text(f"{ev}    (t={ (i+1)/FPS:.1f}s  frame {i+1}/{len(frames)})")
        return list(scatters.values()) + [trail, event_text] + annotations

    anim = FuncAnimation(fig, update, frames=len(frames), init_func=init,
                         interval=1000 / FPS, blit=False, repeat=True)

    if SAVE_GIF:
        print(f"Saving animation to {SAVE_GIF} (this can take a few seconds)...")
        anim.save(SAVE_GIF, writer="pillow", fps=FPS, dpi=100)
        print(f"Saved: {os.path.abspath(SAVE_GIF)}")

    if SHOW_WINDOW:
        plt.show()
    return anim

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    tracking, plays, players = load_data()
    print(f"\nTracking loaded: {len(tracking):,} rows")
    print(f"players.csv: {'yes' if players is not None else 'no'}   "
          f"plays.csv: {'yes' if plays is not None else 'no'}")

    play_ids = (tracking[["gameId", "playId"]].drop_duplicates()
                .sort_values(["gameId", "playId"]).reset_index(drop=True))

    if LIST_PLAYS:
        print(f"\n{len(play_ids)} plays available:")
        print(play_ids.to_string(index=False))
        return

    if GAME_ID is not None and PLAY_ID is not None:
        gid, pid = GAME_ID, PLAY_ID
    elif FIRST_PLAY:
        gid = int(play_ids.iloc[0]["gameId"])
        pid = int(play_ids.iloc[0]["playId"])
    else:
        print("Set FIRST_PLAY = True, or set GAME_ID and PLAY_ID.")
        return

    print(f"\nAnimating game {gid}, play {pid} ...")
    animate_play(tracking, plays, gid, pid)


if __name__ == "__main__":
    main()
