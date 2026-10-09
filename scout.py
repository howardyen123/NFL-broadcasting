"""NFL scout report: animate a play + print scouting metrics. Compact build."""
import glob, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

FIRST_PLAY = True
GAME_ID = None
PLAY_ID = None
SAVE_GIF = "scout_play.gif"
FPS = 10
MPH = 2.045454
FL, FW = 120.0, 53.3

def field(ax):
    ax.add_patch(Rectangle((0, 0), FL, FW, facecolor="#3f9b4e"))
    ax.add_patch(Rectangle((0, 0), 10, FW, facecolor="#2f7a3c"))
    ax.add_patch(Rectangle((FL - 10, 0), 10, FW, facecolor="#2f7a3c"))
    for x in range(10, 111, 5):
        ax.plot([x, x], [0, FW], color="white", lw=1)
    for x in range(20, 101, 10):
        yd = x - 10 if x <= 60 else 110 - x
        ax.text(x, 5, str(yd), color="white", fontsize=12, ha="center", fontweight="bold")
    ax.set_xlim(0, FL)
    ax.set_ylim(0, FW)
    ax.set_aspect("equal")
    ax.axis("off")

def load():
    files = sorted(glob.glob("tracking_2021*.csv"))[:1]
    if not files:
        raise SystemExit("No tracking_2021*.csv found in this folder.")
    print("Loading " + files[0])
    t = pd.read_csv(files[0])
    for c in ("gameId", "playId", "frameId", "nflId"):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce").astype("Int64")
    left = t["playDirection"].astype(str).str.lower() == "left"
    t.loc[left, "x"] = FL - t.loc[left, "x"]
    t.loc[left, "y"] = FW - t.loc[left, "y"]
    plays = pd.read_csv("plays.csv") if os.path.exists("plays.csv") else None
    players = pd.read_csv("players.csv") if os.path.exists("players.csv") else None
    names = {}
    if players is not None:
        pid = pd.to_numeric(players["nflId"], errors="coerce").astype("Int64")
        col = "displayName" if "displayName" in players.columns else "name"
        names = dict(zip(pid, players[col]))
    return t, plays, names

def label(nid, jersey, names):
    if nid in names and pd.notna(names[nid]):
        return str(names[nid])
    return "#" + str(int(jersey)) if pd.notna(jersey) else str(nid)

def run():
    t, plays, names = load()
    print("Tracking rows: " + str(len(t)) + "  plays:" + ("Y" if plays is not None else "N"))
    ids = t[["gameId", "playId"]].drop_duplicates().sort_values(["gameId", "playId"])
    if GAME_ID is not None and PLAY_ID is not None:
        gid, pid = GAME_ID, PLAY_ID
    else:
        gid = int(ids.iloc[0]["gameId"])
        pid = int(ids.iloc[0]["playId"])

    play = t[(t["gameId"] == gid) & (t["playId"] == pid)].sort_values("frameId")
    frames = sorted(play["frameId"].dropna().unique())
    isball = play["team"].astype(str).str.lower().isin(["football", "ball"])
    snap = frames[0]
    if "event" in play.columns:
        s = play[play["event"].astype(str).str.lower() == "ball_snap"]
        if not s.empty:
            snap = int(s["frameId"].iloc[0])

    print("")
    print("====== SCOUT REPORT  game " + str(gid) + " play " + str(pid) + " ======")
    if plays is not None:
        r = plays[(plays["gameId"] == gid) & (plays["playId"] == pid)]
        if not r.empty:
            r = r.iloc[0]
            if pd.notna(r.get("playDescription")):
                print("Play: " + str(r["playDescription"]))
            if pd.notna(r.get("epa")):
                print("EPA : " + format(float(r["epa"]), "+.2f") + " (scoring impact)")

    rows = []
    for nid, g in play[~isball].groupby("nflId"):
        g = g.sort_values("frameId")
        top = float(g["s"].max()) * MPH
        jersey = g["jerseyNumber"].iloc[0] if "jerseyNumber" in g.columns else np.nan
        rows.append((label(nid, jersey, names), top, int(nid)))
    rows.sort(key=lambda z: -z[1])
    print("")
    print("TOP SPEEDS (mph):")
    for nm, sp, _ in rows[:5]:
        print("  " + nm[:24].ljust(24) + format(sp, ".1f"))
    speed_names = {nid: nm for nm, sp, nid in rows[:6]}

    teams = [x for x in play["team"].dropna().unique()]
    pal = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c"]
    cmap, ci = {}, 0
    for tm in teams:
        if str(tm).lower() in ("football", "ball"):
            cmap[tm] = "#8B4513"
        else:
            cmap[tm] = pal[ci % len(pal)]
            ci += 1

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_axes([0.02, 0.05, 0.70, 0.9])
    pan = fig.add_axes([0.74, 0.05, 0.24, 0.9])
    pan.axis("off")
    field(ax)
    fig.suptitle("SCOUT VIEW  game " + str(gid) + " play " + str(pid))
    scat = {}
    for tm, c in cmap.items():
        b = str(tm).lower() in ("football", "ball")
        ln, = ax.plot([], [], "o", color=c, markersize=8 if b else 13,
                      markeredgecolor="black", markeredgewidth=0.7)
        scat[tm] = ln
    txt = pan.text(0, 1, "", va="top", fontsize=9, family="monospace", transform=pan.transAxes)
    ev = ax.text(1, FW + 1.5, "", fontsize=11, fontweight="bold")
    anns = []

    def upd(i):
        fid = frames[i]
        fr = play[play["frameId"] == fid]
        for tm, ln in scat.items():
            sub = fr[fr["team"] == tm]
            ln.set_data(sub["x"].to_numpy(), sub["y"].to_numpy())
        for a in anns:
            a.remove()
        anns.clear()
        fb = fr["team"].astype(str).str.lower().isin(["football", "ball"])
        for _, p in fr[~fb].iterrows():
            if pd.notna(p.get("jerseyNumber")):
                anns.append(ax.text(p["x"], p["y"], str(int(p["jerseyNumber"])),
                            color="white", fontsize=6.5, ha="center", va="center", fontweight="bold"))
        lines = ["LIVE SPEED (mph)", "-" * 20]
        sp = []
        for _, p in fr[~fb].iterrows():
            if int(p["nflId"]) in speed_names and pd.notna(p.get("s")):
                sp.append((speed_names[int(p["nflId"])], p["s"] * MPH))
        sp.sort(key=lambda z: -z[1])
        for nm, mph in sp[:6]:
            lines.append(nm[:16].ljust(16) + format(mph, "4.1f") + " " + "#" * int(min(mph, 22) / 2))
        txt.set_text(chr(10).join(lines))
        e = ""
        if "event" in fr.columns:
            es = fr["event"].dropna()
            es = es[es.astype(str).str.lower() != "none"]
            if not es.empty:
                e = "event: " + str(es.iloc[0])
        ev.set_text(e + "  (t" + format((fid - snap) / FPS, "+.1f") + "s  " + str(i + 1) + "/" + str(len(frames)) + ")")
        return list(scat.values()) + [txt, ev] + anns

    anim = FuncAnimation(fig, upd, frames=len(frames), interval=1000 / FPS, blit=False, repeat=True)
    if SAVE_GIF:
        print("")
        print("Saving " + SAVE_GIF + " ...")
        anim.save(SAVE_GIF, writer="pillow", fps=FPS, dpi=100)
        print("Saved: " + os.path.abspath(SAVE_GIF))
    plt.show()

if __name__ == "__main__":
    run()