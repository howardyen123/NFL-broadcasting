"""NFL broadcast view: polished play animation with team colors,
PRESSURE / WIDE OPEN flashes, a stat bug, and GIF/MP4 export."""
import glob, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyBboxPatch

GAME_ID = None            # None = first play in the file
PLAY_ID = None
SAVE = "broadcast_play.gif"   # ".gif" (always works) or ".mp4" (needs ffmpeg)
FPS = 10
MPH = 2.045454
FL, FW = 120.0, 53.3
PRESSURE_RADIUS = 2.5     # yards from QB that triggers PRESSURE
OPEN_RADIUS = 3.0         # receiver yards from nearest defender for WIDE OPEN

# Primary / secondary colors for all 32 teams (abbrev -> (fill, edge)).
TEAM_COLORS = {
    "ARI": ("#97233F", "#FFFFFF"), "ATL": ("#A71930", "#000000"),
    "BAL": ("#241773", "#9E7C0C"), "BUF": ("#00338D", "#C60C30"),
    "CAR": ("#0085CA", "#101820"), "CHI": ("#0B162A", "#C83803"),
    "CIN": ("#FB4F14", "#000000"), "CLE": ("#311D00", "#FF3C00"),
    "DAL": ("#003594", "#869397"), "DEN": ("#FB4F14", "#002244"),
    "DET": ("#0076B6", "#B0B7BC"), "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#A71930"), "IND": ("#002C5F", "#FFFFFF"),
    "JAX": ("#101820", "#D7A22A"), "KC": ("#E31837", "#FFB81C"),
    "LAC": ("#0080C6", "#FFC20E"), "LA": ("#003594", "#FFA300"),
    "LAR": ("#003594", "#FFA300"), "LV": ("#000000", "#A5ACAF"),
    "MIA": ("#008E97", "#FC4C02"), "MIN": ("#4F2683", "#FFC62F"),
    "NE": ("#002244", "#C60C30"), "NO": ("#D3BC8D", "#101820"),
    "NYG": ("#0B2265", "#A71930"), "NYJ": ("#125740", "#FFFFFF"),
    "PHI": ("#004C54", "#A5ACAF"), "PIT": ("#FFB612", "#101820"),
    "SF": ("#AA0000", "#B3995D"), "SEA": ("#002244", "#69BE28"),
    "TB": ("#D50A0A", "#34302B"), "TEN": ("#0C2340", "#4B92DB"),
    "WAS": ("#5A1414", "#FFB612"),
}

def team_color(abbr):
    return TEAM_COLORS.get(str(abbr).upper(), ("#1f77b4", "#FFFFFF"))

def field(ax):
    ax.add_patch(Rectangle((0, 0), FL, FW, facecolor="#2e7d32"))
    ax.add_patch(Rectangle((0, 0), 10, FW, facecolor="#1b5e20"))
    ax.add_patch(Rectangle((FL - 10, 0), 10, FW, facecolor="#1b5e20"))
    for x in range(10, 111, 5):
        ax.plot([x, x], [0, FW], color="white", lw=1, alpha=0.8)
    for x in range(20, 101, 10):
        yd = x - 10 if x <= 60 else 110 - x
        ax.text(x, 5, str(yd), color="white", fontsize=12, ha="center", fontweight="bold")
        ax.text(x, FW - 5, str(yd), color="white", fontsize=12, ha="center",
                va="center", rotation=180, fontweight="bold")
    ax.set_xlim(0, FL)
    ax.set_ylim(-4, FW + 4)
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

def ctx(plays, gid, pid):
    out = {"desc": "", "epa": None, "down": None, "togo": None,
           "off": None, "deff": None}
    if plays is None:
        return out
    r = plays[(plays["gameId"] == gid) & (plays["playId"] == pid)]
    if r.empty:
        return out
    r = r.iloc[0]
    out["desc"] = str(r.get("playDescription", "")) if pd.notna(r.get("playDescription")) else ""
    if pd.notna(r.get("epa")):
        out["epa"] = float(r["epa"])
    if pd.notna(r.get("down")):
        out["down"] = int(r["down"])
    if pd.notna(r.get("yardsToGo")):
        out["togo"] = int(r["yardsToGo"])
    if pd.notna(r.get("possessionTeam")):
        out["off"] = str(r["possessionTeam"])
    if pd.notna(r.get("defensiveTeam")):
        out["deff"] = str(r["defensiveTeam"])
    return out

def run():
    t, plays, names = load()
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

    info = ctx(plays, gid, pid)
    teams = [x for x in play["team"].dropna().unique()
             if str(x).lower() not in ("football", "ball")]
    off_team = info["off"] if info["off"] in teams else (teams[0] if teams else "")
    def_team = info["deff"] if info["deff"] in teams else (
        teams[1] if len(teams) > 1 else "")

    print("")
    print("====== BROADCAST  game " + str(gid) + " play " + str(pid) + " ======")
    if info["desc"]:
        print("Play: " + info["desc"])
    if info["epa"] is not None:
        print("EPA : " + format(info["epa"], "+.2f"))

    cmap = {}
    for tm in play["team"].dropna().unique():
        if str(tm).lower() in ("football", "ball"):
            cmap[tm] = ("#8B4513", "#3b2109")
        else:
            cmap[tm] = team_color(tm)

    fig = plt.figure(figsize=(14, 6.5), facecolor="#0d0d0d")
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.86])
    field(ax)

    scat = {}
    for tm, (fill, edge) in cmap.items():
        b = str(tm).lower() in ("football", "ball")
        ln, = ax.plot([], [], "o", color=fill, markersize=9 if b else 15,
                      markeredgecolor=edge, markeredgewidth=1.6,
                      zorder=5 if b else 4)
        scat[tm] = ln
    anns = []

    bug = fig.text(0.03, 0.95, "", color="white", fontsize=12,
                   fontweight="bold", family="monospace", va="center")
    epa_txt = fig.text(0.70, 0.95, "", color="#ffd54f", fontsize=13,
                       fontweight="bold", va="center")
    alert = ax.text(FL / 2, FW + 2.5, "", color="#ff1744", fontsize=22,
                    fontweight="bold", ha="center", va="center", zorder=10)
    clock = fig.text(0.97, 0.95, "", color="#bbbbbb", fontsize=10,
                     ha="right", va="center", family="monospace")

    down_str = ""
    if info["down"] is not None:
        suff = {1: "st", 2: "nd", 3: "rd", 4: "th"}.get(info["down"], "th")
        down_str = str(info["down"]) + suff
        if info["togo"] is not None:
            down_str += " & " + str(info["togo"])
    bug_label = (str(off_team) + " vs " + str(def_team)
                 + ("   " + down_str if down_str else ""))

    def ball_xy(fr):
        b = fr[fr["team"].astype(str).str.lower().isin(["football", "ball"])]
        if not b.empty:
            return float(b["x"].iloc[0]), float(b["y"].iloc[0])
        return None

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
                            color="white", fontsize=6.5, ha="center",
                            va="center", zorder=6, fontweight="bold"))

        msg = ""
        bxy = ball_xy(fr)
        if bxy is not None:
            qx, qy = bxy
            defs = fr[(fr["team"] == def_team)]
            offs = fr[(fr["team"] == off_team)]
            for _, d in defs.iterrows():
                if np.hypot(d["x"] - qx, d["y"] - qy) <= PRESSURE_RADIUS:
                    msg = "PRESSURE!"
                    break
            if not msg and not defs.empty:
                for _, o in offs.iterrows():
                    if np.hypot(o["x"] - qx, o["y"] - qy) < 3:
                        continue
                    dmin = defs.apply(
                        lambda d: np.hypot(o["x"] - d["x"], o["y"] - d["y"]),
                        axis=1).min()
                    if dmin >= OPEN_RADIUS + 3:
                        msg = "WIDE OPEN"
                        alert.set_color("#00e676")
                        break
            if msg == "PRESSURE!":
                alert.set_color("#ff1744")
        alert.set_text(msg)

        e = ""
        if "event" in fr.columns:
            es = fr["event"].dropna()
            es = es[es.astype(str).str.lower() != "none"]
            if not es.empty:
                e = str(es.iloc[0])
        bug.set_text(bug_label)
        if info["epa"] is not None:
            epa_txt.set_text("EPA " + format(info["epa"], "+.2f"))
        clock.set_text("t" + format((fid - snap) / FPS, "+.1f") + "s  "
                       + str(i + 1) + "/" + str(len(frames))
                       + ("  " + e if e else ""))
        return (list(scat.values()) + anns
                + [alert, bug, epa_txt, clock])

    anim = FuncAnimation(fig, upd, frames=len(frames),
                         interval=1000 / FPS, blit=False, repeat=True)

    if SAVE:
        ext = os.path.splitext(SAVE)[1].lower()
        print("")
        print("Saving " + SAVE + " ...")
        try:
            if ext == ".mp4":
                anim.save(SAVE, writer="ffmpeg", fps=FPS, dpi=120)
            else:
                anim.save(SAVE, writer="pillow", fps=FPS, dpi=110)
            print("Saved: " + os.path.abspath(SAVE))
        except Exception as err:
            print("Could not save " + SAVE + " (" + str(err) + ")")
            print("Tip: for .mp4 you need ffmpeg; .gif always works.")
    plt.show()

if __name__ == "__main__":
    run()