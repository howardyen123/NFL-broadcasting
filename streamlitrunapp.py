"""NFL Big Data Bowl - interactive scout + broadcast app (Streamlit).

Run it with:   streamlit run app.py
It opens in your web browser. Put this file in the folder with the CSVs.
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st
import analytics

MPH = 2.045454
FL, FW = 120.0, 53.3
FPS = 10
PRESSURE_RADIUS = 2.5
OPEN_RADIUS = 6.0

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

@st.cache_data(show_spinner=True)
def load_tracking(path):
    t = pd.read_csv(path)
    for c in ("gameId", "playId", "frameId", "nflId"):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce").astype("Int64")
    left = t["playDirection"].astype(str).str.lower() == "left"
    t.loc[left, "x"] = FL - t.loc[left, "x"]
    t.loc[left, "y"] = FW - t.loc[left, "y"]
    return t

@st.cache_data(show_spinner=False)
def load_side():
    plays = pd.read_csv("plays.csv") if os.path.exists("plays.csv") else None
    players = pd.read_csv("players.csv") if os.path.exists("players.csv") else None
    pff = pd.read_csv("pffScoutingData.csv") if os.path.exists("pffScoutingData.csv") else None
    if pff is not None:
        for c in ("gameId", "playId", "nflId"):
            if c in pff.columns:
                pff[c] = pd.to_numeric(pff[c], errors="coerce").astype("Int64")
    names = {}
    if players is not None:
        pid = pd.to_numeric(players["nflId"], errors="coerce").astype("Int64")
        col = "displayName" if "displayName" in players.columns else "name"
        names = dict(zip(pid, players[col]))
    return plays, names, pff

def label(nid, jersey, names):
    if nid in names and pd.notna(names[nid]):
        return str(names[nid])
    return "#" + str(int(jersey)) if pd.notna(jersey) else str(nid)

def is_ball(series):
    return series.astype(str).str.lower().isin(["football", "ball"])

def play_context(plays, gid, pid):
    out = {"desc": "", "epa": None, "down": None, "togo": None,
           "off": None, "deff": None, "result": None}
    if plays is None:
        return out
    r = plays[(plays["gameId"] == gid) & (plays["playId"] == pid)]
    if r.empty:
        return out
    r = r.iloc[0]
    if pd.notna(r.get("playDescription")):
        out["desc"] = str(r["playDescription"])
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
    if pd.notna(r.get("playResult")):
        out["result"] = int(r["playResult"])
    return out

def snap_frame(play):
    if "event" in play.columns:
        s = play[play["event"].astype(str).str.lower() == "ball_snap"]
        if not s.empty:
            return int(s["frameId"].iloc[0])
    return int(play["frameId"].min())

def draw_field(ax):
    ax.add_patch(Rectangle((0, 0), FL, FW, facecolor="#2e7d32"))
    ax.add_patch(Rectangle((0, 0), 10, FW, facecolor="#1b5e20"))
    ax.add_patch(Rectangle((FL - 10, 0), 10, FW, facecolor="#1b5e20"))
    for x in range(10, 111, 5):
        ax.plot([x, x], [0, FW], color="white", lw=1, alpha=0.8)
    for x in range(20, 101, 10):
        yd = x - 10 if x <= 60 else 110 - x
        ax.text(x, 5, str(yd), color="white", fontsize=11, ha="center",
                fontweight="bold")
    ax.set_xlim(0, FL)
    ax.set_ylim(-2, FW + 2)
    ax.set_aspect("equal")
    ax.axis("off")

def metrics_table(play, names):
    snap = snap_frame(play)
    cut = snap + FPS
    ballmask = is_ball(play["team"])
    rows = []
    for nid, g in play[~ballmask].groupby("nflId"):
        g = g.sort_values("frameId")
        jersey = g["jerseyNumber"].iloc[0] if "jerseyNumber" in g.columns else np.nan
        top = float(g["s"].max()) * MPH if "s" in g.columns else np.nan
        win = g[(g["frameId"] >= snap) & (g["frameId"] <= cut)]
        burst = np.nan
        if "s" in g.columns and len(win) >= 2:
            burst = float(win["s"].iloc[-1] - win["s"].iloc[0])
        team = g["team"].iloc[0] if "team" in g.columns else ""
        rows.append({
            "player": label(nid, jersey, names),
            "team": team,
            "top_speed_mph": round(top, 1) if pd.notna(top) else np.nan,
            "getoff_yd_s": round(burst, 2) if pd.notna(burst) else np.nan,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("top_speed_mph", ascending=False).reset_index(drop=True)

def render_frame(play, fid, snap, off_team, def_team, cmap, show_alert):
    fig, ax = plt.subplots(figsize=(11, 5.3), facecolor="#0d0d0d")
    draw_field(ax)
    fr = play[play["frameId"] == fid]
    for tm, (fill, edge) in cmap.items():
        sub = fr[fr["team"] == tm]
        b = str(tm).lower() in ("football", "ball")
        ax.plot(sub["x"].to_numpy(), sub["y"].to_numpy(), "o", color=fill,
                markersize=9 if b else 15, markeredgecolor=edge,
                markeredgewidth=1.5, zorder=5 if b else 4)
    ballmask = is_ball(fr["team"])
    if "jerseyNumber" in fr.columns:
        for _, p in fr[~ballmask].iterrows():
            if pd.notna(p.get("jerseyNumber")):
                ax.text(p["x"], p["y"], str(int(p["jerseyNumber"])),
                        color="white", fontsize=6.5, ha="center",
                        va="center", zorder=6, fontweight="bold")
    msg, color = "", "#ff1744"
    if show_alert:
        b = fr[ballmask]
        if not b.empty:
            qx, qy = float(b["x"].iloc[0]), float(b["y"].iloc[0])
            defs = fr[fr["team"] == def_team]
            offs = fr[fr["team"] == off_team]
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
                    if dmin >= OPEN_RADIUS:
                        msg, color = "WIDE OPEN", "#00e676"
                        break
    if msg:
        ax.text(FL / 2, FW + 1, msg, color=color, fontsize=20,
                fontweight="bold", ha="center", va="center", zorder=10)
    ax.text(2, FW + 1, "t" + format((fid - snap) / FPS, "+.1f") + "s",
            color="#bbbbbb", fontsize=10, family="monospace")
    return fig

def build_gif(play, frames, snap, off_team, def_team, cmap, out="app_play.gif"):
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots(figsize=(11, 5.3), facecolor="#0d0d0d")
    draw_field(ax)
    scat = {}
    for tm, (fill, edge) in cmap.items():
        b = str(tm).lower() in ("football", "ball")
        ln, = ax.plot([], [], "o", color=fill, markersize=9 if b else 15,
                      markeredgecolor=edge, markeredgewidth=1.5)
        scat[tm] = ln
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
        bm = is_ball(fr["team"])
        for _, p in fr[~bm].iterrows():
            if pd.notna(p.get("jerseyNumber")):
                anns.append(ax.text(p["x"], p["y"], str(int(p["jerseyNumber"])),
                            color="white", fontsize=6, ha="center",
                            va="center", fontweight="bold"))
        return list(scat.values()) + anns

    anim = FuncAnimation(fig, upd, frames=len(frames), interval=1000 / FPS)
    anim.save(out, writer="pillow", fps=FPS, dpi=90)
    plt.close(fig)
    return out

def main():
    st.set_page_config(page_title="NFL Scout + Broadcast", layout="wide")
    st.title("NFL Big Data Bowl - Scout and Broadcast Tool")

    files = sorted(glob.glob("tracking_2021*.csv"))
    if not files:
        st.error("No tracking_2021*.csv files found in this folder.")
        return

    st.sidebar.header("Controls")
    track_file = st.sidebar.selectbox("Tracking file", files)
    t = load_tracking(track_file)
    plays, names, pff = load_side()

    games = sorted(t["gameId"].dropna().unique().tolist())
    gid = st.sidebar.selectbox("Game", games)
    play_ids = sorted(
        t[t["gameId"] == gid]["playId"].dropna().unique().tolist())
    pid = st.sidebar.selectbox("Play", play_ids)

    play = t[(t["gameId"] == gid) & (t["playId"] == pid)].sort_values("frameId")
    frames = sorted(play["frameId"].dropna().unique())
    if not frames:
        st.warning("No frames for this play.")
        return
    snap = snap_frame(play)

    info = play_context(plays, int(gid), int(pid))
    teams = [x for x in play["team"].dropna().unique()
             if str(x).lower() not in ("football", "ball")]
    off_team = info["off"] if info["off"] in teams else (teams[0] if teams else "")
    def_team = info["deff"] if info["deff"] in teams else (
        teams[1] if len(teams) > 1 else "")

    cmap = {}
    for tm in play["team"].dropna().unique():
        if str(tm).lower() in ("football", "ball"):
            cmap[tm] = ("#8B4513", "#3b2109")
        else:
            cmap[tm] = team_color(tm)

    show_alert = st.sidebar.checkbox("Show PRESSURE / WIDE OPEN", value=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matchup", str(off_team) + " vs " + str(def_team))
    down_s = "-"
    if info["down"] is not None:
        suff = {1: "st", 2: "nd", 3: "rd", 4: "th"}.get(info["down"], "th")
        down_s = str(info["down"]) + suff + (
            " & " + str(info["togo"]) if info["togo"] is not None else "")
    c2.metric("Down and distance", down_s)
    c3.metric("EPA", format(info["epa"], "+.2f") if info["epa"] is not None else "-")
    c4.metric("Result (yds)", str(info["result"]) if info["result"] is not None else "-")

    if info["desc"]:
        st.caption(info["desc"])

    tab_play, tab_analytics = st.tabs(["Play view", "Analytics"])
    with tab_analytics:
        analytics.render_tab(t, plays, pff, names, int(gid))
    with tab_play:
        left, right = st.columns([3, 2])
    with left:
        st.subheader("Play view")
        i = st.slider("Frame", 1, len(frames), 1)
        fid = frames[i - 1]
        fig = render_frame(play, fid, snap, off_team, def_team, cmap, show_alert)
        st.pyplot(fig)
        plt.close(fig)
        if st.button("Render animated GIF"):
            with st.spinner("Rendering GIF ..."):
                out = build_gif(play, frames, snap, off_team, def_team, cmap)
            st.success("Saved " + out)
            st.image(out)

    with right:
        st.subheader("Scout metrics")
        mt = metrics_table(play, names)
        st.dataframe(mt, use_container_width=True, height=340)
        st.subheader("Speed over play (fastest player)")
        ballmask = is_ball(play["team"])
        if not mt.empty:
            fastest = mt.iloc[0]["player"]
            fid_to_nid = {}
            for nid, g in play[~ballmask].groupby("nflId"):
                jersey = g["jerseyNumber"].iloc[0] if "jerseyNumber" in g.columns else np.nan
                if label(nid, jersey, names) == fastest:
                    fid_to_nid = {"nid": int(nid)}
                    break
            if fid_to_nid:
                gg = play[play["nflId"] == fid_to_nid["nid"]].sort_values("frameId")
                chart = pd.DataFrame({
                    "frame": range(1, len(gg) + 1),
                    "mph": (gg["s"] * MPH).to_numpy(),
                })
                st.line_chart(chart, x="frame", y="mph")

if __name__ == "__main__":
    main()