"""Analytics charts for the NFL scout app.

Each chart relates data layers (tracking motion + PFF outcomes + plays EPA)
so the output is scouting insight, not just a single column plotted.

Used by app.py:  import analytics; analytics.render_tab(t, plays, pff, names, gid)
Charts degrade gracefully when a file (pff / plays) is missing.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

MPH = 2.045454
FPS = 10

def _num(v):
    try:
        if v is None:
            return np.nan
        if isinstance(v, str) and v.strip().upper() in ("NA", "", "NAN"):
            return np.nan
        return float(v)
    except (ValueError, TypeError):
        return np.nan

def _is_ball(series):
    return series.astype(str).str.lower().isin(["football", "ball"])

def _label(nid, jersey, names):
    if nid in names and pd.notna(names.get(nid)):
        return str(names[nid])
    return "#" + str(int(jersey)) if pd.notna(jersey) else str(nid)

def _snap(play):
    if "event" in play.columns:
        s = play[play["event"].astype(str).str.lower() == "ball_snap"]
        if not s.empty:
            return int(s["frameId"].iloc[0])
    return int(play["frameId"].min())

def player_frame(t, pff, names, gid):
    """Build a per-player table for one game: motion + PFF outcome."""
    sub = t[t["gameId"] == gid]
    ball = _is_ball(sub["team"])
    rows = []
    for (pid, nid), g in sub[~ball].groupby(["playId", "nflId"]):
        g = g.sort_values("frameId")
        snap = _snap(g)
        cut = snap + FPS
        top = float(g["s"].max()) * MPH if "s" in g.columns else np.nan
        win = g[(g["frameId"] >= snap) & (g["frameId"] <= cut)]
        burst = np.nan
        if "s" in g.columns and len(win) >= 2:
            burst = float(win["s"].iloc[-1] - win["s"].iloc[0])
        jersey = g["jerseyNumber"].iloc[0] if "jerseyNumber" in g.columns else np.nan
        rows.append({
            "playId": pid, "nflId": int(nid),
            "player": _label(nid, jersey, names),
            "team": g["team"].iloc[0] if "team" in g.columns else "",
            "top_speed_mph": top, "getoff": burst,
        })
    df = pd.DataFrame(rows)
    if pff is not None and not df.empty:
        p = pff[pff["gameId"] == gid].copy()
        p["pressure"] = (p.get("pff_hit").map(_num).fillna(0)
                         + p.get("pff_hurry").map(_num).fillna(0)
                         + p.get("pff_sack").map(_num).fillna(0))
        p["beaten"] = p.get("pff_beatenByDefender").map(_num).fillna(0)
        keep = p[["playId", "nflId", "pff_role", "pressure", "beaten"]]
        df = df.merge(keep, on=["playId", "nflId"], how="left")
    return df

def chart_top_speed(df):
    agg = (df.groupby("player")["top_speed_mph"].max()
           .dropna().sort_values(ascending=False).head(12))
    if agg.empty:
        st.info("No speed data.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(agg.index[::-1], agg.values[::-1], color="#1f77b4")
    ax.set_xlabel("Top speed (mph)")
    ax.set_title("Top speed leaderboard")
    for i, v in enumerate(agg.values[::-1]):
        ax.text(v + 0.1, i, format(v, ".1f"), va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def chart_pass_rush(df):
    if "pressure" not in df.columns:
        st.info("pffScoutingData.csv needed for pass-rush win rate.")
        return
    rush = df[df["pff_role"].astype(str).str.contains("Rush", na=False)]
    if rush.empty:
        st.info("No pass-rush snaps in this game.")
        return
    agg = rush.groupby("player").agg(
        reps=("pressure", "size"), pressures=("pressure", "sum")).reset_index()
    agg = agg[agg["reps"] >= 3]
    agg["win_rate"] = (agg["pressures"] / agg["reps"] * 100).round(1)
    agg = agg.sort_values("win_rate", ascending=False).head(12)
    if agg.empty:
        st.info("Not enough rush reps (need >=3 per player).")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(agg["player"][::-1], agg["win_rate"][::-1], color="#d62728")
    ax.set_xlabel("Pressure rate (%)")
    ax.set_title("Pass-rush pressure rate (min 3 reps)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def chart_pressure_pie(df):
    st.caption("Pressure type mix needs the raw PFF columns; shown via roles.")
    if "pff_role" not in df.columns:
        st.info("pffScoutingData.csv needed for role breakdown.")
        return
    counts = (df["pff_role"].dropna().astype(str)
              .value_counts().head(6))
    if counts.empty:
        st.info("No role data.")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 8})
    ax.set_title("Player-role distribution")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def chart_speed_vs_getoff(df):
    d = df.dropna(subset=["top_speed_mph", "getoff"])
    d = d.groupby("player").agg(
        top_speed_mph=("top_speed_mph", "max"),
        getoff=("getoff", "mean")).reset_index()
    if d.empty:
        st.info("No data for scatter.")
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(d["getoff"], d["top_speed_mph"], color="#2ca02c", alpha=0.7)
    ax.set_xlabel("Avg get-off burst (yd/s in first 1s)")
    ax.set_ylabel("Top speed (mph)")
    ax.set_title("Athletic archetypes: speed vs explosiveness")
    top = d.sort_values("top_speed_mph", ascending=False).head(5)
    for _, r in top.iterrows():
        ax.annotate(r["player"], (r["getoff"], r["top_speed_mph"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def chart_speed_over_time(t, names, gid, pid):
    play = t[(t["gameId"] == gid) & (t["playId"] == pid)].sort_values("frameId")
    ball = _is_ball(play["team"])
    opts = {}
    for nid, g in play[~ball].groupby("nflId"):
        jersey = g["jerseyNumber"].iloc[0] if "jerseyNumber" in g.columns else np.nan
        opts[_label(nid, jersey, names)] = int(nid)
    if not opts:
        st.info("No players on this play.")
        return
    pick = st.selectbox("Player for speed-over-time", sorted(opts.keys()))
    g = play[play["nflId"] == opts[pick]].sort_values("frameId")
    chart = pd.DataFrame({
        "frame": range(1, len(g) + 1),
        "mph": (g["s"] * MPH).to_numpy(),
    })
    st.line_chart(chart, x="frame", y="mph")

def render_tab(t, plays, pff, names, gid):
    st.header("Scouting analytics")
    st.caption("Charts relate motion (tracking), outcomes (PFF) and value "
               "(EPA) across this game. Load more games for richer trends.")
    df = player_frame(t, pff, names, gid)
    if df.empty:
        st.warning("No player data for this game.")
        return

    a, b = st.columns(2)
    with a:
        chart_top_speed(df)
        chart_pressure_pie(df)
    with b:
        chart_pass_rush(df)
        chart_speed_vs_getoff(df)

    st.subheader("Speed over time (single play)")
    pids = sorted(t[t["gameId"] == gid]["playId"].dropna().unique().tolist())
    if pids:
        pid = st.selectbox("Play for time series", pids, key="ts_play")
        chart_speed_over_time(t, names, gid, int(pid))

    with st.expander("Show raw per-player table"):
        st.dataframe(df, use_container_width=True)