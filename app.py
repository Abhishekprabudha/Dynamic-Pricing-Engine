import time
from pathlib import Path
from datetime import datetime, timedelta
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page setup + CSS tightening
# -----------------------------
st.set_page_config(page_title="Dynamic Pricing Engine (Capacity-Based)", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.0rem;}
      .stMetric {padding: 6px 10px;}
      div[data-testid="stVerticalBlockBorderWrapper"] {padding: 10px;}
      .tight-card {padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(49,51,63,0.15);}
      .muted {opacity: 0.75;}
      .pill {display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.18);}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚚✈️📦 Dynamic Pricing Engine — Capacity-Based")
st.caption(
    "Left: MP4 loading feed + asset selector | Right: Capacity telemetry + tiered pricing + graph + calculator."
)

# -----------------------------
# Pricing rules (as requested)
# -----------------------------
MARKET_RATE = 200.0  # $/ton
TIER_1_MAX = 0.60
TIER_2_MAX = 0.85
FACTOR_BELOW_60 = 0.80
FACTOR_60_85 = 1.00
FACTOR_ABOVE_85 = 1.25


def pricing_factor(fill_ratio: float) -> float:
    if fill_ratio < TIER_1_MAX:
        return FACTOR_BELOW_60
    if fill_ratio <= TIER_2_MAX:
        return FACTOR_60_85
    return FACTOR_ABOVE_85


def tier_label(fill_ratio: float) -> str:
    if fill_ratio < TIER_1_MAX:
        return "Below 60% (Discount)"
    if fill_ratio <= TIER_2_MAX:
        return "60–85% (Market)"
    return "Above 85% (Premium)"


def rate_per_ton(fill_ratio: float) -> float:
    return MARKET_RATE * pricing_factor(fill_ratio)


# -----------------------------
# Video loading
# -----------------------------
VIDEO_DIR = Path("videos")
fallback_uploaded = Path("/mnt/data/3817769649-preview.mp4")  # optional user-uploaded file in this env

video_files = []
if VIDEO_DIR.exists():
    video_files = sorted([p for p in VIDEO_DIR.glob("*.mp4")])

if not video_files and fallback_uploaded.exists():
    video_files = [fallback_uploaded]

if not video_files:
    st.error("❌ No MP4 found. Create /videos and add MP4 files, or provide a valid MP4 path.")
    st.stop()

# -----------------------------
# Asset catalog (demo)
# -----------------------------
ASSET_CATALOG = {
    "Truck (FTL)": {"capacity_tons": 20.0, "load_profile": "smooth"},
    "Shipping Container (20ft)": {"capacity_tons": 22.0, "load_profile": "bursty"},
    "Aircraft Cargo (ULD)": {"capacity_tons": 15.0, "load_profile": "bursty"},
}

# -----------------------------
# Sidebar controls (global)
# -----------------------------
with st.sidebar:
    st.header("Controls")
    autoplay = st.toggle("Autoplay telemetry", value=True)
    tick_ms = st.slider("Refresh speed (ms)", 150, 1500, 350, 10)

    st.divider()
    st.subheader("Video selection")

    if "video_idx" not in st.session_state:
        st.session_state.video_idx = 0

    chosen = st.selectbox(
        "Pick a video",
        options=list(range(len(video_files))),
        format_func=lambda i: f"{i+1}. {video_files[i].name}",
        index=st.session_state.video_idx
    )
    st.session_state.video_idx = chosen

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("⏮ Prev"):
            st.session_state.video_idx = (st.session_state.video_idx - 1) % len(video_files)
    with colB:
        if st.button("▶ Next"):
            st.session_state.video_idx = (st.session_state.video_idx + 1) % len(video_files)
    with colC:
        if st.button("🔁 Reset"):
            st.session_state.video_idx = 0

    st.divider()
    st.subheader("Pricing & loading realism (demo)")
    loading_speed = st.slider("Loading speed", 0.5, 3.0, 1.4, 0.1)   # affects curve
    noise = st.slider("Signal noise", 0.0, 3.0, 0.6, 0.1)            # affects jitter
    burstiness = st.slider("Burstiness", 0.0, 2.5, 1.0, 0.1)         # affects step-like bursts


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.25, 1.0], gap="large")
current_video = video_files[st.session_state.video_idx]


# -----------------------------
# Telemetry generation (capacity fill over time)
# -----------------------------
def make_loading_series(
    seed: int,
    asset_cfg: dict,
    n: int = 320,
    noise: float = 0.6,
    loading_speed: float = 1.4,
    burstiness: float = 1.0
):
    """
    Returns:
      t: ticks
      fill_ratio: [0..1]
      loaded_tons: tons loaded
      inst_load_rate_tph: instantaneous loading rate (tons/hr) proxy
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    cap = float(asset_cfg["capacity_tons"])
    profile = asset_cfg.get("load_profile", "smooth")

    # Base smooth curve (logistic-ish)
    x = (t - n * 0.35) / (n * 0.12)
    smooth = 1 / (1 + np.exp(-loading_speed * x))  # 0..1

    # Make it start near 0
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)

    # Add bursty behavior (step increments) for certain profiles
    burst = np.zeros(n)
    if profile == "bursty" or burstiness > 0:
        for k in range(15, n, rng.integers(18, 35)):
            width = int(rng.integers(2, 7))
            inc = float(rng.uniform(0.015, 0.06)) * (0.6 + 0.5 * burstiness)
            burst[k:k + width] += inc / max(width, 1)

        burst = np.cumsum(burst)
        burst = burst / (burst.max() + 1e-9) * (0.18 * burstiness)

    fill = smooth + burst

    # Add noise + clamp
    fill += rng.normal(0, noise * 0.01, size=n)
    fill = np.clip(fill, 0, 1.0)

    # Ensure monotonic-ish increasing (loading typically doesn't decrease)
    fill = np.maximum.accumulate(fill)

    loaded_tons = fill * cap

    # Instantaneous "rate" proxy: delta tons per tick -> tons/hr
    tick_minutes = 6.0
    dtons = np.diff(loaded_tons, prepend=loaded_tons[0])
    inst_rate_tph = (dtons / (tick_minutes / 60.0))
    inst_rate_tph = np.clip(inst_rate_tph, 0, None)

    return t, fill, loaded_tons, inst_rate_tph


# -----------------------------
# Cursor + autoplay
# -----------------------------
if "cursor" not in st.session_state:
    st.session_state.cursor = 0

# Reset cursor when video/asset changes
if "asset" not in st.session_state:
    st.session_state.asset = list(ASSET_CATALOG.keys())[0]

current_key = (current_video.name, st.session_state.asset)
if st.session_state.get("last_key") != current_key:
    st.session_state.last_key = current_key
    st.session_state.cursor = 0

cursor = int(np.clip(st.session_state.cursor, 0, 319))
st.session_state.cursor = cursor

if autoplay:
    st.session_state.cursor = min(st.session_state.cursor + 2, 319)
    time.sleep(tick_ms / 1000.0)
    st.rerun()


# -----------------------------
# LEFT: Asset selection + video + executive summary + quick query
# -----------------------------
with left:
    st.subheader("🧾 Asset Selector")

    asset_names = list(ASSET_CATALOG.keys())
    asset_choice = st.selectbox("Select asset", asset_names, index=asset_names.index(st.session_state.asset))
    st.session_state.asset = asset_choice
    asset_cfg = ASSET_CATALOG[asset_choice]

    cap_tons = float(asset_cfg["capacity_tons"])

    st.markdown(
        f"<span class='pill'><b>Active:</b> {asset_choice} | <b>Capacity:</b> {cap_tons:.1f} tons</span>",
        unsafe_allow_html=True
    )

    st.subheader("🎥 Live Loading Feed")
    st.write(f"**Now playing:** {current_video.name}")
    st.video(str(current_video))

# -----------------------------
# Generate telemetry (seeded by video + asset)
# -----------------------------
seed = abs(hash((current_video.name, st.session_state.asset))) % (10**6)
t, fill_ratio, loaded_tons, inst_rate_tph = make_loading_series(
    seed,
    asset_cfg=asset_cfg,
    noise=noise,
    loading_speed=loading_speed,
    burstiness=burstiness
)

# Slice up to cursor
tt = t[:cursor + 1]
ff = fill_ratio[:cursor + 1]
lt = loaded_tons[:cursor + 1]
rr = inst_rate_tph[:cursor + 1]

fill_now = float(ff[-1])
tons_now = float(lt[-1])
factor_now = pricing_factor(fill_now)
rate_now = rate_per_ton(fill_now)
tier_now = tier_label(fill_now)

# Simple ETA to 60/85/100 based on recent slope
def eta_to_target(target_ratio: float, window: int = 25) -> float:
    if fill_now >= target_ratio:
        return 0.0
    w = min(window, len(ff) - 1)
    if w <= 3:
        return float("inf")
    # average slope per tick
    slope = float(np.mean(np.diff(ff[-w:])))
    if slope <= 1e-6:
        return float("inf")
    ticks_needed = (target_ratio - fill_now) / slope
    tick_minutes = 6.0
    return max(0.0, ticks_needed * tick_minutes / 60.0)

eta_60 = eta_to_target(TIER_1_MAX)
eta_85 = eta_to_target(TIER_2_MAX)
eta_100 = eta_to_target(1.0)

def fmt_eta(hours: float) -> str:
    if hours == float("inf"):
        return "—"
    if hours <= 0:
        return "Now"
    if hours < 1:
        return f"{hours*60:.0f} min"
    return f"{hours:.1f} hrs"


# -----------------------------
# GENBI: rule-based query engine (offline)
# -----------------------------
def genbi_answer(q: str, cursor_now: int):
    ql = q.strip().lower()
    if not ql:
        return None, None

    asset_label = st.session_state.asset

    if ("fill" in ql or "capacity" in ql) and ("current" in ql or "now" in ql):
        return (f"[{asset_label}] Current fill is **{fill_now*100:.1f}%** "
                f"(**{tons_now:.2f} / {cap_tons:.1f} tons**). Tier: **{tier_now}**."), None

    if "rate" in ql or "price" in ql:
        return (f"[{asset_label}] Market rate is **${MARKET_RATE:.0f}/ton**. "
                f"At **{fill_now*100:.1f}%** fill → factor **{factor_now:.2f}x** → **${rate_now:.0f}/ton** "
                f"({tier_now})."), None

    if "when" in ql and ("60" in ql or "85" in ql or "100" in ql or "full" in ql):
        return (f"[{asset_label}] ETA estimates (demo): "
                f"60% → **{fmt_eta(eta_60)}**, 85% → **{fmt_eta(eta_85)}**, Full → **{fmt_eta(eta_100)}**."), None

    # trend plot request
    m = re.search(r"last\s+(\d+)\s+ticks", ql)
    n = int(m.group(1)) if m else 120
    n = int(np.clip(n, 20, 240))
    s = max(0, cursor_now - n)

    def line_fig(x, y, name, ytitle):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Tick", yaxis_title=ytitle)
        return fig

    if "fill" in ql or "capacity" in ql:
        fig = line_fig(t[s:cursor_now + 1], fill_ratio[s:cursor_now + 1] * 100.0, "Capacity fill %", "Fill (%)")
        fig.add_hline(y=TIER_1_MAX * 100, line_width=1)
        fig.add_hline(y=TIER_2_MAX * 100, line_width=1)
        return f"[{asset_label}] Showing last **{cursor_now - s}** ticks of **Fill %** (includes 60% & 85% lines).", fig

    if "rate" in ql or "price" in ql:
        rates = np.array([rate_per_ton(x) for x in fill_ratio[s:cursor_now + 1]])
        fig = line_fig(t[s:cursor_now + 1], rates, "Dynamic rate ($/ton)", "Rate ($/ton)")
        return f"[{asset_label}] Showing last **{cursor_now - s}** ticks of **Dynamic Rate**.", fig

    return (f"[{asset_label}] Try: 'current fill now', 'current rate', "
            f"'when will it cross 60/85/full', 'show last 150 ticks fill', 'show last 150 ticks rate'."), None


# -----------------------------
# LEFT: Executive summary + quick query
# -----------------------------
with left:
    st.markdown('<div class="tight-card">', unsafe_allow_html=True)
    st.markdown(f"### 📌 Pricing Executive Summary — {st.session_state.asset}")

    a, b, c = st.columns(3)
    a.metric("Fill (%)", f"{fill_now*100:.1f}%")
    b.metric("Loaded (tons)", f"{tons_now:.2f} / {cap_tons:.1f}")
    c.metric("Tier", tier_now)

    d, e, f = st.columns(3)
    d.metric("Market rate", f"${MARKET_RATE:.0f}/ton")
    e.metric("Factor", f"{factor_now:.2f}x")
    f.metric("Dynamic rate", f"${rate_now:.0f}/ton")

    st.markdown(
        f"<span class='muted'>Rule: &lt;60% → {FACTOR_BELOW_60:.2f}x | 60–85% → {FACTOR_60_85:.2f}x | &gt;85% → {FACTOR_ABOVE_85:.2f}x</span>",
        unsafe_allow_html=True
    )

    st.markdown("#### 🔎 GenBI Quick Query")
    quick_q = st.text_input(
        "Ask about fill, rate, ETA, trends…",
        placeholder="e.g., 'current rate' or 'show last 150 ticks fill' or 'when will it cross 85%'"
    )
    st.markdown("</div>", unsafe_allow_html=True)

quick_answer, quick_fig = genbi_answer(quick_q, cursor) if quick_q else (None, None)
if quick_q and quick_answer:
    with left:
        st.info(quick_answer)
        if quick_fig is not None:
            st.plotly_chart(quick_fig, use_container_width=True)


# -----------------------------
# RIGHT: KPIs + Tabs (Graph + Calculator)
# -----------------------------
with right:
    st.subheader(f"📟 Capacity & Pricing Dashboard — {st.session_state.asset}")

    r1, r2, r3 = st.columns(3)
    r1.metric("Fill (%)", f"{fill_now*100:.1f}%")
    r2.metric("Dynamic rate", f"${rate_now:.0f}/ton")
    r3.metric("Instant load rate (tph)", f"{rr[-1]:.1f}")

    r4, r5, r6 = st.columns(3)
    r4.metric("ETA to 60%", fmt_eta(eta_60))
    r5.metric("ETA to 85%", fmt_eta(eta_85))
    r6.metric("ETA to Full", fmt_eta(eta_100))

    tabs = st.tabs(["📈 Live Graph", "🧮 Pricing Calculator", "💬 GenBI Query"])

    with tabs[0]:
        window = 160
        start = max(0, cursor - window)

        # Build dynamic rate series for the window
        win_fill = fill_ratio[start:cursor + 1]
        win_rate = np.array([rate_per_ton(x) for x in win_fill])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[start:cursor + 1], y=win_fill * 100.0, mode="lines", name="Fill (%)"))
        fig.add_trace(go.Scatter(x=t[start:cursor + 1], y=win_rate, mode="lines", name="Rate ($/ton)", yaxis="y2"))

        fig.add_hline(y=TIER_1_MAX * 100, line_width=1)
        fig.add_hline(y=TIER_2_MAX * 100, line_width=1)
        fig.add_vline(x=t[cursor], line_width=2)

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Telemetry Tick",
            yaxis=dict(title="Fill (%)", rangemode="tozero"),
            yaxis2=dict(title="Rate ($/ton)", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

        colX, colY = st.columns([1, 2])
        with colX:
            if st.button("⏩ Advance telemetry"):
                st.session_state.cursor = min(st.session_state.cursor + 10, len(t) - 1)
                st.rerun()
        with colY:
            st.progress(int((cursor / (len(t) - 1)) * 100))

        st.markdown(
            "<div class='muted'>Interpretation: rate is discounted until 60% fill, market-aligned between 60–85%, "
            "and premium after 85%.</div>",
            unsafe_allow_html=True
        )

    with tabs[1]:
        st.markdown("### 🧮 Capacity → Rate Calculator")

        calc_fill = st.slider("Capacity fill (%)", 0.0, 100.0, float(fill_now * 100.0), 0.5)
        calc_ratio = calc_fill / 100.0

        calc_factor = pricing_factor(calc_ratio)
        calc_rate = rate_per_ton(calc_ratio)
        calc_tier = tier_label(calc_ratio)

        c1, c2, c3 = st.columns(3)
        c1.metric("Tier", calc_tier)
        c2.metric("Factor vs market", f"{calc_factor:.2f}x")
        c3.metric("Rate ($/ton)", f"${calc_rate:.0f}/ton")

        st.markdown("#### Optional: total charge for a shipment")
        wt = st.number_input("Shipment weight (tons)", min_value=0.0, value=min(5.0, cap_tons), step=0.5)
        total = wt * calc_rate
        st.metric("Total charge", f"${total:,.0f}")

        # Step function visualization (clear tier breakpoints)
        xs = np.linspace(0, 100, 301)
        ys = np.array([rate_per_ton(x/100.0) for x in xs])

        step_fig = go.Figure()
        step_fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Rate curve"))
        step_fig.add_vline(x=60, line_width=1)
        step_fig.add_vline(x=85, line_width=1)
        step_fig.add_hline(y=MARKET_RATE, line_width=1)

        # Marker at current calculator point
        step_fig.add_trace(go.Scatter(
            x=[calc_fill],
            y=[calc_rate],
            mode="markers",
            name="Selected fill"
        ))

        step_fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Capacity fill (%)",
            yaxis_title="Rate ($/ton)"
        )
        st.plotly_chart(step_fig, use_container_width=True)

        st.markdown(
            f"<div class='muted'>Using market ${MARKET_RATE:.0f}/ton: "
            f"&lt;60% → {FACTOR_BELOW_60:.2f}x (${MARKET_RATE*FACTOR_BELOW_60:.0f}/ton), "
            f"60–85% → {FACTOR_60_85:.2f}x (${MARKET_RATE*FACTOR_60_85:.0f}/ton), "
            f"&gt;85% → {FACTOR_ABOVE_85:.2f}x (${MARKET_RATE*FACTOR_ABOVE_85:.0f}/ton).</div>",
            unsafe_allow_html=True
        )

    with tabs[2]:
        st.markdown("### 💬 GenBI Query")
        st.caption("Plain English (rule-based/offline). Upgrade to LLM later.")

        q = st.text_input("Your question", placeholder="e.g., What is current fill and what rate are we charging now?")
        ans, fig2 = genbi_answer(q, cursor) if q else (None, None)
        if ans:
            st.info(ans)
        if fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True)
