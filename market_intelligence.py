
import json
import re
import sqlite3
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from market_core import (
    analyze,
    format_central,
    get_latest_bar,
    get_news,
    get_prices,
    now_central,
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent

GREEN = "#39E6A5"
RED = "#FF7185"
AMBER = "#FFC778"
BLUE = "#85AEFF"
PURPLE = "#B9A0FF"
MUTED = "#A4B3C7"

VALID = re.compile(r"^[A-Z0-9^][A-Z0-9.^=-]{0,19}$")


def money(value):
    return f"${float(value):,.2f}"


def html(content):
    # Prevent indented HTML becoming visible Markdown code.
    flattened = re.sub(r"\s*\n\s*", " ", str(content))
    st.markdown(flattened.strip(), unsafe_allow_html=True)


def color_for_signal(signal):
    if signal in ("BUY ZONE", "BULLISH", "BREAKOUT"):
        return GREEN
    if signal in ("SELL ZONE", "BEARISH", "BREAKDOWN"):
        return RED
    return AMBER


def color_for_change(change):
    return GREEN if change >= 0 else RED


def pill(text, color):
    return (
        f'<span class="mi-pill" '
        f'style="color:{color};background:{color}20;'
        f'border:1px solid {color}55">'
        f'{escape(str(text))}</span>'
    )


def panel(title, description):
    html(f"""
    <div class="mi-panel">
        <h4>{escape(title)}</h4>
        <p>{escape(description)}</p>
    </div>
    """)


def action_card(title, value, description, color):
    html(f"""
    <div class="mi-card" style="border-top:3px solid {color}">
        <div class="mi-eyebrow" style="color:{color}">
            {escape(title)}
        </div>
        <div class="mi-card-price" style="color:{color}">
            {escape(value)}
        </div>
        <div class="mi-card-body">
            {escape(description)}
        </div>
    </div>
    """)


# ============================================================
# THEME
# ============================================================

st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 85% 0%,#17273b,transparent 35%),
        radial-gradient(ellipse at 0% 35%,#112b27,transparent 36%),
        #090e16;
    color:#f4f7fc;
}

[data-testid="stHeader"] {
    background:transparent;
}

.block-container {
    max-width:1190px;
    padding-top:1.1rem;
    padding-bottom:4rem;
}

h1,h2,h3 {
    letter-spacing:-.04em !important;
}

h1 {
    font-weight:850 !important;
}

p,li {
    color:#c4cfdd;
}

[data-testid="stMetric"] {
    background:linear-gradient(140deg,#192637,#111b29);
    border:1px solid #304156;
    border-radius:17px;
    padding:17px;
}

[data-testid="stMetricLabel"] {
    color:#a4b3c7;
}

[data-testid="stMetricValue"] {
    color:#f7f9fd;
}

.stButton button,
.stDownloadButton button {
    border-radius:12px !important;
    min-height:44px;
    font-weight:750;
    border-color:#33465a;
    background:#172434;
    color:#edf4ff;
}

.stButton button[kind="primary"] {
    background:#c9fa75;
    color:#0a1510;
    border:none;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap:22px;
    overflow-x:auto;
    border-bottom:1px solid #304156;
}

[data-testid="stTabs"] button {
    font-weight:800;
    color:#9eb0c5;
    white-space:nowrap;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color:#39e6a5 !important;
}

[data-testid="stTabs"]
[data-baseweb="tab-highlight"] {
    background:#39e6a5;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div {
    background:#142132 !important;
    color:#f6f9fc !important;
    border-radius:12px !important;
    border-color:#304156 !important;
}

.mi-hero {
    background:linear-gradient(115deg,#1b3036,#17243a 65%,#26233d);
    border:1px solid #3b5062;
    border-radius:25px;
    padding:28px;
    margin:15px 0 22px;
}

.mi-eyebrow {
    color:#a6bbcf;
    font-size:12px;
    font-weight:850;
    letter-spacing:1.4px;
    text-transform:uppercase;
}

.mi-price {
    font-size:clamp(40px,6vw,67px);
    letter-spacing:-3px;
    color:#fff;
    font-weight:850;
    margin:7px 0;
}

.mi-pill {
    display:inline-block;
    padding:6px 11px;
    border-radius:999px;
    font-size:12px;
    font-weight:850;
    letter-spacing:.2px;
}

.mi-card {
    background:linear-gradient(145deg,#1b2a3b,#121d2a);
    border:1px solid #34465a;
    border-radius:20px;
    padding:20px;
    margin:6px 0 15px;
    min-height:172px;
}

.mi-card-price {
    font-size:clamp(21px,2.5vw,30px);
    font-weight:850;
    letter-spacing:-1px;
    margin:15px 0 8px;
}

.mi-card-body {
    font-size:13px;
    color:#b7c6d7;
    line-height:1.65;
}

.mi-panel {
    background:linear-gradient(125deg,#172637,#101b29);
    border:1px solid #304156;
    border-radius:18px;
    padding:19px;
    margin:12px 0 17px;
}

.mi-panel h4 {
    color:#f4f7fc;
    margin:0 0 8px;
    font-size:17px;
}

.mi-panel p {
    color:#b7c6d7;
    font-size:14px;
    margin:0;
    line-height:1.65;
}

.mi-stock {
    background:linear-gradient(145deg,#19283a,#111c29);
    border:1px solid #34465a;
    border-left:4px solid var(--accent);
    border-radius:18px;
    padding:19px;
    margin-bottom:12px;
}

.mi-stock-top {
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:12px;
    margin-bottom:14px;
}

.mi-stock-symbol {
    font-size:23px;
    font-weight:850;
    color:#f4f7fc;
}

.mi-stock-price {
    font-size:25px;
    font-weight:850;
    color:#f4f7fc;
    margin-bottom:10px;
}

.mi-stock-bottom {
    display:flex;
    gap:8px;
    flex-wrap:wrap;
}

.mi-gallery-header {
    background:linear-gradient(115deg,#1a2a3e,#121e2d);
    border:1px solid #33485e;
    border-radius:20px;
    padding:20px;
    margin:12px 0 14px;
}

.mi-gallery-counter {
    font-size:12px;
    font-weight:850;
    letter-spacing:1.2px;
    color:#39e6a5;
}

.mi-gallery-title {
    font-size:25px;
    font-weight:850;
    color:#f4f7fc;
    margin:9px 0;
}

.mi-gallery-description {
    color:#b7c6d7;
    line-height:1.65;
    font-size:14px;
}

@media(max-width:650px) {
    .block-container {
        padding:.7rem .65rem 3rem;
    }
    .mi-hero {
        padding:19px;
    }
    .mi-card {
        min-height:0;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def prices(symbol):
    return get_prices(symbol)


@st.cache_data(ttl=300, show_spinner=False)
def latest_price(symbol):
    return get_latest_bar(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def headlines(symbol):
    return get_news(symbol)


# ============================================================
# WATCHLIST STORAGE
# ============================================================

try:
    config = json.loads(
        (ROOT / "alert_config.json").read_text()
    )
except (OSError, ValueError):
    config = {}

defaults = config.get(
    "watchlist",
    ["NVDA", "CRDO", "VOO", "AAPL", "MSFT", "AMD"],
)

if not isinstance(defaults, list):
    defaults = []

defaults = list(dict.fromkeys(
    str(x).upper()
    for x in defaults
    if VALID.fullmatch(str(x).upper())
))

if not defaults:
    defaults = ["NVDA", "CRDO", "VOO"]

DB = ROOT / "personal_watchlist.db"


def watchlist_db():
    conn = sqlite3.connect(DB, timeout=15)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            added_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist_settings (
            setting TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    conn.commit()
    return conn


def initialize_watchlist():
    with watchlist_db() as conn:
        initialized = conn.execute("""
            SELECT value
            FROM watchlist_settings
            WHERE setting='initialized'
        """).fetchone()

        if initialized:
            return

        count = conn.execute(
            "SELECT COUNT(*) FROM watchlist"
        ).fetchone()[0]

        if count == 0:
            for ticker in defaults:
                conn.execute("""
                    INSERT OR IGNORE INTO watchlist
                    VALUES (?, ?)
                """, (
                    ticker,
                    now_central().isoformat(),
                ))

        conn.execute("""
            INSERT OR REPLACE INTO watchlist_settings
            VALUES ('initialized', '1')
        """)

        conn.commit()


def get_watchlist():
    with watchlist_db() as conn:
        rows = conn.execute("""
            SELECT ticker
            FROM watchlist
            ORDER BY added_at,ticker
        """).fetchall()

    return [row[0] for row in rows]


def add_watchlist(ticker):
    if not VALID.fullmatch(ticker):
        return False

    try:
        data = prices(ticker)
        if data is None or data.empty:
            return False
    except Exception:
        return False

    with watchlist_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO watchlist
            VALUES (?, ?)
        """, (
            ticker,
            now_central().isoformat(),
        ))
        conn.commit()

    return True


def remove_watchlist(ticker):
    with watchlist_db() as conn:
        conn.execute(
            "DELETE FROM watchlist WHERE ticker=?",
            (ticker,),
        )
        conn.commit()


initialize_watchlist()


# ============================================================
# PRICE ZONES
# ============================================================

def price_levels(df):
    if len(df) < 65:
        return None

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    previous = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous).abs(),
        (low - previous).abs(),
    ], axis=1).max(axis=1)

    atr = float(tr.tail(14).mean())
    support = float(low.tail(60).min())
    resistance = float(high.tail(60).max())

    if (
        not np.isfinite(atr)
        or atr <= 0
        or resistance <= support
    ):
        return None

    width = min(
        atr * 0.5,
        (resistance - support) * 0.2,
    )

    return {
        "buy_low": support,
        "buy_high": support + width,
        "sell_low": resistance - width,
        "sell_high": resistance,
        "risk": max(0.01, support - atr),
        "support": support,
        "resistance": resistance,
        "atr": atr,
    }


def price_state(current, zones):
    if zones is None:
        return "UNAVAILABLE"

    if current < zones["risk"]:
        return "BREAKDOWN"

    if current <= zones["buy_high"]:
        return "BUY ZONE"

    if current > zones["sell_high"]:
        return "BREAKOUT"

    if current >= zones["sell_low"]:
        return "SELL ZONE"

    return "HOLD / WAIT"


def price_explanation(state):
    explanations = {
        "BUY ZONE":
            "Price is near a recent low. "
            "Research a possible entry, but check "
            "why the stock declined.",

        "HOLD / WAIT":
            "Price is between recent support and "
            "resistance. Monitor for a clearer entry.",

        "SELL ZONE":
            "Price is near a recent high. "
            "Existing holders may review profit-taking.",

        "BREAKDOWN":
            "Price fell below historical support. "
            "Downside risk may be elevated.",

        "BREAKOUT":
            "Price exceeded its recent high. "
            "Momentum may be changing.",

        "UNAVAILABLE":
            "Insufficient history for price zones.",
    }

    return explanations[state]


def trend_signal(info):
    if info["score"] >= 3 and info["rsi"] < 70:
        return "BULLISH"
    if info["score"] <= 1:
        return "BEARISH"
    return "MIXED"


# ============================================================
# CHART DESIGN
# ============================================================

def chart_style(fig, height=430, percent=False):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#D2DEEC",
            size=13,
        ),
        margin=dict(
            l=30,
            r=40,
            t=60,
            b=45,
        ),
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            gridcolor="#2C4055",
            tickfont=dict(size=12),
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            y=1.16,
            x=0,
            font=dict(size=12),
        ),
    )

    if percent:
        fig.update_yaxes(ticksuffix="%")

    return fig


def show_chart(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": False,
            "responsive": True,
        },
    )


def stock_chart(df, zones=None):
    close = df["Close"].astype(float)

    color = GREEN if (
        close.iloc[-1] >= close.iloc[0]
    ) else RED

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=close,
        name="Stock price",
        line=dict(color=color, width=3),
    ))

    if zones:
        fig.add_hrect(
            y0=zones["buy_low"],
            y1=zones["buy_high"],
            fillcolor=GREEN,
            opacity=0.16,
            line_width=0,
        )

        fig.add_hrect(
            y0=zones["sell_low"],
            y1=zones["sell_high"],
            fillcolor=RED,
            opacity=0.16,
            line_width=0,
        )

        fig.add_hline(
            y=zones["risk"],
            line_color=AMBER,
            line_dash="dot",
        )

    chart_style(fig, 390)
    fig.update_yaxes(tickprefix="$", side="right")
    fig.update_layout(showlegend=False)

    return fig


# ============================================================
# ANALYTICS CHART GALLERY
# ============================================================

ANALYTICS_SLIDES = [
    {
        "title": "Price position",
        "description":
            "Where the current price sits between its "
            "recent 60-day low and high.",
        "explanation":
            "Closer to the low can indicate a potential "
            "support area. Closer to the high can indicate "
            "resistance. Neither guarantees a reversal.",
    },
    {
        "title": "Moving averages",
        "description":
            "Compare stock price against 20-day "
            "and 50-day trends.",
        "explanation":
            "Green is the shorter-term average. "
            "Amber is the longer-term average. "
            "Crossovers can signal changes in momentum.",
    },
    {
        "title": "Trading volume",
        "description":
            "See how much trading activity occurs "
            "during each session.",
        "explanation":
            "Green bars indicate sessions closing at or "
            "above their open. Red bars indicate sessions "
            "closing below their open. Higher volume can "
            "indicate stronger participation.",
    },
    {
        "title": "Technical health",
        "description":
            "Four technical checks displayed "
            "as an easy-to-read scorecard.",
        "explanation":
            "Green means a condition is met. "
            "Red means it is not met. "
            "These are momentum indicators, "
            "not guaranteed buy or sell signals.",
    },
    {
        "title": "Daily returns",
        "description":
            "Understand how frequently the stock "
            "experiences different daily percentage moves.",
        "explanation":
            "A wider distribution means larger daily "
            "moves have occurred more often. "
            "This helps assess historical volatility.",
    },
    {
        "title": "Historical drawdown",
        "description":
            "See how far the stock has fallen "
            "from earlier price peaks.",
        "explanation":
            "Deeper drawdowns mean larger historical "
            "declines. They do not establish a maximum "
            "possible future loss.",
    },
    {
        "title": "RSI momentum",
        "description":
            "Track relative strength over time.",
        "explanation":
            "RSI above 70 is often described as "
            "overbought; below 30 as oversold. "
            "Stocks can remain in either condition.",
    },
]


def chart_price_position(df, zones, current):
    if not zones:
        return None

    low = zones["support"]
    high = zones["resistance"]

    position = np.clip(
        (current - low) / (high - low) * 100,
        0,
        100,
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(position),
        number=dict(
            suffix="%",
            font=dict(size=50, color="#FFFFFF"),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickfont=dict(
                    color="#D2DEEC",
                    size=13,
                ),
            ),
            bar=dict(
                color="#FFFFFF",
                thickness=0.18,
            ),
            bgcolor="#172434",
            borderwidth=0,
            steps=[
                dict(
                    range=[0, 25],
                    color="#205A49",
                ),
                dict(
                    range=[25, 75],
                    color="#67522E",
                ),
                dict(
                    range=[75, 100],
                    color="#703C4C",
                ),
            ],
        ),
    ))

    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=45,
            r=45,
            t=45,
            b=25,
        ),
    )

    return fig


def chart_moving_averages(df):
    close = df["Close"].astype(float)

    fig = go.Figure()

    for name, values, color in [
        ("Stock price", close, BLUE),
        ("20-day average", close.rolling(20).mean(), GREEN),
        ("50-day average", close.rolling(50).mean(), AMBER),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=values,
            name=name,
            line=dict(
                color=color,
                width=3,
            ),
        ))

    chart_style(fig)
    fig.update_yaxes(tickprefix="$")

    return fig


def chart_volume(df):
    data = df.tail(90)

    colors = np.where(
        data["Close"] >= data["Open"],
        GREEN,
        RED,
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Daily volume",
        marker_color=colors,
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["Volume"].rolling(20).mean(),
        name="20-day average",
        line=dict(
            color=AMBER,
            width=3,
        ),
    ))

    chart_style(fig)
    fig.update_yaxes(tickformat="~s")

    return fig


def chart_technical(info):
    labels = [
        "Above 20-day average",
        "Above 50-day average",
        "Positive 1-month momentum",
        "RSI between 40 and 65",
    ]

    values = [
        int(value)
        for value in info["checks"]
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=[
            GREEN if x else RED
            for x in values
        ],
        text=[
            "PASS" if x else "NOT MET"
            for x in values
        ],
        textposition="inside",
        textfont=dict(
            color="#FFFFFF",
            size=14,
        ),
    ))

    chart_style(fig)

    fig.update_xaxes(
        visible=False,
        range=[0, 1.15],
    )

    fig.update_yaxes(
        autorange="reversed",
        showgrid=False,
        tickfont=dict(size=13),
        automargin=True,
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(
            l=25,
            r=30,
            t=30,
            b=30,
        ),
    )

    return fig


def chart_returns(df):
    returns = (
        df["Close"]
        .astype(float)
        .pct_change()
        .dropna() * 100
    )

    fig = go.Figure(go.Histogram(
        x=returns,
        nbinsx=35,
        marker_color=BLUE,
        name="Daily returns",
    ))

    fig.add_vline(
        x=0,
        line_color=AMBER,
        line_dash="dot",
    )

    chart_style(fig)

    fig.update_layout(
        showlegend=False,
        bargap=0.08,
    )

    fig.update_xaxes(
        title="Daily return (%)"
    )

    fig.update_yaxes(
        title="Number of sessions"
    )

    return fig


def chart_drawdown(df):
    close = df["Close"].astype(float)

    drawdown = (
        close / close.cummax() - 1
    ) * 100

    fig = go.Figure(go.Scatter(
        x=df.index,
        y=drawdown,
        fill="tozeroy",
        fillcolor="rgba(255,113,133,0.18)",
        line=dict(
            color=RED,
            width=3,
        ),
        name="Drawdown",
    ))

    chart_style(fig, percent=True)
    fig.update_layout(showlegend=False)

    return fig


def chart_rsi(df):
    close = df["Close"].astype(float)

    delta = close.diff()

    gain = (
        delta.clip(lower=0)
        .ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )

    loss = (
        (-delta.clip(upper=0))
        .ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    rsi = (
        100 - 100 / (1 + rs)
    ).fillna(50)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=rsi,
        name="RSI",
        line=dict(
            color=PURPLE,
            width=3,
        ),
    ))

    fig.add_hrect(
        y0=70,
        y1=100,
        fillcolor=RED,
        opacity=0.12,
        line_width=0,
    )

    fig.add_hrect(
        y0=0,
        y1=30,
        fillcolor=GREEN,
        opacity=0.12,
        line_width=0,
    )

    fig.add_hline(
        y=70,
        line_color=RED,
        line_dash="dot",
    )

    fig.add_hline(
        y=30,
        line_color=GREEN,
        line_dash="dot",
    )

    chart_style(fig)

    fig.update_yaxes(
        range=[0, 100],
        title="RSI",
    )

    return fig


def analytics_figure(index, df, zones, current, info):
    if index == 0:
        return chart_price_position(df, zones, current)

    if index == 1:
        return chart_moving_averages(df.tail(180))

    if index == 2:
        return chart_volume(df)

    if index == 3:
        return chart_technical(info)

    if index == 4:
        return chart_returns(df)

    if index == 5:
        return chart_drawdown(df)

    return chart_rsi(df.tail(180))


# ============================================================
# PREDICTIVE ANALYTICS
# ============================================================

FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "ma_gap_10",
    "ma_gap_20",
    "ma_gap_50",
    "volatility_10",
    "volatility_20",
    "rsi_14",
    "range_position",
    "volume_ratio",
]


def prediction_features(df):
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    volume = (
        df["Volume"].astype(float)
        if "Volume" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    f = pd.DataFrame(index=df.index)

    f["return_1"] = np.log(
        close / close.shift(1)
    )

    for n in (5, 10, 20):
        f[f"return_{n}"] = np.log(
            close / close.shift(n)
        )

    for n in (10, 20, 50):
        f[f"ma_gap_{n}"] = (
            close / close.rolling(n).mean() - 1
        )

    daily_return = np.log(
        close / close.shift(1)
    )

    for n in (10, 20):
        f[f"volatility_{n}"] = (
            daily_return.rolling(n).std()
        )

    delta = close.diff()

    gain = (
        delta.clip(lower=0)
        .ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )

    loss = (
        (-delta.clip(upper=0))
        .ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    f["rsi_14"] = (
        100 - 100 / (1 + rs)
    ).fillna(50) / 100

    rolling_low = low.rolling(20).min()
    rolling_high = high.rolling(20).max()

    width = (
        rolling_high - rolling_low
    ).replace(0, np.nan)

    f["range_position"] = (
        (close - rolling_low) / width
    ).fillna(0.5)

    volume_mean = (
        volume.rolling(20).mean()
        .replace(0, np.nan)
    )

    f["volume_ratio"] = (
        volume / volume_mean
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(1.0)

    return f.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def new_model():
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=20.0),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def run_prediction(df, horizon):
    if len(df) < 300:
        return None

    close = df["Close"].astype(float)
    features = prediction_features(df)

    target = np.log(
        close.shift(-horizon) / close
    ).rename("target")

    dataset = features.copy()
    dataset["target"] = target

    dataset = dataset.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    if len(dataset) < 200:
        return None

    split = int(len(dataset) * 0.8)

    # Embargo prevents training labels from
    # reaching into the holdout period.
    train_end = split - horizon

    if train_end < 120:
        return None

    train = dataset.iloc[:train_end]
    test = dataset.iloc[split:]

    model = new_model()

    model.fit(
        train[FEATURE_COLUMNS],
        train["target"],
    )

    predictions = model.predict(
        test[FEATURE_COLUMNS]
    )

    actual = test["target"].to_numpy()

    model_mae = float(
        np.mean(
            np.abs(
                np.expm1(predictions)
                - np.expm1(actual)
            )
        ) * 100
    )

    baseline_mae = float(
        np.mean(
            np.abs(np.expm1(actual))
        ) * 100
    )

    accuracy = float(
        np.mean(
            (predictions > 0) == (actual > 0)
        )
    )

    residuals = actual - predictions

    final_model = new_model()

    final_model.fit(
        dataset[FEATURE_COLUMNS],
        dataset["target"],
    )

    latest_features = features[
        FEATURE_COLUMNS
    ].dropna()

    if latest_features.empty:
        return None

    predicted_log_return = float(
        final_model.predict(
            latest_features.iloc[[-1]]
        )[0]
    )

    base_price = float(close.iloc[-1])

    predicted_price = (
        base_price
        * np.exp(predicted_log_return)
    )

    lower = base_price * np.exp(
        predicted_log_return
        + np.quantile(residuals, 0.1)
    )

    upper = base_price * np.exp(
        predicted_log_return
        + np.quantile(residuals, 0.9)
    )

    return {
        "base_price": base_price,
        "predicted_price": predicted_price,
        "predicted_return": float(
            np.expm1(predicted_log_return) * 100
        ),
        "lower": float(lower),
        "upper": float(upper),
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "accuracy": accuracy,
        "test_count": len(test),
        "test_dates": list(test.index),
        "test_actual": list(
            np.expm1(actual) * 100
        ),
        "test_predicted": list(
            np.expm1(predictions) * 100
        ),
    }


def forecast_chart(df, prediction, horizon):
    history = df["Close"].tail(90)

    last_date = pd.Timestamp(history.index[-1])

    if last_date.tzinfo is not None:
        last_date = last_date.tz_localize(None)

    future_date = (
        last_date + pd.offsets.BDay(horizon)
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history.index,
        y=history,
        name="Historical price",
        line=dict(color=BLUE, width=3),
    ))

    fig.add_trace(go.Scatter(
        x=[last_date, future_date],
        y=[
            prediction["base_price"],
            prediction["predicted_price"],
        ],
        name="Forecast",
        mode="lines+markers",
        line=dict(
            color=GREEN,
            width=3,
            dash="dash",
        ),
        marker=dict(size=9),
    ))

    fig.add_trace(go.Scatter(
        x=[future_date, future_date],
        y=[
            prediction["lower"],
            prediction["upper"],
        ],
        name="Historical-error range",
        mode="lines+markers",
        line=dict(
            color=AMBER,
            width=8,
        ),
        marker=dict(size=8),
    ))

    chart_style(fig, 440)
    fig.update_yaxes(tickprefix="$")

    return fig


def backtest_chart(prediction):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prediction["test_dates"],
        y=prediction["test_actual"],
        name="Actual return",
        line=dict(color=BLUE, width=2),
    ))

    fig.add_trace(go.Scatter(
        x=prediction["test_dates"],
        y=prediction["test_predicted"],
        name="Model prediction",
        line=dict(color=GREEN, width=2),
    ))

    fig.add_hline(
        y=0,
        line_color=AMBER,
        line_dash="dot",
    )

    chart_style(fig, 390, percent=True)

    return fig


# ============================================================
# JOURNAL
# ============================================================

def journal_db():
    conn = sqlite3.connect(
        ROOT / "market_journal.db"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY,
            created TEXT,
            ticker TEXT,
            stance TEXT,
            price REAL,
            thesis TEXT
        )
    """)

    conn.commit()
    return conn


# ============================================================
# APP HEADER
# ============================================================

st.title("📈 Market Intelligence")

st.caption(
    "Your personal market desk · "
    "Color-coded intelligence · "
    "Predictive analytics"
)

watchlist = get_watchlist()

a, b, c = st.columns(
    [2.2, 1.3, 0.8],
    vertical_alignment="bottom",
)

with a:
    search = st.text_input(
        "Search any stock ticker",
        placeholder="NVDA, TSLA, CRDO, PLTR...",
    ).strip().upper()

with b:
    selected = st.selectbox(
        "Your watchlist",
        watchlist or ["NVDA"],
    )

symbol = search or selected

with c:
    if st.button(
        "↻ Refresh",
        type="primary",
        use_container_width=True,
    ):
        prices.clear()
        latest_price.clear()
        headlines.clear()
        run_prediction.clear()
        st.rerun()

if not VALID.fullmatch(symbol):
    st.error("Enter a valid ticker.")
    st.stop()

a, b = st.columns(
    [1.5, 3],
    vertical_alignment="center",
)

with a:
    if symbol in watchlist:
        if st.button(
            f"✓ Remove {symbol}",
            use_container_width=True,
        ):
            remove_watchlist(symbol)
            st.rerun()
    else:
        if st.button(
            f"＋ Add {symbol} to Watchlist",
            type="primary",
            use_container_width=True,
        ):
            if add_watchlist(symbol):
                st.rerun()
            else:
                st.warning(
                    "Could not verify ticker."
                )

with b:
    st.caption(
        f"{len(watchlist)} saved stocks"
    )


# ============================================================
# LOAD STOCK
# ============================================================

try:
    daily = prices(symbol)

    if daily is None or daily.empty:
        st.error("No price data available.")
        st.stop()

    info = analyze(daily)

    if info is None:
        st.error("Insufficient price history.")
        st.stop()

    latest = latest_price(symbol)

    if latest is None:
        latest = get_latest_bar(symbol, daily)

    current = (
        float(latest["price"])
        if latest
        else float(daily["Close"].iloc[-1])
    )

except Exception as exc:
    st.error(f"Data error: {exc}")
    st.stop()

zones = price_levels(daily)

state = price_state(current, zones)
state_color = color_for_signal(state)

change = float(info["month_change"])

st.caption(
    f"↻ Dashboard: {format_central(now_central())}"
    " · "
    f"◷ Market: "
    f"{format_central(latest['timestamp'], daily=latest['daily']) if latest else 'Unavailable'}"
)


# ============================================================
# TABS
# ============================================================

home, discover, analytics, predictions = st.tabs([
    "⌂ Home",
    "◈ Discover",
    "▤ Analytics",
    "◉ Predictions",
])


# ============================================================
# HOME
# ============================================================

with home:

    movement = color_for_change(change)
    arrow = "▲" if change >= 0 else "▼"

    html(f"""
    <div class="mi-hero">
        <div class="mi-eyebrow">
            {escape(symbol)} · MARKET OVERVIEW
        </div>
        <div class="mi-price">
            {money(current)}
        </div>
        <div style="
            color:{movement};
            font-weight:850;
            margin-bottom:16px;
        ">
            {arrow} {change:+.2f}% · Past month
        </div>
        {pill(state,state_color)}
        <p style="margin:17px 0 0;line-height:1.75">
            {escape(price_explanation(state))}
        </p>
    </div>
    """)

    st.subheader("When to buy, hold, or sell")

    if zones:
        a, b, c = st.columns(3)

        with a:
            action_card(
                "↓ BUY ZONE",
                f"{money(zones['buy_low'])} – "
                f"{money(zones['buy_high'])}",
                "Near historical support. "
                "Research a possible entry.",
                GREEN,
            )

        with b:
            action_card(
                "◷ HOLD / WAIT",
                f"{money(zones['buy_high'])} – "
                f"{money(zones['sell_low'])}",
                "Between support and resistance. "
                "Monitor the price.",
                AMBER,
            )

        with c:
            action_card(
                "↑ SELL ZONE",
                f"{money(zones['sell_low'])} – "
                f"{money(zones['sell_high'])}",
                "Near historical resistance. "
                "Existing holders may review "
                "profit-taking.",
                RED,
            )

        panel(
            f"Current assessment: {state}",
            price_explanation(state)
            + f" Risk reference: below "
            f"{money(zones['risk'])}. "
            "These are historical price zones, "
            "not intrinsic valuations.",
        )

    st.subheader("Price chart")

    period = st.radio(
        "Period",
        ["1M", "3M", "6M", "1Y", "ALL"],
        index=2,
        horizontal=True,
        label_visibility="collapsed",
    )

    days = {
        "1M": 22,
        "3M": 66,
        "6M": 132,
        "1Y": 252,
        "ALL": len(daily),
    }[period]

    show_chart(
        stock_chart(
            daily.tail(days),
            zones,
        )
    )

    st.caption(
        "Green = potential buy zone · "
        "Red = potential sell zone · "
        "Amber dotted = risk reference"
    )


# ============================================================
# DISCOVER - COLOR CODED
# ============================================================

with discover:

    st.subheader("Your watchlist")

    saved = get_watchlist()

    panel(
        f"{len(saved)} saved stocks",
        "Your personalized stock radar. "
        "Green = positive momentum or buy zone. "
        "Amber = mixed or hold. "
        "Red = negative momentum or sell zone.",
    )

    rows = []

    for ticker in saved:
        try:
            data = prices(ticker)
            result = analyze(data)

            if result is None:
                continue

            price = float(result["price"])
            movement = float(result["month_change"])

            z = price_levels(data)

            rows.append({
                "Stock": ticker,
                "Price": price,
                "1M change": movement,
                "Price zone": price_state(price, z),
                "Trend": trend_signal(result),
            })

        except Exception:
            continue

    if rows:
        table = pd.DataFrame(rows)

        table = table.sort_values(
            "1M change",
            ascending=False,
        )

        # Color-coded cards, not plain spreadsheet rows.
        for start in range(0, len(table), 3):
            columns = st.columns(3)

            batch = table.iloc[start:start + 3]

            for column, (_, row) in zip(
                columns,
                batch.iterrows(),
            ):
                ticker = row["Stock"]
                price = row["Price"]
                movement = row["1M change"]
                zone = row["Price zone"]
                trend = row["Trend"]

                accent = color_for_change(movement)
                zone_color = color_for_signal(zone)
                trend_color = color_for_signal(trend)

                arrow = "▲" if movement >= 0 else "▼"

                with column:
                    html(f"""
                    <div class="mi-stock" style="--accent:{accent}">
                        <div class="mi-stock-top">
                            <div class="mi-stock-symbol">
                                {escape(ticker)}
                            </div>
                            {pill(trend,trend_color)}
                        </div>
                        <div class="mi-stock-price">
                            {money(price)}
                        </div>
                        <div style="
                            color:{accent};
                            font-size:18px;
                            font-weight:850;
                            margin-bottom:14px;
                        ">
                            {arrow} {movement:+.2f}% · 1M
                        </div>
                        <div class="mi-stock-bottom">
                            {pill(zone,zone_color)}
                        </div>
                    </div>
                    """)

        st.subheader("Performance comparison")

        performance = table.sort_values(
            "1M change"
        )

        fig = go.Figure(go.Bar(
            x=performance["1M change"],
            y=performance["Stock"],
            orientation="h",
            marker_color=[
                color_for_change(x)
                for x in performance["1M change"]
            ],
            text=[
                f"{x:+.2f}%"
                for x in performance["1M change"]
            ],
            textposition="outside",
            cliponaxis=False,
        ))

        chart_style(
            fig,
            max(300, len(performance) * 52 + 100),
        )

        fig.update_layout(
            showlegend=False,
            margin=dict(
                l=20,
                r=75,
                t=30,
                b=45,
            ),
        )

        fig.update_xaxes(
            title="One-month return (%)",
            zeroline=True,
            zerolinecolor="#70849A",
        )

        show_chart(fig)

        st.download_button(
            "↓ Export watchlist",
            table.to_csv(index=False),
            "watchlist.csv",
            "text/csv",
        )

    with st.expander("Manage watchlist"):
        for ticker in saved:
            a, b = st.columns(
                [4, 1],
                vertical_alignment="center",
            )

            a.write(f"**{ticker}**")

            with b:
                if st.button(
                    "Remove",
                    key=f"remove_{ticker}",
                ):
                    remove_watchlist(ticker)
                    st.rerun()

    st.subheader("Compare two stocks")

    a, b = st.columns(2)

    first = a.text_input(
        "Stock A",
        value=saved[0] if saved else symbol,
    ).strip().upper()

    second = b.text_input(
        "Stock B",
        value=saved[1] if len(saved) > 1 else "VOO",
    ).strip().upper()

    if (
        first != second
        and VALID.fullmatch(first)
        and VALID.fullmatch(second)
    ):
        try:
            x = prices(first)["Close"].rename(first)
            y = prices(second)["Close"].rename(second)

            combined = pd.concat(
                [x, y],
                axis=1,
            ).dropna().tail(126)

            if len(combined) > 1:
                normalized = (
                    combined / combined.iloc[0] * 100
                )

                fig = go.Figure()

                for ticker, color in [
                    (first, GREEN),
                    (second, BLUE),
                ]:
                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized[ticker],
                        name=ticker,
                        line=dict(
                            color=color,
                            width=3,
                        ),
                    ))

                chart_style(fig)
                show_chart(fig)

                st.caption(
                    "Both stocks start at 100 "
                    "for relative comparison."
                )

        except Exception:
            st.warning("Comparison unavailable.")


# ============================================================
# ANALYTICS - ARROW NAVIGATION
# ============================================================

with analytics:

    st.subheader(
        f"{symbol} · Analytics gallery"
    )

    panel(
        "Explore one chart at a time",
        "Use the left and right arrows to move "
        "between seven interactive visualizations. "
        "Each chart has a plain-English explanation.",
    )

    # Reset gallery when a different ticker is selected.
    if st.session_state.get(
        "analytics_ticker"
    ) != symbol:
        st.session_state.analytics_ticker = symbol
        st.session_state.analytics_index = 0

    if "analytics_index" not in st.session_state:
        st.session_state.analytics_index = 0

    total = len(ANALYTICS_SLIDES)

    index = st.session_state.analytics_index
    slide = ANALYTICS_SLIDES[index]

    # Arrow buttons genuinely update shared state.
    prev_col, counter_col, next_col = st.columns(
        [1, 2.5, 1],
        vertical_alignment="center",
    )

    with prev_col:
        if st.button(
            "← Previous",
            key="analytics_prev",
            use_container_width=True,
        ):
            st.session_state.analytics_index = (
                st.session_state.analytics_index - 1
            ) % total
            st.rerun()

    with counter_col:
        st.markdown(
            f"<div style='text-align:center;"
            f"color:{GREEN};font-weight:850'>"
            f"{index + 1} / {total}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with next_col:
        if st.button(
            "Next →",
            key="analytics_next",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.analytics_index = (
                st.session_state.analytics_index + 1
            ) % total
            st.rerun()

    html(f"""
    <div class="mi-gallery-header">
        <div class="mi-gallery-counter">
            VISUALIZATION {index + 1} OF {total}
        </div>
        <div class="mi-gallery-title">
            {escape(slide["title"])}
        </div>
        <div class="mi-gallery-description">
            {escape(slide["description"])}
        </div>
    </div>
    """)

    figure = analytics_figure(
        index,
        daily,
        zones,
        current,
        info,
    )

    if figure is not None:
        show_chart(figure)
    else:
        st.info(
            "Not enough data for this visualization."
        )

    # Contextual information underneath only ONE chart.
    panel(
        "How to read this chart",
        slide["explanation"],
    )

    if index == 0 and zones:
        a, b, c = st.columns(3)

        a.metric(
            "Recent low",
            money(zones["support"]),
        )

        b.metric(
            "Current price",
            money(current),
        )

        c.metric(
            "Recent high",
            money(zones["resistance"]),
        )

    if index == 3:
        a, b, c = st.columns(3)

        a.metric(
            "Checks passed",
            f"{info['score']} / 4",
        )

        b.metric(
            "RSI",
            f"{info['rsi']:.1f}",
        )

        c.metric(
            "Trend",
            trend_signal(info),
        )

    if index == 5:
        drawdown = (
            daily["Close"]
            / daily["Close"].cummax()
            - 1
        ) * 100

        st.metric(
            "Worst historical drawdown",
            f"{drawdown.min():.2f}%",
        )

    # Optional direct selection without stacking charts.
    with st.expander("Jump to a visualization"):
        chosen = st.selectbox(
            "Select chart",
            range(total),
            index=index,
            format_func=lambda i: (
                f"{i + 1}. "
                f"{ANALYTICS_SLIDES[i]['title']}"
            ),
        )

        if chosen != index:
            st.session_state.analytics_index = chosen
            st.rerun()

    # Research tools remain available without cluttering
    # the gallery itself.
    with st.expander("Financial news and research"):

        st.subheader("Recent financial headlines")

        try:
            articles = headlines(symbol)
        except Exception:
            articles = []

        if not articles:
            st.info("No recent headlines available.")

        for article in articles[:10]:
            title = article.get("title", "Untitled")

            with st.container(border=True):
                st.write(f"**{title}**")

                tone = article.get(
                    "tone",
                    "Neutral",
                )

                tone_color = {
                    "Positive": GREEN,
                    "Negative": RED,
                    "Neutral": AMBER,
                }.get(tone, AMBER)

                html(pill(
                    f"{tone} headline tone",
                    tone_color,
                ))

                if article.get("url"):
                    st.link_button(
                        "Read source",
                        article["url"],
                    )

        st.caption(
            "Headline tone is not a prediction "
            "of stock performance."
        )

        st.subheader("Research journal")

        with st.form("journal"):
            stance = st.selectbox(
                "Research stance",
                [
                    "Research buying",
                    "Watch",
                    "Review holding",
                    "Research selling",
                ],
            )

            note = st.text_area(
                "Your research notes"
            )

            save = st.form_submit_button(
                "Save note"
            )

        if save:
            if not note.strip():
                st.warning("Enter a note first.")
            else:
                with journal_db() as conn:
                    conn.execute("""
                        INSERT INTO journal
                        (
                            created,
                            ticker,
                            stance,
                            price,
                            thesis
                        )
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        now_central().isoformat(),
                        symbol,
                        stance,
                        current,
                        note.strip(),
                    ))

                st.success("Research note saved.")

        try:
            with journal_db() as conn:
                notes = pd.read_sql_query("""
                    SELECT
                        created,
                        ticker,
                        stance,
                        price,
                        thesis
                    FROM journal
                    ORDER BY id DESC
                    LIMIT 100
                """, conn)

            if not notes.empty:
                st.dataframe(
                    notes,
                    hide_index=True,
                    use_container_width=True,
                )

                st.download_button(
                    "↓ Export journal",
                    notes.to_csv(index=False),
                    "research_journal.csv",
                    "text/csv",
                )

        except Exception:
            pass


# ============================================================
# PREDICTIONS
# ============================================================

with predictions:

    st.subheader(
        f"{symbol} · Predictive intelligence"
    )

    panel(
        "Machine learning forecast",
        "Historical momentum, RSI, volatility, "
        "moving averages, and volume are used "
        "to estimate future returns. "
        "The model is tested on later historical data.",
    )

    horizon = st.radio(
        "Forecast horizon",
        [7, 14, 30],
        index=1,
        horizontal=True,
        format_func=lambda x: f"{x} trading days",
    )

    with st.spinner("Training prediction model..."):
        try:
            prediction = run_prediction(
                daily,
                horizon,
            )
        except Exception as exc:
            prediction = None
            st.warning(
                f"Prediction unavailable: {exc}"
            )

    if prediction is None:
        st.info(
            "At least 300 daily observations "
            "are required for predictions."
        )

    else:
        predicted_change = prediction[
            "predicted_return"
        ]

        direction_color = color_for_change(
            predicted_change
        )

        if predicted_change > 2:
            direction = "UPWARD BIAS"
        elif predicted_change < -2:
            direction = "DOWNWARD BIAS"
        else:
            direction = "SIDEWAYS / UNCERTAIN"
            direction_color = AMBER

        html(f"""
        <div class="mi-hero">
            <div class="mi-eyebrow">
                {escape(symbol)} ·
                {horizon}-TRADING-DAY FORECAST
            </div>
            <div class="mi-price">
                {money(prediction["predicted_price"])}
            </div>
            {pill(direction,direction_color)}
            <p style="margin-top:17px">
                Estimated change:
                <b style="color:{direction_color}">
                    {predicted_change:+.2f}%
                </b>
            </p>
        </div>
        """)

        a, b, c = st.columns(3)

        with a:
            action_card(
                "CURRENT DAILY CLOSE",
                money(prediction["base_price"]),
                "Forecast starting price.",
                BLUE,
            )

        with b:
            action_card(
                "PREDICTED PRICE",
                money(prediction["predicted_price"]),
                f"Estimated after {horizon} "
                "trading days.",
                direction_color,
            )

        with c:
            action_card(
                "HISTORICAL-ERROR RANGE",
                f"{money(prediction['lower'])} – "
                f"{money(prediction['upper'])}",
                "Based on past prediction errors. "
                "Not a guaranteed future range.",
                AMBER,
            )

        st.subheader(
            "Historical price vs. forecast"
        )

        show_chart(
            forecast_chart(
                daily,
                prediction,
                horizon,
            )
        )

        st.caption(
            "The amber range reflects past holdout "
            "errors, not a calibrated probability "
            "interval. The target date uses business "
            "days and may differ from exchange sessions."
        )

        st.subheader(
            "Historical model performance"
        )

        a, b, c = st.columns(3)

        a.metric(
            "Direction accuracy",
            f"{prediction['accuracy']:.1%}",
        )

        b.metric(
            "Model error",
            f"{prediction['model_mae']:.2f} pp",
        )

        c.metric(
            "Baseline error",
            f"{prediction['baseline_mae']:.2f} pp",
        )

        if (
            prediction["model_mae"]
            < prediction["baseline_mae"]
        ):
            st.success(
                "Model beat the no-change baseline "
                "on this historical holdout."
            )
        else:
            st.warning(
                "Model did not beat the no-change "
                "baseline. Treat the forecast "
                "with additional caution."
            )

        with st.expander(
            "View historical backtesting chart"
        ):
            show_chart(
                backtest_chart(prediction)
            )

            st.caption(
                f"Evaluated on "
                f"{prediction['test_count']} "
                "later observations. "
                "Overlapping forecast horizons "
                "create correlated test observations."
            )

        st.subheader(
            "Investment scenario"
        )

        investment = st.number_input(
            "Hypothetical investment ($)",
            min_value=0.0,
            value=500.0,
            step=50.0,
        )

        future_value = investment * (
            1 + predicted_change / 100
        )

        a, b = st.columns(2)

        a.metric(
            "Value under model forecast",
            f"${future_value:,.2f}",
        )

        b.metric(
            "Hypothetical gain / loss",
            f"${future_value-investment:+,.2f}",
        )

        st.caption(
            "Illustrative only. Predictions are "
            "uncertain and exclude taxes, "
            "fees, and dividends."
        )


# ============================================================
# FOOTER
# ============================================================

with st.expander("Automated email alerts"):
    st.write(
        "Your existing GitHub Actions email "
        "alerts remain separate and unchanged."
    )

st.divider()

st.caption(
    "Market Intelligence · "
    "Historical technical levels · "
    "Predictive estimates are not guarantees · "
    "No automatic trading"
)

