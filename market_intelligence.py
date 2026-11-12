
import html
import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_core import (
    analyze,
    format_central,
    get_latest_bar,
    get_news,
    get_prices,
    now_central,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent

CONFIG = json.loads(
    (ROOT / "alert_config.json").read_text()
)

WATCHLIST = CONFIG["watchlist"]

# ============================================================
# COLOR PALETTE
# ============================================================

GOLD = "#DCC18C"
GREEN = "#24D18B"
RED = "#FF5267"
AMBER = "#F3BE62"
BLUE = "#68A8FF"
PURPLE = "#B39AFF"

BG = "#0B1017"
PANEL = "#171F2B"
MUTED = "#B4C0CF"


def signal_color(signal):
    return {
        "RESEARCH ENTRY": GREEN,
        "WATCH": AMBER,
        "REVIEW RISK": RED,
    }.get(signal, AMBER)


def signal_class(signal):
    return {
        "RESEARCH ENTRY": "bullish",
        "WATCH": "neutral",
        "REVIEW RISK": "bearish",
    }.get(signal, "neutral")


def change_color(value):
    if value > 0:
        return GREEN
    if value < 0:
        return RED
    return AMBER


# ============================================================
# PREMIUM DARK TRADING TERMINAL CSS
# ============================================================

st.markdown("""
<style>

html, body, .stApp,
[data-testid="stAppViewContainer"] {
    background: #0B1017;
    color: #F5F1E9;
}

.block-container {
    max-width: 1150px;
    padding: 1rem 1rem 4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    letter-spacing: -.035em;
}

/* HERO */

.hero {
    padding: 28px;
    border: 1px solid #364456;
    border-radius: 22px;
    background:
        radial-gradient(
            circle at 90% 10%,
            rgba(36,209,139,.13),
            transparent 40%
        ),
        linear-gradient(
            120deg, #273341, #141B26
        );
    margin-bottom: 18px;
}

.hero h1 {
    font-size: clamp(1.9rem, 5vw, 3rem);
    margin: .4rem 0;
    color: #F5F1E9;
}

.hero p {
    color: #B4C0CF;
    margin: 0;
}

.eyebrow {
    color: #DCC18C;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .15em;
}

/* TIMESTAMP CARDS */

.timebox {
    background: #171F2B;
    border: 1px solid #344050;
    border-radius: 15px;
    padding: 15px;
    min-height: 106px;
}

.timebox .small {
    color: #B4C0CF;
    font-size: .75rem;
    margin-bottom: 8px;
}

.timebox .value {
    font-weight: 700;
    font-size: clamp(.82rem, 2vw, 1.05rem);
    overflow-wrap: anywhere;
}

/* METRICS */

[data-testid="stMetric"] {
    background: #171F2B;
    border: 1px solid #344050;
    border-radius: 15px;
    padding: 16px;
}

[data-testid="stMetricValue"] {
    font-size: clamp(1.2rem, 3vw, 1.85rem);
}

[data-testid="stMetricDelta"] {
    font-weight: 750;
}

/* BUTTONS */

.stButton button[kind="primary"] {
    background: #DCC18C;
    color: #0B1017;
    border-color: #DCC18C;
    font-weight: 750;
    min-height: 48px;
    border-radius: 12px;
}

.stButton button,
.stDownloadButton button {
    border-radius: 12px;
    min-height: 44px;
}

/* TABS */

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    overflow-x: auto;
    scrollbar-width: thin;
}

[data-testid="stTabs"] button {
    white-space: nowrap;
}

/* BULLISH / BEARISH / NEUTRAL */

.bullish {
    color: #24D18B;
    background: rgba(36, 209, 139, .10);
    border: 1px solid rgba(36, 209, 139, .35);
}

.bearish {
    color: #FF5267;
    background: rgba(255, 82, 103, .10);
    border: 1px solid rgba(255, 82, 103, .35);
}

.neutral {
    color: #F3BE62;
    background: rgba(243, 190, 98, .10);
    border: 1px solid rgba(243, 190, 98, .35);
}

.indicator-pill {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: .82rem;
    font-weight: 800;
    margin: 4px 0 12px;
}

/* SIGNAL CARD */

.signal {
    border-left-width: 5px;
    border-left-style: solid;
    border-radius: 15px;
    padding: 20px;
    margin: 14px 0;
}

/* INDICATOR ROWS */

.indicator-row {
    background: #171F2B;
    border: 1px solid #344050;
    border-radius: 12px;
    padding: 12px 15px;
    margin: 7px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}

.indicator-row .label {
    color: #E1E7EF;
    font-size: .9rem;
}

.indicator-row .status {
    font-size: .82rem;
    font-weight: 800;
}

@media (max-width: 640px) {

    .block-container {
        padding: .6rem .7rem 4rem;
    }

    .hero {
        padding: 18px 15px;
    }

    .hero h1 {
        font-size: 1.8rem;
    }

    .hero p {
        font-size: .87rem;
    }

    .timebox {
        padding: 10px;
        min-height: 104px;
    }

    .timebox .value {
        font-size: .81rem;
    }

    [data-testid="stMetric"] {
        padding: 11px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
    }

    .stButton button {
        width: 100%;
    }

    .indicator-row .label {
        font-size: .8rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED MARKET DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def prices(symbol):
    return get_prices(symbol)


@st.cache_data(ttl=300, show_spinner=False)
def observation(symbol):
    return get_latest_bar(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def news(symbol):
    return get_news(symbol)


# ============================================================
# COLORFUL PRICE CHART
# ============================================================

def chart(df, info, days):
    part = df.tail(days)

    direction_color = change_color(
        info["month_change"]
    )

    fig = go.Figure()

    # Main price line changes color with momentum
    fig.add_trace(go.Scatter(
        x=part.index,
        y=part["Close"],
        name="Stock price",
        mode="lines",
        line=dict(
            color=direction_color,
            width=3,
        ),
        fill="tozeroy",
        fillcolor=(
            "rgba(36,209,139,0.07)"
            if info["month_change"] > 0
            else "rgba(255,82,103,0.07)"
        ),
    ))

    # 20-day moving average
    fig.add_trace(go.Scatter(
        x=part.index,
        y=info["ma20"].reindex(part.index),
        name="20-day average",
        line=dict(
            color=BLUE,
            width=2,
            dash="dot",
        ),
    ))

    # 50-day moving average
    fig.add_trace(go.Scatter(
        x=part.index,
        y=info["ma50"].reindex(part.index),
        name="50-day average",
        line=dict(
            color=PURPLE,
            width=2,
            dash="dash",
        ),
    ))

    fig.update_layout(
        height=360,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=MUTED),
        margin=dict(
            l=5,
            r=5,
            t=30,
            b=5,
        ),
        xaxis=dict(
            showgrid=False,
        ),
        yaxis=dict(
            gridcolor="#303A48",
            tickprefix="$",
            rangemode="normal",
        ),
        legend=dict(
            orientation="h",
            y=1.17,
        ),
        hovermode="x unified",
    )

    return fig


# ============================================================
# RESEARCH JOURNAL
# ============================================================

def journal_db():
    connection = sqlite3.connect(
        ROOT / "market_journal.db"
    )

    connection.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY,
            created TEXT,
            ticker TEXT,
            stance TEXT,
            price REAL,
            thesis TEXT
        )
    """)

    connection.commit()
    return connection


# ============================================================
# MAIN PAGE HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">
        THE PERSONAL MARKET DESK
    </div>
    <h1>Market Intelligence</h1>
    <p>
        Live-inspired market research,
        colorful technical indicators,
        and the stories moving your watchlist.
    </p>
</div>
""", unsafe_allow_html=True)

stock_col, range_col = st.columns([2, 1])

with stock_col:
    symbol = st.selectbox(
        "Watchlist stock",
        WATCHLIST,
    )

with range_col:
    period = st.selectbox(
        "Chart range",
        ["1M", "3M", "6M", "1Y", "2Y"],
        index=1,
    )


# ============================================================
# DASHBOARD
# ============================================================

@st.fragment(run_every=300)
def dashboard():

    if st.button(
        "↻ Refresh Market Data",
        type="primary",
        use_container_width=True,
    ):
        prices.clear()
        observation.clear()
        news.clear()

    try:
        with st.spinner("Checking market data..."):
            df = prices(symbol)
            info = analyze(df)

            if info is None:
                st.warning(
                    "Not enough price history "
                    "for this ticker."
                )
                return

            bar = observation(symbol)

            if bar is None:
                bar = get_latest_bar(symbol, df)

            if bar is None:
                st.warning(
                    "Price observation unavailable."
                )
                return

            checked = now_central()

    except Exception as exc:
        st.error(
            f"Market data check failed: {exc}"
        )
        return

    # ========================================================
    # TIMESTAMPS
    # ========================================================

    a, b = st.columns(2)

    with a:
        st.markdown(
            '<div class="timebox">'
            '<div class="small">'
            '↻ Dashboard updated'
            '</div>'
            '<div class="value">'
            + html.escape(
                format_central(checked)
            )
            + '</div></div>',
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            '<div class="timebox">'
            '<div class="small">'
            '◈ Latest market data'
            '</div>'
            '<div class="value">'
            + html.escape(
                format_central(
                    bar["timestamp"],
                    daily=bar["daily"],
                )
            )
            + '</div></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"Price observation: {bar['kind']}. "
        "Automatically checks every five minutes "
        "while the dashboard is open. "
        "Quotes may be delayed."
    )

    # ========================================================
    # TABS
    # ========================================================

    overview, radar, compare, lab, journal = st.tabs([
        "Overview",
        "Radar",
        "Compare",
        "What-if",
        "Journal",
    ])

    # ========================================================
    # OVERVIEW
    # ========================================================

    with overview:

        st.subheader(
            f"{symbol} · Market Overview"
        )

        # Positive / negative momentum pill
        month_change = info["month_change"]

        if month_change > 0:
            momentum_label = "▲ BULLISH MOMENTUM"
            momentum_class = "bullish"
        elif month_change < 0:
            momentum_label = "▼ BEARISH MOMENTUM"
            momentum_class = "bearish"
        else:
            momentum_label = "● FLAT MOMENTUM"
            momentum_class = "neutral"

        st.markdown(
            f'<div class="indicator-pill '
            f'{momentum_class}">'
            f'{momentum_label}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # MAIN METRICS
        # ----------------------------------------------------

        a, b = st.columns(2)

        a.metric(
            "Latest observed price",
            f"${bar['price']:,.2f}",
        )

        b.metric(
            "~1-month change",
            f"{month_change:+.2f}%",
            delta=f"{month_change:+.2f}%",
            delta_color="normal",
        )

        a, b = st.columns(2)

        a.metric(
            "Positive indicators",
            f"{info['score']} / 4",
        )

        b.metric(
            "Annualized volatility",
            f"{info['volatility']:.1f}%",
        )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        days = {
            "1M": 22,
            "3M": 66,
            "6M": 132,
            "1Y": 252,
            "2Y": 504,
        }[period]

        st.plotly_chart(
            chart(df, info, days),
            use_container_width=True,
            config={
                "displayModeBar": False
            },
        )

        # ----------------------------------------------------
        # TECHNICAL OUTLOOK
        # ----------------------------------------------------

        explanation = {
            "RESEARCH ENTRY":
                "Several technical indicators are "
                "positive. Review fundamentals and "
                "valuation before considering an entry.",

            "WATCH":
                "Technical indicators are mixed. "
                "Continue monitoring price action.",

            "REVIEW RISK":
                "Recent technical indicators are "
                "weaker. Reassess your thesis "
                "and risk exposure.",
        }[info["signal"]]

        color = signal_color(
            info["signal"]
        )

        st.markdown(
            f"""
            <div class="signal" style="
                border-left-color: {color};
                background: linear-gradient(
                    90deg,
                    {color}18,
                    #171F2B 75%
                );
            ">
                <div class="eyebrow">
                    TECHNICAL RESEARCH OUTLOOK
                </div>

                <h3 style="color: {color};">
                    {html.escape(info["signal"])}
                </h3>

                <p>
                    {html.escape(explanation)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # INDICATOR BREAKDOWN
        # ----------------------------------------------------

        st.subheader(
            "Technical indicators"
        )

        labels = [
            "Price above 20-day average",
            "Price above 50-day average",
            "Positive monthly momentum",
            "RSI in preferred range (40–65)",
        ]

        for label, passed in zip(
            labels,
            info["checks"],
        ):
            color = GREEN if passed else RED

            status = (
                "✓ POSITIVE"
                if passed
                else "✕ NOT MET"
            )

            st.markdown(
                f"""
                <div class="indicator-row">
                    <span class="label">
                        {html.escape(label)}
                    </span>
                    <span class="status"
                          style="color:{color};">
                        {status}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # NEWS
        # ----------------------------------------------------

        st.subheader(
            "Latest financial headlines"
        )

        try:
            articles = news(symbol)

            if not articles:
                st.info(
                    "No headlines available."
                )

            for article in articles[:6]:

                tone = article["tone"]

                if tone == "Positive":
                    tone_color = GREEN
                elif tone == "Negative":
                    tone_color = RED
                else:
                    tone_color = AMBER

                with st.container(border=True):

                    st.markdown(
                        "**"
                        + article["title"]
                        + "**"
                    )

                    st.markdown(
                        f'<span style="color:'
                        f'{tone_color};font-weight:700;">'
                        f'● {html.escape(tone)} '
                        f'headline tone</span>',
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        article["published"]
                    )

                    st.link_button(
                        "Read article",
                        article["url"],
                    )

        except Exception:
            st.info(
                "News feed temporarily unavailable."
            )

    # ========================================================
    # OPPORTUNITY RADAR
    # ========================================================

    with radar:

        st.subheader(
            "Opportunity radar"
        )

        st.caption(
            "Compare technical conditions "
            "across your watchlist."
        )

        rows = []

        for ticker in WATCHLIST:
            try:
                result = analyze(
                    prices(ticker)
                )

                if result:
                    rows.append({
                        "Ticker": ticker,
                        "Close ($)": round(
                            result["price"], 2
                        ),
                        "Positive signs":
                            result["score"],
                        "Month %": round(
                            result["month_change"], 2
                        ),
                        "Outlook":
                            result["signal"],
                    })

            except Exception:
                pass

        if rows:

            table = pd.DataFrame(
                rows
            ).sort_values(
                ["Positive signs", "Month %"],
                ascending=False,
            )

            def radar_style(value):
                if value == "RESEARCH ENTRY":
                    return (
                        "color: #24D18B; "
                        "font-weight: bold"
                    )

                if value == "REVIEW RISK":
                    return (
                        "color: #FF5267; "
                        "font-weight: bold"
                    )

                if value == "WATCH":
                    return (
                        "color: #F3BE62; "
                        "font-weight: bold"
                    )

                return ""

            def change_style(value):
                if value > 0:
                    return (
                        "color: #24D18B; "
                        "font-weight: bold"
                    )

                if value < 0:
                    return (
                        "color: #FF5267; "
                        "font-weight: bold"
                    )

                return "color: #F3BE62"

            styled = (
                table.style
                .map(
                    radar_style,
                    subset=["Outlook"],
                )
                .map(
                    change_style,
                    subset=["Month %"],
                )
            )

            st.dataframe(
                styled,
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Export radar",
                table.to_csv(
                    index=False
                ),
                "market_radar.csv",
                "text/csv",
            )

        else:
            st.info(
                "Radar data unavailable."
            )

    # ========================================================
    # STOCK COMPARISON
    # ========================================================

    with compare:

        st.subheader(
            "Compare stocks"
        )

        a, b = st.columns(2)

        with a:
            first = st.selectbox(
                "First stock",
                WATCHLIST,
                key="first",
            )

        with b:
            second = st.selectbox(
                "Second stock",
                WATCHLIST,
                index=min(
                    1,
                    len(WATCHLIST) - 1,
                ),
                key="second",
            )

        if first == second:

            st.info(
                "Choose two different stocks."
            )

        else:

            try:
                first_prices = prices(first)[
                    "Close"
                ].rename(first)

                second_prices = prices(second)[
                    "Close"
                ].rename(second)

                aligned = pd.concat(
                    [
                        first_prices,
                        second_prices,
                    ],
                    axis=1,
                ).dropna().tail(126)

                if len(aligned) > 1:

                    normalized = (
                        aligned.div(
                            aligned.iloc[0]
                        ) * 100
                    )

                    fig = go.Figure()

                    for ticker, color in [
                        (first, GREEN),
                        (second, BLUE),
                    ]:

                        fig.add_trace(
                            go.Scatter(
                                x=normalized.index,
                                y=normalized[ticker],
                                name=ticker,
                                line=dict(
                                    color=color,
                                    width=3,
                                ),
                            )
                        )

                    fig.add_hline(
                        y=100,
                        line_dash="dot",
                        line_color=AMBER,
                        opacity=.5,
                    )

                    fig.update_layout(
                        height=340,
                        paper_bgcolor=PANEL,
                        plot_bgcolor=PANEL,
                        font=dict(
                            color=MUTED
                        ),
                        margin=dict(
                            l=5,
                            r=5,
                            t=20,
                            b=5,
                        ),
                        xaxis=dict(
                            showgrid=False
                        ),
                        yaxis=dict(
                            gridcolor="#303A48"
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={
                            "displayModeBar": False
                        },
                    )

                    st.caption(
                        "Both stocks start at an "
                        "indexed value of 100."
                    )

            except Exception:
                st.info(
                    "Comparison data unavailable."
                )

    # ========================================================
    # INVESTMENT SIMULATOR
    # ========================================================

    with lab:

        st.subheader(
            "Investment what-if lab"
        )

        amount = st.number_input(
            "Hypothetical investment ($)",
            min_value=0.0,
            value=500.0,
            step=50.0,
        )

        change = st.slider(
            "Hypothetical price move (%)",
            -80,
            150,
            10,
        )

        value = amount * (
            1 + change / 100
        )

        gain = value - amount

        a, b = st.columns(2)

        a.metric(
            "Potential value",
            f"${value:,.2f}",
        )

        b.metric(
            "Gain / loss",
            f"${gain:+,.2f}",
            delta=f"{change:+d}%",
            delta_color="normal",
        )

        if gain > 0:
            st.success(
                f"Hypothetical gain: "
                f"${gain:,.2f}"
            )

        elif gain < 0:
            st.error(
                f"Hypothetical loss: "
                f"${abs(gain):,.2f}"
            )

        else:
            st.info(
                "No hypothetical change."
            )

        st.caption(
            "Illustrative arithmetic only. "
            "Excludes fees, taxes and dividends."
        )

    # ========================================================
    # RESEARCH JOURNAL
    # ========================================================

    with journal:

        st.subheader(
            "Research journal"
        )

        st.caption(
            "Record your investment ideas. "
            "Export notes regularly because "
            "Streamlit's local storage "
            "may reset."
        )

        with st.form("journal_form"):

            stance = st.selectbox(
                "Research idea",
                [
                    "Research buying",
                    "Watch",
                    "Review holding",
                    "Research selling",
                ],
            )

            thesis = st.text_area(
                "Your investment thesis"
            )

            submitted = (
                st.form_submit_button(
                    "Save note"
                )
            )

        if submitted:

            if thesis.strip():

                try:
                    with journal_db() as conn:
                        conn.execute(
                            """
                            INSERT INTO journal
                            (
                                created,
                                ticker,
                                stance,
                                price,
                                thesis
                            )
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                now_central().isoformat(),
                                symbol,
                                stance,
                                bar["price"],
                                thesis.strip(),
                            ),
                        )

                    st.success(
                        "Research note saved."
                    )

                except Exception as exc:
                    st.error(
                        f"Could not save: {exc}"
                    )

            else:
                st.warning(
                    "Write a note first."
                )

        try:
            with journal_db() as conn:

                notes = pd.read_sql_query(
                    """
                    SELECT
                        created,
                        ticker,
                        stance,
                        price,
                        thesis
                    FROM journal
                    ORDER BY id DESC
                    LIMIT 100
                    """,
                    conn,
                )

            if not notes.empty:

                st.dataframe(
                    notes,
                    hide_index=True,
                    use_container_width=True,
                )

                st.download_button(
                    "Export journal",
                    notes.to_csv(
                        index=False
                    ),
                    "research_journal.csv",
                    "text/csv",
                )

        except Exception:
            st.info(
                "Journal unavailable."
            )


# ============================================================
# RUN DASHBOARD
# ============================================================

dashboard()


# ============================================================
# EMAIL ALERT INFORMATION
# ============================================================

with st.expander(
    "Automated email alerts"
):

    st.write(
        "Your GitHub Actions automation "
        "monitors the watchlist independently "
        "and sends email notifications for "
        "configured price thresholds, "
        "technical signal changes, and "
        "matching financial headlines."
    )

    st.caption(
        "The dashboard does not need "
        "to remain open for GitHub Actions "
        "to run its scheduled checks."
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Market Intelligence · "
    "Educational research only · "
    "No automatic trades or guaranteed signals."
)
