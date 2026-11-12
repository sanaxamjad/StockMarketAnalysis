
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
# PAGE SETTINGS
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

WATCHLIST = CONFIG.get(
    "watchlist",
    ["NVDA", "CRDO", "VOO", "AAPL", "MSFT", "AMD"],
)

# ============================================================
# COLOR SYSTEM
# ============================================================

BG = "#0B0F14"
CARD = "#151C24"
CARD_LIGHT = "#1B2530"
BORDER = "#28333F"

WHITE = "#F5F7FA"
MUTED = "#95A2B2"

GREEN = "#2DDCA0"
RED = "#FF596B"
AMBER = "#F4BC67"
LIME = "#C5FA5F"
BLUE = "#75AFFF"


def movement_color(value):
    if value > 0:
        return GREEN
    if value < 0:
        return RED
    return MUTED


def outlook_color(signal):
    return {
        "RESEARCH ENTRY": GREEN,
        "WATCH": AMBER,
        "REVIEW RISK": RED,
    }.get(signal, MUTED)


def outlook_label(signal):
    return {
        "RESEARCH ENTRY": "Positive trend",
        "WATCH": "Mixed signals",
        "REVIEW RISK": "Weak trend",
    }.get(signal, "Unavailable")


def safe(text):
    return html.escape(str(text))


def metric_card(label, value, color=WHITE, detail=""):
    return f"""
    <div class="mini-card">
        <div class="mini-label">{safe(label)}</div>
        <div class="mini-value"
             style="color:{color}">
            {safe(value)}
        </div>
        <div class="mini-detail">{safe(detail)}</div>
    </div>
    """


# ============================================================
# ROBINHOOD-INSPIRED STYLING
# ============================================================

st.markdown("""
<style>

html, body, .stApp,
[data-testid="stAppViewContainer"] {
    background: #0B0F14 !important;
    color: #F5F7FA;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1080px;
    padding: 1.2rem 1.1rem 4rem;
}

h1, h2, h3 {
    letter-spacing: -0.035em;
}

h2 {
    font-size: 1.45rem !important;
}

h3 {
    font-size: 1.12rem !important;
}

/* BRAND HEADER */

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.brand-logo {
    width: 40px;
    height: 40px;
    border-radius: 13px;
    background: #C5FA5F;
    color: #0B0F14;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
    font-weight: 900;
}

.brand-title {
    color: #F5F7FA;
    font-size: 1.13rem;
    font-weight: 800;
    letter-spacing: -.035em;
}

.brand-subtitle {
    color: #95A2B2;
    font-size: .69rem;
    letter-spacing: .07em;
}

/* PRICE HERO */

.price-section {
    padding: 12px 0 7px;
}

.stock-symbol {
    color: #95A2B2;
    font-size: .92rem;
    font-weight: 600;
}

.stock-price {
    color: #F5F7FA;
    font-size: clamp(2.5rem, 6vw, 4.2rem);
    font-weight: 750;
    line-height: 1.15;
    letter-spacing: -.055em;
    margin: 6px 0;
}

.stock-change {
    font-size: 1.02rem;
    font-weight: 750;
    margin-top: 7px;
}

.stock-period {
    color: #95A2B2;
    font-size: .86rem;
    margin-left: 8px;
}

/* SIGNAL LABEL */

.signal-pill {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 750;
    margin-top: 12px;
}

/* QUICK STAT CARDS */

.mini-card {
    background: #151C24;
    border: 1px solid #28333F;
    border-radius: 16px;
    padding: 17px;
    min-height: 110px;
    margin-bottom: 6px;
}

.mini-label {
    color: #95A2B2;
    font-size: .78rem;
    margin-bottom: 10px;
}

.mini-value {
    font-size: 1.16rem;
    font-weight: 800;
    letter-spacing: -.025em;
}

.mini-detail {
    color: #8493A5;
    font-size: .72rem;
    margin-top: 7px;
}

/* SECTION HEADERS */

.section-heading {
    color: #F5F7FA;
    font-size: 1.12rem;
    font-weight: 750;
    margin: 24px 0 14px;
}

/* INDICATOR ROW */

.indicator {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 2px;
    border-bottom: 1px solid #28333F;
    gap: 14px;
}

.indicator-name {
    color: #DCE4ED;
    font-size: .89rem;
}

.indicator-status {
    font-size: .82rem;
    font-weight: 750;
    white-space: nowrap;
}

/* TIMESTAMPS */

.data-footer {
    border-top: 1px solid #28333F;
    padding-top: 13px;
    margin-top: 18px;
    color: #95A2B2;
    font-size: .76rem;
    line-height: 1.8;
}

/* STREAMLIT CONTROLS */

.stButton button[kind="primary"] {
    background: #C5FA5F !important;
    color: #0B0F14 !important;
    border: none !important;
    font-weight: 800;
}

.stButton button {
    border-radius: 12px !important;
    min-height: 40px;
}

.stButton button[kind="secondary"] {
    background: #151C24 !important;
    border: 1px solid #28333F !important;
    color: #E9F0F7 !important;
}

[data-testid="stSelectbox"] > div > div {
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: #151C24;
    border: 1px solid #28333F;
    border-radius: 15px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    font-size: 1.35rem;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 22px;
    border-bottom: 1px solid #28333F;
    overflow-x: auto;
    scrollbar-width: none;
}

[data-testid="stTabs"] button {
    white-space: nowrap;
    font-weight: 650;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: #C5FA5F !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: #C5FA5F !important;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: #28333F;
}

/* MOBILE */

@media (max-width: 640px) {

    .block-container {
        padding: .7rem .8rem 4rem;
    }

    .stock-price {
        font-size: 2.8rem;
    }

    .brand-title {
        font-size: 1rem;
    }

    .mini-card {
        padding: 12px;
        min-height: 102px;
    }

    .mini-value {
        font-size: 1rem;
    }

    .mini-label {
        font-size: .72rem;
    }

    .mini-detail {
        font-size: .68rem;
    }

    .indicator-name {
        font-size: .79rem;
    }

    .indicator-status {
        font-size: .73rem;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 12px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def cached_prices(symbol, period="2y", interval="1d"):
    return get_prices(
        symbol,
        period=period,
        interval=interval,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_observation(symbol):
    return get_latest_bar(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(symbol):
    return get_news(symbol)


# ============================================================
# CHARTS
# ============================================================

def price_chart(df, change):
    color = movement_color(change)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Price",
            line=dict(
                color=color,
                width=3,
            ),
            hovertemplate=(
                "%{x|%b %d, %Y}"
                "<br>$%{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=350,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=MUTED),
        margin=dict(
            l=0, r=8, t=10, b=5
        ),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#202A35",
            zeroline=False,
            tickprefix="$",
            side="right",
        ),
    )

    return fig


def comparison_chart(data):
    fig = go.Figure()

    colors = [GREEN, BLUE, AMBER, RED]

    for index, ticker in enumerate(data.columns):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[ticker],
                name=ticker,
                mode="lines",
                line=dict(
                    color=colors[index % len(colors)],
                    width=2.7,
                ),
            )
        )

    fig.add_hline(
        y=100,
        line_color="#64748B",
        line_dash="dot",
        opacity=.55,
    )

    fig.update_layout(
        height=340,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=MUTED),
        margin=dict(
            l=0, r=0, t=25, b=5
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.16,
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            gridcolor="#202A35"
        ),
    )

    return fig


# ============================================================
# JOURNAL
# ============================================================

def journal_connection():
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
# HEADER
# ============================================================

st.markdown("""
<div class="brand">
    <div class="brand-logo">↗</div>
    <div>
        <div class="brand-title">
            Market Intelligence
        </div>
        <div class="brand-subtitle">
            YOUR PERSONAL MARKET DESK
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STOCK SELECTOR
# ============================================================

selector_col, refresh_col = st.columns(
    [4, 1],
    vertical_alignment="bottom",
)

with selector_col:
    selected_stock = st.selectbox(
        "Choose a stock",
        WATCHLIST,
        key="selected_stock",
    )

with refresh_col:
    refresh_requested = st.button(
        "↻ Refresh",
        use_container_width=True,
        help="Fetch fresh market data",
    )

if refresh_requested:
    cached_prices.clear()
    cached_observation.clear()
    cached_news.clear()


# ============================================================
# AUTOMATIC UPDATE
# ============================================================

@st.fragment(run_every=300)
def market_dashboard(symbol):

    try:
        with st.spinner("Loading market data..."):

            daily = cached_prices(symbol)
            analysis = analyze(daily)

            if analysis is None:
                st.warning(
                    "Not enough price history "
                    "for this stock."
                )
                return

            latest = cached_observation(symbol)

            if latest is None:
                latest = get_latest_bar(
                    symbol,
                    daily,
                )

            if latest is None:
                st.warning(
                    "Latest market price unavailable."
                )
                return

            checked_at = now_central()

    except Exception as exc:
        st.error(
            f"Market data unavailable: {exc}"
        )
        return

    # ========================================================
    # THREE SIMPLE NAVIGATION TABS
    # ========================================================

    home, discover, research = st.tabs([
        "Home",
        "Discover",
        "Research",
    ])

    # ========================================================
    # HOME
    # ========================================================

    with home:

        price = latest["price"]
        month_change = analysis["month_change"]

        color = movement_color(month_change)

        if month_change > 0:
            arrow = "↗"
        elif month_change < 0:
            arrow = "↘"
        else:
            arrow = "→"

        # ----------------------------------------------------
        # LARGE PRICE HEADER
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="price-section">
                <div class="stock-symbol">
                    {safe(symbol)}
                </div>

                <div class="stock-price">
                    ${price:,.2f}
                </div>

                <div class="stock-change"
                     style="color:{color}">
                    {arrow} {month_change:+.2f}%
                    <span class="stock-period">
                        Past month
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # SIMPLE TREND LABEL
        # ----------------------------------------------------

        outlook = analysis["signal"]
        outlook_text = outlook_label(outlook)
        outlook_tint = outlook_color(outlook)

        st.markdown(
            f"""
            <div class="signal-pill" style="
                color:{outlook_tint};
                background:{outlook_tint}18;
                border:1px solid {outlook_tint}55;
            ">
                ● {safe(outlook_text)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # CHART RANGE
        # ----------------------------------------------------

        chart_range = st.radio(
            "Time range",
            ["1M", "3M", "6M", "1Y", "ALL"],
            index=1,
            horizontal=True,
            label_visibility="collapsed",
            key="home_chart_range",
        )

        periods = {
            "1M": 22,
            "3M": 66,
            "6M": 132,
            "1Y": 252,
            "ALL": len(daily),
        }

        days = periods[chart_range]

        chart_data = daily.tail(days)

        # Chart color reflects selected range,
        # not the fixed one-month indicator.
        if len(chart_data) >= 2:
            range_change = (
                float(chart_data["Close"].iloc[-1])
                / float(chart_data["Close"].iloc[0])
                - 1
            ) * 100
        else:
            range_change = 0

        st.plotly_chart(
            price_chart(
                chart_data,
                range_change,
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            },
        )

        st.caption(
            f"Chart period return: "
            f"{range_change:+.2f}%. "
            "The headline percentage shows "
            "approximately one month."
        )

        # ----------------------------------------------------
        # AT A GLANCE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-heading">'
            'At a glance'
            '</div>',
            unsafe_allow_html=True,
        )

        a, b = st.columns(2)

        with a:
            st.markdown(
                metric_card(
                    "Price momentum",
                    (
                        "Positive"
                        if month_change > 0
                        else "Negative"
                        if month_change < 0
                        else "Flat"
                    ),
                    color,
                    "Approx. one month",
                ),
                unsafe_allow_html=True,
            )

        with b:
            score = analysis["score"]

            score_color = (
                GREEN if score >= 3
                else AMBER if score == 2
                else RED
            )

            st.markdown(
                metric_card(
                    "Technical indicators",
                    f"{score} of 4 positive",
                    score_color,
                    "Trend checklist",
                ),
                unsafe_allow_html=True,
            )

        a, b = st.columns(2)

        with a:
            rsi = analysis["rsi"]

            rsi_color = (
                RED if rsi >= 70
                else GREEN if 40 <= rsi <= 65
                else AMBER
            )

            st.markdown(
                metric_card(
                    "RSI",
                    f"{rsi:.1f}",
                    rsi_color,
                    "14-period momentum index",
                ),
                unsafe_allow_html=True,
            )

        with b:
            st.markdown(
                metric_card(
                    "Annualized volatility",
                    f"{analysis['volatility']:.1f}%",
                    WHITE,
                    "Based on recent daily returns",
                ),
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # TECHNICAL OUTLOOK
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-heading">'
            'Technical outlook'
            '</div>',
            unsafe_allow_html=True,
        )

        explanations = {
            "RESEARCH ENTRY":
                "Several trend indicators are "
                "positive. Research valuation "
                "and fundamentals before investing.",

            "WATCH":
                "Signals are mixed. Monitor "
                "the trend before drawing conclusions.",

            "REVIEW RISK":
                "Trend indicators are weaker. "
                "Review risk and your investment thesis.",
        }

        st.markdown(
            f"""
            <div class="mini-card" style="
                border-left:4px solid {outlook_tint};
            ">
                <div class="mini-value"
                     style="color:{outlook_tint}">
                    {safe(outlook_text)}
                </div>
                <div style="
                    color:#B5C0CD;
                    margin-top:9px;
                    font-size:.87rem;
                    line-height:1.6;
                ">
                    {safe(explanations.get(
                        outlook,
                        "Insufficient signal data."
                    ))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # MARKET TIMESTAMPS
        # ----------------------------------------------------

        updated = format_central(
            checked_at
        )

        observed = format_central(
            latest["timestamp"],
            daily=latest["daily"],
        )

        st.markdown(
            f"""
            <div class="data-footer">
                ↻ Dashboard checked:
                {safe(updated)}
                <br>
                ◷ Latest market data:
                {safe(observed)}
                <br>
                Source observation:
                {safe(latest['kind'])}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Dashboard checks run approximately "
            "every five minutes while open. "
            "Market quotes may be delayed."
        )

    # ========================================================
    # DISCOVER
    # ========================================================

    with discover:

        st.subheader("Your watchlist")

        st.caption(
            "See which stocks have positive "
            "or negative price momentum."
        )

        rows = []

        for ticker in WATCHLIST:

            try:
                stock_data = cached_prices(ticker)
                result = analyze(stock_data)

                if result is None:
                    continue

                rows.append({
                    "Stock": ticker,
                    "Price": round(
                        result["price"], 2
                    ),
                    "1M change": round(
                        result["month_change"], 2
                    ),
                    "Indicators": result["score"],
                    "Trend": outlook_label(
                        result["signal"]
                    ),
                })

            except Exception:
                continue

        if rows:

            watchlist_df = pd.DataFrame(
                rows
            )

            watchlist_df = watchlist_df.sort_values(
                "1M change",
                ascending=False,
            )

            def color_change(value):
                return (
                    f"color: {movement_color(value)};"
                    "font-weight: 750"
                )

            def color_trend(value):
                colors = {
                    "Positive trend": GREEN,
                    "Mixed signals": AMBER,
                    "Weak trend": RED,
                }

                return (
                    f"color: {colors.get(value, MUTED)};"
                    "font-weight: 700"
                )

            styled = (
                watchlist_df.style
                .map(
                    color_change,
                    subset=["1M change"],
                )
                .map(
                    color_trend,
                    subset=["Trend"],
                )
                .format({
                    "Price": "${:,.2f}",
                    "1M change": "{:+.2f}%",
                    "Indicators": "{}/4",
                })
            )

            st.dataframe(
                styled,
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Export watchlist",
                watchlist_df.to_csv(
                    index=False
                ),
                "market_watchlist.csv",
                "text/csv",
            )

        else:
            st.info(
                "Watchlist data is unavailable."
            )

        # ----------------------------------------------------
        # STOCK COMPARISON
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-heading">'
            'Compare stocks'
            '</div>',
            unsafe_allow_html=True,
        )

        a, b = st.columns(2)

        with a:
            first = st.selectbox(
                "Stock A",
                WATCHLIST,
                key="compare_a",
            )

        with b:
            second = st.selectbox(
                "Stock B",
                WATCHLIST,
                index=min(
                    1,
                    len(WATCHLIST) - 1,
                ),
                key="compare_b",
            )

        if first == second:

            st.info(
                "Select two different stocks."
            )

        else:

            try:
                first_series = cached_prices(
                    first
                )["Close"].rename(first)

                second_series = cached_prices(
                    second
                )["Close"].rename(second)

                combined = pd.concat(
                    [
                        first_series,
                        second_series,
                    ],
                    axis=1,
                ).dropna().tail(126)

                if len(combined) > 1:

                    normalized = (
                        combined.div(
                            combined.iloc[0]
                        ) * 100
                    )

                    st.plotly_chart(
                        comparison_chart(normalized),
                        use_container_width=True,
                        config={
                            "displayModeBar": False
                        },
                    )

                    st.caption(
                        "Comparison uses approximately "
                        "six months of common trading "
                        "dates. Both stocks start at 100."
                    )

                else:
                    st.info(
                        "Not enough overlapping "
                        "data to compare."
                    )

            except Exception:
                st.info(
                    "Comparison data unavailable."
                )

    # ========================================================
    # RESEARCH
    # ========================================================

    with research:

        st.subheader(
            f"Research {symbol}"
        )

        # ----------------------------------------------------
        # INDICATOR CHECKLIST
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-heading">'
            'Technical indicators'
            '</div>',
            unsafe_allow_html=True,
        )

        labels = [
            "Above 20-day moving average",
            "Above 50-day moving average",
            "Positive monthly momentum",
            "RSI between 40 and 65",
        ]

        for label, passed in zip(
            labels,
            analysis["checks"],
        ):

            color = GREEN if passed else RED

            status = (
                "✓ Positive"
                if passed
                else "— Not met"
            )

            st.markdown(
                f"""
                <div class="indicator">
                    <span class="indicator-name">
                        {safe(label)}
                    </span>
                    <span class="indicator-status"
                          style="color:{color}">
                        {status}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Indicators are descriptive, "
            "not predictions or trading instructions."
        )

        # ----------------------------------------------------
        # FINANCIAL NEWS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-heading">'
            'Latest financial news'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            articles = cached_news(symbol)

            if not articles:
                st.info(
                    "No recent articles available."
                )

            for article in articles[:8]:

                tone = article["tone"]

                tone_color = {
                    "Positive": GREEN,
                    "Negative": RED,
                    "Neutral": AMBER,
                }.get(tone, MUTED)

                with st.container(border=True):

                    st.markdown(
                        f"**{safe(article['title'])}**"
                    )

                    st.markdown(
                        f"""
                        <span style="
                            color:{tone_color};
                            font-size:.8rem;
                            font-weight:700;
                        ">
                            ● {safe(tone)} wording
                        </span>
                        """,
                        unsafe_allow_html=True,
                    )

                    if article.get("published"):
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

        st.caption(
            "News tone is based on headline wording, "
            "not a verified assessment of the company."
        )

        # ----------------------------------------------------
        # WHAT-IF SIMULATOR
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-heading">'
            'Investment simulator'
            '</div>',
            unsafe_allow_html=True,
        )

        amount = st.number_input(
            "Hypothetical investment ($)",
            min_value=0.0,
            value=500.0,
            step=50.0,
            key="sim_amount",
        )

        hypothetical_change = st.slider(
            "Hypothetical price change (%)",
            min_value=-80,
            max_value=150,
            value=10,
            key="sim_change",
        )

        future_value = amount * (
            1 + hypothetical_change / 100
        )

        profit = future_value - amount

        a, b = st.columns(2)

        with a:
            st.markdown(
                metric_card(
                    "Potential value",
                    f"${future_value:,.2f}",
                    WHITE,
                ),
                unsafe_allow_html=True,
            )

        with b:
            st.markdown(
                metric_card(
                    "Gain or loss",
                    f"${profit:+,.2f}",
                    movement_color(profit),
                ),
                unsafe_allow_html=True,
            )

        st.caption(
            "Hypothetical scenarios only. "
            "Taxes, fees and dividends excluded."
        )

        # ----------------------------------------------------
        # RESEARCH JOURNAL
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-heading">'
            'Research journal'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.form("research_journal"):

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
                "Your investment thesis",
                placeholder=(
                    "What makes this stock "
                    "interesting to you?"
                ),
            )

            submitted = st.form_submit_button(
                "Save research note"
            )

        if submitted:

            if not thesis.strip():
                st.warning(
                    "Write a note before saving."
                )

            else:

                try:
                    with journal_connection() as conn:

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
                                latest["price"],
                                thesis.strip(),
                            ),
                        )

                    st.success(
                        "Research note saved."
                    )

                except Exception as exc:
                    st.error(
                        f"Could not save note: {exc}"
                    )

        try:
            with journal_connection() as conn:

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
                "Journal temporarily unavailable."
            )

        st.caption(
            "Journal notes are stored locally. "
            "On Streamlit Community Cloud, "
            "they may disappear after a restart. "
            "Export regularly."
        )


# ============================================================
# RUN DASHBOARD
# ============================================================

market_dashboard(selected_stock)


# ============================================================
# AUTOMATED EMAIL INFORMATION
# ============================================================

with st.expander("Email alert settings"):

    st.write(
        "GitHub Actions monitors your watchlist "
        "separately and sends emails when "
        "configured price thresholds, "
        "technical signal changes or matching "
        "financial headlines are detected."
    )

    st.caption(
        "The dashboard does not need to stay "
        "open for scheduled email monitoring."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Market Intelligence · "
    "Research and education only · "
    "Quotes may be delayed · "
    "No automatic trades"
)
