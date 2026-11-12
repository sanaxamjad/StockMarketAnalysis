
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

# ==========================================================
# SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent

with open(ROOT / "alert_config.json") as f:
    CONFIG = json.load(f)

WATCHLIST = CONFIG.get(
    "watchlist",
    ["NVDA", "CRDO", "VOO", "AAPL", "MSFT", "AMD"],
)

BG = "#0B1016"
CARD = "#18212B"
GREEN = "#25D695"
RED = "#FF596B"
AMBER = "#FFC166"
WHITE = "#F5F7FA"
MUTED = "#AAB7C5"
BLUE = "#73A9FF"

# ==========================================================
# STYLE
# ==========================================================

st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"] {
    background: #0B1016;
    color: #F5F7FA;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1050px;
    padding-top: 1.1rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    letter-spacing: -.035em;
}

[data-testid="stMetric"] {
    background: #18212B;
    border: 1px solid #293747;
    border-radius: 16px;
    padding: 15px 17px;
}

[data-testid="stMetricLabel"] {
    color: #AAB7C5;
}

[data-testid="stMetricValue"] {
    color: #F5F7FA;
}

[data-testid="stMetricDelta"] {
    font-weight: 750;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}

.stButton button {
    border-radius: 12px;
    min-height: 42px;
}

.stButton button[kind="primary"] {
    background: #C5FA5F;
    border-color: #C5FA5F;
    color: #0B1016;
    font-weight: 800;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 16px;
    overflow-x: auto;
}

[data-testid="stTabs"] button {
    white-space: nowrap;
    font-weight: 700;
}

[data-testid="stTabs"]
[data-baseweb="tab-highlight"] {
    background: #C5FA5F;
}

div[data-testid="stRadio"] label {
    font-size: .85rem;
}

@media (max-width: 640px) {
    .block-container {
        padding: .7rem .7rem 3rem;
    }

    [data-testid="stMetric"] {
        padding: 12px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }

    h1 {
        font-size: 1.7rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# DATA
# ==========================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_prices(symbol):
    return get_prices(symbol)


@st.cache_data(ttl=300, show_spinner=False)
def load_observation(symbol):
    return get_latest_bar(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def load_news(symbol):
    return get_news(symbol)


# ==========================================================
# BUY / HOLD / SELL TECHNICAL MODEL
# ==========================================================

def recommendation(info):
    """
    Educational technical signal.

    BUY: At least 3 positive indicators,
         with RSI below 70.

    SELL: At most 1 positive indicator.

    HOLD: Mixed indicators or overbought RSI.

    These are technical classifications,
    not personalized trading advice.
    """

    score = int(info["score"])
    rsi = float(info["rsi"])

    if score >= 3 and rsi < 70:
        label = "BUY"
        color = GREEN
        explanation = (
            "Technical momentum is favorable. "
            "The price trend has several positive "
            "indicators, but check valuation, "
            "earnings and your risk tolerance "
            "before considering a purchase."
        )

    elif score <= 1:
        label = "SELL"
        color = RED
        explanation = (
            "Technical momentum is weak. "
            "Several trend indicators are negative. "
            "This is a prompt to review risk, "
            "not an instruction to liquidate "
            "a long-term investment."
        )

    else:
        label = "HOLD"
        color = AMBER

        if rsi >= 70:
            explanation = (
                "The stock may be overbought "
                "based on RSI. Momentum could "
                "continue, but buying after a "
                "strong run carries additional risk."
            )
        else:
            explanation = (
                "Technical signals are mixed. "
                "There is no strong directional "
                "signal from this model."
            )

    return {
        "label": label,
        "color": color,
        "explanation": explanation,
        "score": score,
    }


def signal_strength(info):
    """
    Measures how many of the four
    technical checks agree.

    Not a probability of success.
    """
    score = int(info["score"])

    if score in (0, 4):
        return "Strong alignment"
    if score in (1, 3):
        return "Moderate alignment"
    return "Mixed indicators"


def change_color(value):
    if value > 0:
        return GREEN
    if value < 0:
        return RED
    return AMBER


# ==========================================================
# PRICE CHART
# ==========================================================

def create_price_chart(df):
    close = df["Close"].astype(float)

    first = float(close.iloc[0])
    last = float(close.iloc[-1])

    color = GREEN if last >= first else RED

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=close,
            mode="lines",
            line=dict(
                color=color,
                width=3,
            ),
            name="Price",
            hovertemplate=(
                "%{x|%b %d, %Y}"
                "<br>$%{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=310,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(
            color=MUTED,
            size=12,
        ),
        margin=dict(
            l=0,
            r=8,
            t=8,
            b=8,
        ),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#263340",
            tickprefix="$",
            side="right",
            zeroline=False,
        ),
    )

    return fig


def create_comparison(df):
    fig = go.Figure()

    for i, ticker in enumerate(df.columns):
        color = [GREEN, BLUE, AMBER, RED][i % 4]

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[ticker],
                name=ticker,
                line=dict(
                    color=color,
                    width=3,
                ),
            )
        )

    fig.add_hline(
        y=100,
        line_color=MUTED,
        line_dash="dot",
        opacity=.5,
    )

    fig.update_layout(
        height=310,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=MUTED),
        margin=dict(
            l=0, r=0, t=25, b=0
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.15,
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#263340"),
    )

    return fig


# ==========================================================
# RESEARCH JOURNAL
# ==========================================================

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


# ==========================================================
# HEADER
# ==========================================================

st.title("📈 Market Intelligence")

st.caption(
    "Your personal market desk · "
    "Simple signals. Clear research."
)

# ==========================================================
# STOCK SELECTION
# ==========================================================

left, right = st.columns(
    [4, 1],
    vertical_alignment="bottom",
)

with left:
    symbol = st.selectbox(
        "Stock",
        WATCHLIST,
    )

with right:
    refresh = st.button(
        "↻ Refresh",
        type="primary",
        use_container_width=True,
    )

if refresh:
    load_prices.clear()
    load_observation.clear()
    load_news.clear()


# ==========================================================
# MAIN DASHBOARD
# ==========================================================

@st.fragment(run_every=300)
def dashboard(ticker):

    try:
        with st.spinner("Checking market data..."):
            daily = load_prices(ticker)

            if daily is None or daily.empty:
                st.warning(
                    "No market data available."
                )
                return

            info = analyze(daily)

            if info is None:
                st.warning(
                    "Not enough history to calculate "
                    "technical indicators."
                )
                return

            latest = load_observation(ticker)

            if latest is None:
                latest = get_latest_bar(
                    ticker,
                    daily,
                )

            if latest is None:
                st.warning(
                    "Latest price unavailable."
                )
                return

            checked_at = now_central()

    except Exception as exc:
        st.error(
            f"Market data could not load: {exc}"
        )
        return

    signal = recommendation(info)

    price = float(latest["price"])
    month_change = float(info["month_change"])

    home, discover, research = st.tabs([
        "Home",
        "Discover",
        "Research",
    ])

    # ======================================================
    # HOME
    # ======================================================

    with home:

        # --------------------------------------------------
        # PRICE FIRST
        # --------------------------------------------------

        st.caption(f"{ticker} · Latest observed price")

        st.markdown(
            f"## ${price:,.2f}"
        )

        change_symbol = (
            "▲" if month_change > 0
            else "▼" if month_change < 0
            else "●"
        )

        st.markdown(
            f":{'green' if month_change > 0 else 'red' if month_change < 0 else 'orange'}["
            f"**{change_symbol} {month_change:+.2f}%**"
            f"]  ·  Past month"
        )

        st.write("")

        # --------------------------------------------------
        # BUY / HOLD / SELL
        # --------------------------------------------------

        st.subheader("Technical signal")

        if signal["label"] == "BUY":
            st.success(
                "🟢 BUY — Favorable technical setup",
                icon="📈",
            )

        elif signal["label"] == "SELL":
            st.error(
                "🔴 SELL — Weak technical setup",
                icon="📉",
            )

        else:
            st.warning(
                "🟡 HOLD — Mixed technical setup",
                icon="⚖️",
            )

        st.write(
            signal["explanation"]
        )

        a, b = st.columns(2)

        with a:
            st.metric(
                "Positive indicators",
                f"{signal['score']} / 4",
            )

        with b:
            st.metric(
                "Signal alignment",
                signal_strength(info),
            )

        st.caption(
            "BUY, HOLD and SELL are automated "
            "technical research labels, not "
            "personalized investment advice. "
            "Signal alignment is not a prediction "
            "of future returns."
        )

        # --------------------------------------------------
        # PRICE CHART
        # --------------------------------------------------

        st.subheader("Price history")

        period = st.radio(
            "Chart range",
            ["1M", "3M", "6M", "1Y", "ALL"],
            index=1,
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

        chart_data = daily.tail(days)

        if len(chart_data) >= 2:
            st.plotly_chart(
                create_price_chart(chart_data),
                use_container_width=True,
                config={
                    "displayModeBar": False
                },
            )

            period_return = (
                float(
                    chart_data["Close"].iloc[-1]
                )
                / float(
                    chart_data["Close"].iloc[0]
                )
                - 1
            ) * 100

            st.caption(
                f"{period} price change: "
                f"{period_return:+.2f}%"
            )

        # --------------------------------------------------
        # QUICK MARKET SNAPSHOT
        # --------------------------------------------------

        st.subheader("At a glance")

        a, b = st.columns(2)

        with a:
            st.metric(
                "RSI",
                f"{info['rsi']:.1f}",
            )

        with b:
            st.metric(
                "Volatility",
                f"{info['volatility']:.1f}%",
            )

        st.caption(
            "RSI measures recent price momentum. "
            "Volatility is annualized from "
            "recent daily returns."
        )

        # --------------------------------------------------
        # TIMESTAMPS
        # --------------------------------------------------

        st.divider()

        st.caption(
            "↻ Dashboard checked: "
            + format_central(checked_at)
        )

        st.caption(
            "◷ Latest market data: "
            + format_central(
                latest["timestamp"],
                daily=latest["daily"],
            )
        )

        st.caption(
            f"Observation: {latest['kind']}. "
            "Checks run about every five minutes "
            "while the dashboard is open. "
            "Data may be delayed."
        )

    # ======================================================
    # DISCOVER
    # ======================================================

    with discover:

        st.subheader("Watchlist")

        st.caption(
            "Compare stock momentum and "
            "technical research signals."
        )

        rows = []

        for stock in WATCHLIST:
            try:
                result = analyze(
                    load_prices(stock)
                )

                if result is None:
                    continue

                rec = recommendation(result)

                rows.append({
                    "Stock": stock,
                    "Price": result["price"],
                    "1M change": result[
                        "month_change"
                    ],
                    "Signal": rec["label"],
                    "Score": result["score"],
                })

            except Exception:
                continue

        if rows:
            table = pd.DataFrame(rows)

            table = table.sort_values(
                "1M change",
                ascending=False,
            )

            def color_percent(value):
                return (
                    f"color: {change_color(value)};"
                    "font-weight: bold"
                )

            def color_signal(value):
                colors = {
                    "BUY": GREEN,
                    "HOLD": AMBER,
                    "SELL": RED,
                }

                return (
                    f"color: {colors.get(value, WHITE)};"
                    "font-weight: bold"
                )

            styled = (
                table.style
                .map(
                    color_percent,
                    subset=["1M change"],
                )
                .map(
                    color_signal,
                    subset=["Signal"],
                )
                .format({
                    "Price": "${:,.2f}",
                    "1M change": "{:+.2f}%",
                    "Score": "{}/4",
                })
            )

            st.dataframe(
                styled,
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Export watchlist",
                table.to_csv(index=False),
                "watchlist.csv",
                "text/csv",
            )

        else:
            st.info(
                "Watchlist data unavailable."
            )

        # --------------------------------------------------
        # COMPARISON
        # --------------------------------------------------

        st.subheader("Compare stocks")

        a, b = st.columns(2)

        with a:
            first = st.selectbox(
                "Stock A",
                WATCHLIST,
                key="compare_first",
            )

        with b:
            second = st.selectbox(
                "Stock B",
                WATCHLIST,
                index=min(
                    1,
                    len(WATCHLIST) - 1,
                ),
                key="compare_second",
            )

        if first == second:
            st.info(
                "Choose two different stocks."
            )

        else:
            try:
                first_data = load_prices(
                    first
                )["Close"].rename(first)

                second_data = load_prices(
                    second
                )["Close"].rename(second)

                combined = pd.concat(
                    [
                        first_data,
                        second_data,
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
                        create_comparison(normalized),
                        use_container_width=True,
                        config={
                            "displayModeBar": False
                        },
                    )

                    st.caption(
                        "Both stocks are indexed "
                        "to 100 at the beginning "
                        "of the comparison."
                    )

            except Exception:
                st.info(
                    "Comparison unavailable."
                )

    # ======================================================
    # RESEARCH
    # ======================================================

    with research:

        st.subheader(
            f"{ticker} research"
        )

        # --------------------------------------------------
        # INDICATOR EXPLANATIONS
        # --------------------------------------------------

        st.subheader("Why this signal?")

        indicators = [
            "Above 20-day average",
            "Above 50-day average",
            "Positive 1-month momentum",
            "RSI between 40 and 65",
        ]

        for label, passed in zip(
            indicators,
            info["checks"],
        ):
            if passed:
                st.success(
                    f"✓ {label}"
                )
            else:
                st.error(
                    f"— {label} not met"
                )

        st.caption(
            "These four checks determine "
            "the technical signal. They do "
            "not measure valuation, earnings "
            "quality, or your personal finances."
        )

        # --------------------------------------------------
        # NEWS
        # --------------------------------------------------

        st.subheader("Financial news")

        try:
            articles = load_news(ticker)

            if not articles:
                st.info(
                    "No recent headlines available."
                )

            for article in articles[:8]:

                with st.container(border=True):
                    st.markdown(
                        f"**{article['title']}**"
                    )

                    tone = article["tone"]

                    if tone == "Positive":
                        st.caption(
                            "🟢 Positive headline wording"
                        )

                    elif tone == "Negative":
                        st.caption(
                            "🔴 Negative headline wording"
                        )

                    else:
                        st.caption(
                            "🟡 Neutral headline wording"
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
            "Headline tone is a text-based "
            "sentiment estimate, not a "
            "verified assessment of the news."
        )

        # --------------------------------------------------
        # WHAT-IF SIMULATOR
        # --------------------------------------------------

        st.subheader("Investment simulator")

        amount = st.number_input(
            "Hypothetical investment ($)",
            min_value=0.0,
            value=500.0,
            step=50.0,
        )

        move = st.slider(
            "Hypothetical price move (%)",
            -80,
            150,
            10,
        )

        future_value = amount * (
            1 + move / 100
        )

        profit = future_value - amount

        a, b = st.columns(2)

        with a:
            st.metric(
                "Potential value",
                f"${future_value:,.2f}",
            )

        with b:
            st.metric(
                "Gain / loss",
                f"${profit:+,.2f}",
                delta=f"{move:+d}%",
            )

        st.caption(
            "Illustrative arithmetic only. "
            "Excludes fees, taxes and dividends."
        )

        # --------------------------------------------------
        # JOURNAL
        # --------------------------------------------------

        st.subheader("Research journal")

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
                "Your research notes"
            )

            save = st.form_submit_button(
                "Save note"
            )

        if save:
            if not thesis.strip():
                st.warning(
                    "Write a note first."
                )
            else:
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
                                ticker,
                                stance,
                                price,
                                thesis.strip(),
                            ),
                        )

                    st.success(
                        "Note saved."
                    )

                except Exception as exc:
                    st.error(
                        f"Could not save note: {exc}"
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

        st.caption(
            "Journal storage on Streamlit "
            "Community Cloud is temporary. "
            "Export notes regularly."
        )


# ==========================================================
# RUN
# ==========================================================

dashboard(symbol)

# ==========================================================
# EMAIL AUTOMATION
# ==========================================================

with st.expander("Automated email alerts"):
    st.write(
        "Your GitHub Actions workflow monitors "
        "stocks separately from this dashboard."
    )

    st.caption(
        "Emails continue to use the alert "
        "rules in your existing alert_worker.py. "
        "The dashboard's BUY / HOLD / SELL "
        "labels do not change those email rules."
    )

st.divider()

st.caption(
    "Market Intelligence · "
    "Technical research only · "
    "Not personalized investment advice · "
    "No automatic trading"
)
