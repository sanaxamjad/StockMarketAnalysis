
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
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent

with open(ROOT / "alert_config.json") as file:
    CONFIG = json.load(file)

WATCHLIST = CONFIG.get(
    "watchlist",
    ["NVDA", "CRDO", "VOO", "AAPL", "MSFT", "AMD"],
)

BG = "#0B1016"
PANEL = "#17212C"
GREEN = "#29D99A"
RED = "#FF6173"
AMBER = "#F4BE62"
BLUE = "#79AEFF"
WHITE = "#F5F7FA"
MUTED = "#AAB7C5"

# ============================================================
# DESIGN
# ============================================================

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
    padding-top: 1rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    letter-spacing: -.035em;
}

[data-testid="stMetric"] {
    background: #17212C;
    border: 1px solid #293747;
    border-radius: 15px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #AAB7C5;
}

[data-testid="stMetricValue"] {
    color: #F5F7FA;
}

.stButton button {
    border-radius: 12px;
    min-height: 42px;
}

.stButton button[kind="primary"] {
    background: #C5FA5F;
    color: #0B1016;
    border: none;
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

@media (max-width: 640px) {
    .block-container {
        padding: .7rem .7rem 3rem;
    }

    [data-testid="stMetric"] {
        padding: 11px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_prices(symbol):
    return get_prices(symbol)


@st.cache_data(ttl=300, show_spinner=False)
def load_latest(symbol):
    return get_latest_bar(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def load_news(symbol):
    return get_news(symbol)


# ============================================================
# FREE FINBERT AI MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading financial AI...")
def load_finbert():
    """
    Downloads FinBERT once and reuses it.
    No API key or paid service required.
    """
    from transformers import pipeline

    return pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
        device=-1,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def analyze_headlines(titles):
    """
    Classify headlines as positive,
    negative, or neutral using FinBERT.
    """

    if not titles:
        return None

    model = load_finbert()

    predictions = model(
        list(titles),
        truncation=True,
        max_length=256,
        batch_size=8,
    )

    scores = {
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 0.0,
    }

    analyzed = []

    for title, prediction in zip(
        titles,
        predictions,
    ):
        label = prediction["label"].lower()
        confidence = float(prediction["score"])

        if label not in scores:
            label = "neutral"

        scores[label] += confidence

        analyzed.append({
            "title": title,
            "sentiment": label,
            "confidence": confidence,
        })

    total = sum(scores.values())

    if total <= 0:
        return None

    positive = scores["positive"] / total
    negative = scores["negative"] / total
    neutral = scores["neutral"] / total

    net = positive - negative

    if net > 0.20:
        direction = "POSITIVE"
    elif net < -0.20:
        direction = "NEGATIVE"
    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "net": net,
        "articles": analyzed,
        "count": len(analyzed),
    }


def news_assessment(articles):
    if not articles:
        return None, "No headlines available."

    titles = tuple(
        str(article.get("title", "")).strip()
        for article in articles[:10]
        if article.get("title")
    )

    if not titles:
        return None, "No usable headlines."

    try:
        return analyze_headlines(titles), None
    except Exception:
        return None, (
            "Financial AI could not load. "
            "Showing technical-only analysis."
        )


# ============================================================
# COMBINED BUY / HOLD / SELL RESEARCH SIGNAL
# ============================================================

def make_signal(info, news_result):
    """
    Technical score: 0-4 points.

    FinBERT sentiment adjustment:
      Positive: +1
      Neutral:   0
      Negative: -1

    Research signal, not investment advice.
    """

    technical_score = int(info["score"])
    rsi = float(info["rsi"])

    if news_result is None:
        adjustment = 0
        news_label = "Unavailable"
    else:
        news_label = news_result["direction"]

        adjustment = {
            "POSITIVE": 1,
            "NEUTRAL": 0,
            "NEGATIVE": -1,
        }[news_label]

    total = max(
        0,
        min(5, technical_score + adjustment),
    )

    if total >= 3 and rsi < 70:
        label = "BUY"
        color = GREEN

    elif total <= 1:
        label = "SELL"
        color = RED

    else:
        label = "HOLD"
        color = AMBER

    if news_result is None:
        explanation = (
            "This assessment uses technical "
            "indicators only because AI news "
            "analysis is unavailable."
        )

    elif label == "BUY":
        explanation = (
            "The combined technical and headline "
            "sentiment score is favorable. "
            "Review valuation, fundamentals, "
            "and company-specific risks before "
            "considering a purchase."
        )

    elif label == "SELL":
        explanation = (
            "The combined indicators suggest "
            "a weak setup. Review the company's "
            "fundamentals and your investment "
            "horizon before changing a position."
        )

    else:
        explanation = (
            "The combined indicators are mixed "
            "or momentum may be stretched. "
            "Monitoring further developments "
            "may be more useful than relying "
            "on a directional signal."
        )

    return {
        "label": label,
        "color": color,
        "technical": technical_score,
        "news": news_label,
        "total": total,
        "explanation": explanation,
    }


def movement_color(value):
    if value > 0:
        return GREEN
    if value < 0:
        return RED
    return AMBER


# ============================================================
# CHARTS
# ============================================================

def price_chart(df):
    close = df["Close"].astype(float)

    positive = close.iloc[-1] >= close.iloc[0]
    color = GREEN if positive else RED

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
            hovertemplate=(
                "%{x|%b %d, %Y}"
                "<br>$%{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=315,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=MUTED),
        margin=dict(
            l=0,
            r=8,
            t=10,
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
            gridcolor="#273443",
            tickprefix="$",
            side="right",
            zeroline=False,
        ),
    )

    return fig


def comparison_chart(df):
    fig = go.Figure()

    for i, ticker in enumerate(df.columns):
        color = [GREEN, BLUE][i % 2]

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
        line_dash="dot",
        line_color=MUTED,
        opacity=.5,
    )

    fig.update_layout(
        height=320,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=MUTED),
        margin=dict(
            l=0, r=0, t=25, b=0
        ),
        legend=dict(
            orientation="h",
            y=1.15,
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#273443"),
    )

    return fig


# ============================================================
# RESEARCH JOURNAL
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
# HEADER
# ============================================================

st.title("📈 Market Intelligence")

st.caption(
    "Your personal market desk · "
    "Technical analysis + free financial AI"
)

left, right = st.columns(
    [4, 1],
    vertical_alignment="bottom",
)

with left:
    symbol = st.selectbox(
        "Choose a stock",
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
    load_latest.clear()
    load_news.clear()
    analyze_headlines.clear()


# ============================================================
# MAIN DASHBOARD
# ============================================================

@st.fragment(run_every=300)
def dashboard(ticker):

    try:
        with st.spinner("Loading market data..."):

            daily = load_prices(ticker)

            if daily is None or daily.empty:
                st.warning(
                    "Market data unavailable."
                )
                return

            info = analyze(daily)

            if info is None:
                st.warning(
                    "Insufficient price history."
                )
                return

            latest = load_latest(ticker)

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
            f"Market data error: {exc}"
        )
        return

    price = float(latest["price"])
    month_change = float(info["month_change"])

    try:
        articles = load_news(ticker)
    except Exception:
        articles = []

    news_result, news_error = news_assessment(
        articles
    )

    signal = make_signal(
        info,
        news_result,
    )

    # ========================================================
    # SMALL REFRESH STATUS LINE AT TOP
    # ========================================================

    dashboard_time = format_central(checked_at)

    market_time = format_central(
        latest["timestamp"],
        daily=latest["daily"],
    )

    st.caption(
        f"↻ Dashboard: {dashboard_time}   ·   "
        f"◷ Market: {market_time}"
    )

    # ========================================================
    # NAVIGATION
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

        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        st.caption(
            f"{ticker} · Latest observed price"
        )

        st.markdown(
            f"# ${price:,.2f}"
        )

        arrow = (
            "▲" if month_change > 0
            else "▼" if month_change < 0
            else "●"
        )

        color_name = (
            "green" if month_change > 0
            else "red" if month_change < 0
            else "orange"
        )

        st.markdown(
            f":{color_name}["
            f"**{arrow} {month_change:+.2f}%**"
            f"] · Past month"
        )

        st.divider()

        # ----------------------------------------------------
        # COMBINED SIGNAL
        # ----------------------------------------------------

        st.subheader(
            "BUY / HOLD / SELL"
        )

        if signal["label"] == "BUY":
            st.success(
                "🟢 BUY · Favorable research setup"
            )

        elif signal["label"] == "SELL":
            st.error(
                "🔴 SELL · Weak research setup"
            )

        else:
            st.warning(
                "🟡 HOLD · Mixed research setup"
            )

        st.write(
            signal["explanation"]
        )

        # ----------------------------------------------------
        # CORE INDICATORS
        # ----------------------------------------------------

        a, b = st.columns(2)

        with a:
            st.metric(
                "Technical indicators",
                f"{signal['technical']} / 4",
            )

        with b:
            st.metric(
                "AI news sentiment",
                signal["news"].title(),
            )

        # ----------------------------------------------------
        # HEADLINE SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "What the headlines suggest"
        )

        if news_result:

            direction = news_result["direction"]

            if direction == "POSITIVE":
                st.success(
                    "Recent headline wording "
                    "leans positive."
                )

            elif direction == "NEGATIVE":
                st.error(
                    "Recent headline wording "
                    "leans negative."
                )

            else:
                st.info(
                    "Recent headline wording "
                    "is mixed or neutral."
                )

            st.caption(
                f"FinBERT analyzed "
                f"{news_result['count']} headlines."
            )

            a, b, c = st.columns(3)

            a.metric(
                "Positive",
                f"{news_result['positive']:.0%}",
            )

            b.metric(
                "Negative",
                f"{news_result['negative']:.0%}",
            )

            c.metric(
                "Neutral",
                f"{news_result['neutral']:.0%}",
            )

        else:
            st.info(
                news_error
                or "News analysis unavailable."
            )

        # ----------------------------------------------------
        # PRICE CHART
        # ----------------------------------------------------

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

        selected_data = daily.tail(days)

        if len(selected_data) > 1:

            st.plotly_chart(
                price_chart(selected_data),
                use_container_width=True,
                config={
                    "displayModeBar": False
                },
            )

            period_change = (
                float(
                    selected_data["Close"].iloc[-1]
                )
                / float(
                    selected_data["Close"].iloc[0]
                )
                - 1
            ) * 100

            st.caption(
                f"{period} price change: "
                f"{period_change:+.2f}%"
            )

        # ----------------------------------------------------
        # QUICK STATS
        # ----------------------------------------------------

        st.subheader("At a glance")

        a, b = st.columns(2)

        with a:
            st.metric(
                "RSI",
                f"{info['rsi']:.1f}",
            )

        with b:
            st.metric(
                "Annualized volatility",
                f"{info['volatility']:.1f}%",
            )

        st.caption(
            "Research signals use technical "
            "indicators and headline sentiment. "
            "They do not measure fair value "
            "or guarantee returns. "
            "The dashboard checks approximately "
            "every five minutes while open."
        )

    # ========================================================
    # DISCOVER
    # ========================================================

    with discover:

        st.subheader("Your watchlist")

        st.caption(
            "Compare stock prices and "
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

                stock_signal = make_signal(
                    result,
                    None,
                )

                rows.append({
                    "Stock": stock,
                    "Price": result["price"],
                    "1M change": result[
                        "month_change"
                    ],
                    "Signal": stock_signal["label"],
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

            def style_change(value):
                return (
                    f"color:{movement_color(value)};"
                    "font-weight:bold"
                )

            def style_signal(value):
                colors = {
                    "BUY": GREEN,
                    "HOLD": AMBER,
                    "SELL": RED,
                }

                return (
                    f"color:{colors.get(value, WHITE)};"
                    "font-weight:bold"
                )

            styled = (
                table.style
                .map(
                    style_change,
                    subset=["1M change"],
                )
                .map(
                    style_signal,
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

            st.caption(
                "Watchlist signals are technical-only. "
                "Select a stock on Home to include "
                "FinBERT headline sentiment."
            )

            st.download_button(
                "Export watchlist",
                table.to_csv(index=False),
                "watchlist.csv",
                "text/csv",
            )

        # ----------------------------------------------------
        # COMPARE STOCKS
        # ----------------------------------------------------

        st.subheader("Compare stocks")

        a, b = st.columns(2)

        with a:
            first = st.selectbox(
                "Stock A",
                WATCHLIST,
                key="first_stock",
            )

        with b:
            second = st.selectbox(
                "Stock B",
                WATCHLIST,
                index=min(
                    1,
                    len(WATCHLIST) - 1,
                ),
                key="second_stock",
            )

        if first == second:
            st.info(
                "Select two different stocks."
            )

        else:
            try:
                first_close = load_prices(
                    first
                )["Close"].rename(first)

                second_close = load_prices(
                    second
                )["Close"].rename(second)

                combined = pd.concat(
                    [
                        first_close,
                        second_close,
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
                        "Both stocks are normalized "
                        "to 100 at the start."
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
            f"{ticker} · Detailed research"
        )

        # ----------------------------------------------------
        # TECHNICAL INDICATORS
        # ----------------------------------------------------

        st.subheader(
            "Technical indicator breakdown"
        )

        labels = [
            "Price above 20-day average",
            "Price above 50-day average",
            "Positive monthly momentum",
            "RSI between 40 and 65",
        ]

        for label, passed in zip(
            labels,
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

        # ----------------------------------------------------
        # AI HEADLINE BREAKDOWN
        # ----------------------------------------------------

        st.subheader(
            "AI-analyzed financial headlines"
        )

        analyzed_lookup = {}

        if news_result:
            for item in news_result["articles"]:
                analyzed_lookup[
                    item["title"]
                ] = item

        if not articles:
            st.info(
                "No recent headlines available."
            )

        for article in articles[:10]:

            title = article.get(
                "title",
                "Untitled article",
            )

            with st.container(border=True):

                st.markdown(
                    f"**{title}**"
                )

                result = analyzed_lookup.get(
                    title
                )

                if result:

                    sentiment = result[
                        "sentiment"
                    ]

                    confidence = result[
                        "confidence"
                    ]

                    if sentiment == "positive":
                        st.success(
                            "🟢 Positive headline sentiment"
                        )

                    elif sentiment == "negative":
                        st.error(
                            "🔴 Negative headline sentiment"
                        )

                    else:
                        st.info(
                            "🟡 Neutral headline sentiment"
                        )

                    st.caption(
                        "Model classification score: "
                        f"{confidence:.0%}. "
                        "Not a probability of "
                        "stock price movement."
                    )

                if article.get("published"):
                    st.caption(
                        article["published"]
                    )

                if article.get("url"):
                    st.link_button(
                        "Read article",
                        article["url"],
                    )

        st.caption(
            "FinBERT analyzes headline text, "
            "not full article bodies. "
            "Sentiment may miss nuance, "
            "duplicates, or misleading headlines."
        )

        # ----------------------------------------------------
        # INVESTMENT SIMULATOR
        # ----------------------------------------------------

        st.subheader(
            "Investment simulator"
        )

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

        future = amount * (
            1 + move / 100
        )

        gain = future - amount

        a, b = st.columns(2)

        with a:
            st.metric(
                "Potential value",
                f"${future:,.2f}",
            )

        with b:
            st.metric(
                "Gain / loss",
                f"${gain:+,.2f}",
                delta=f"{move:+d}%",
            )

        st.caption(
            "Illustrative arithmetic only. "
            "Excludes fees, taxes and dividends."
        )

        # ----------------------------------------------------
        # JOURNAL
        # ----------------------------------------------------

        st.subheader(
            "Research journal"
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
                            (created, ticker, stance,
                             price, thesis)
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
                        "Research note saved."
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
                    notes.to_csv(index=False),
                    "research_journal.csv",
                    "text/csv",
                )

        except Exception:
            st.info(
                "Journal unavailable."
            )

        st.caption(
            "Research journal storage on "
            "Streamlit Community Cloud may "
            "reset after an app restart. "
            "Export notes regularly."
        )


# ============================================================
# RUN DASHBOARD
# ============================================================

dashboard(symbol)

# ============================================================
# EMAIL AUTOMATION
# ============================================================

with st.expander("Automated email alerts"):

    st.write(
        "Your GitHub Actions email automation "
        "continues to operate independently."
    )

    st.caption(
        "Existing email alerts are unchanged. "
        "They do not yet use FinBERT's "
        "combined research signal."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Market Intelligence · "
    "Free financial AI · "
    "Educational research only · "
    "No automatic trading"
)
