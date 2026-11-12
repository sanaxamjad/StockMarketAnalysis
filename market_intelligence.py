
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

st.set_page_config(
    page_title="Market Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent

CONFIG = json.loads(
    (ROOT / "alert_config.json").read_text()
)

WATCHLIST = CONFIG["watchlist"]

GOLD = "#DCC18C"
GREEN = "#71DDB1"
PANEL = "#171F2B"


# ============================================================
# MOBILE-FIRST DARK DESIGN
# ============================================================

st.markdown("""
<style>
html, body, .stApp,
[data-testid="stAppViewContainer"] {
    background: #0B1017;
    color: #F5F1E9;
}

.block-container {
    max-width: 1080px;
    padding: 1rem 1rem 4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    letter-spacing: -.035em;
}

.hero {
    padding: 24px;
    border: 1px solid #364456;
    border-radius: 20px;
    background: linear-gradient(
        120deg, #273341, #141B26
    );
    margin-bottom: 18px;
}

.hero h1 {
    font-size: clamp(1.9rem, 5vw, 2.8rem);
    margin: .4rem 0;
}

.hero p {
    color: #B4C0CF;
    margin: 0;
}

.eyebrow {
    color: #DCC18C;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .15em;
}

.timebox {
    background: #171F2B;
    border: 1px solid #344050;
    border-radius: 15px;
    padding: 14px;
    min-height: 106px;
}

.timebox .small {
    color: #B4C0CF;
    font-size: .74rem;
    margin-bottom: 7px;
}

.timebox .value {
    font-weight: 700;
    font-size: clamp(.82rem, 2vw, 1.05rem);
    overflow-wrap: anywhere;
}

[data-testid="stMetric"] {
    background: #171F2B;
    border: 1px solid #344050;
    border-radius: 15px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    font-size: clamp(1.2rem, 3vw, 1.85rem);
}

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

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    overflow-x: auto;
    scrollbar-width: thin;
}

[data-testid="stTabs"] button {
    white-space: nowrap;
}

.signal {
    border-left: 4px solid #DCC18C;
    background: #171F2B;
    border-radius: 12px;
    padding: 18px;
    margin: 12px 0;
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
        padding: 10px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
    }

    .stButton button {
        width: 100%;
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
# PRICE CHART
# ============================================================

def chart(df, info, days):
    part = df.tail(days)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=part.index,
        y=part["Close"],
        name="Price",
        line=dict(
            color=GOLD,
            width=3,
        ),
    ))

    fig.add_trace(go.Scatter(
        x=part.index,
        y=info["ma20"].reindex(part.index),
        name="20-day avg",
        line=dict(
            color=GREEN,
            dash="dot",
        ),
    ))

    fig.update_layout(
        height=330,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color="#B4C0CF"),
        margin=dict(l=5, r=5, t=28, b=5),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            gridcolor="#303A48",
            tickprefix="$",
        ),
        legend=dict(
            orientation="h",
            y=1.16,
        ),
        hovermode="x unified",
    )

    return fig


# ============================================================
# RESEARCH JOURNAL DATABASE
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
        Trends, research signals, and the
        stories moving your watchlist.
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
# AUTOMATIC DASHBOARD REFRESH
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

    # --------------------------------------------------------
    # TWO SEPARATE TIMESTAMPS
    # --------------------------------------------------------

    a, b = st.columns(2)

    with a:
        st.markdown(
            '<div class="timebox">'
            '<div class="small">'
            '↻ Dashboard updated'
            '</div>'
            '<div class="value">'
            + html.escape(format_central(checked))
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
        "Five-minute auto-checks while open. "
        "Data may be delayed; a bar timestamp "
        "is not the last individual trade time."
    )

    # --------------------------------------------------------
    # DASHBOARD TABS
    # --------------------------------------------------------

    overview, radar, compare, lab, journal = st.tabs([
        "Overview",
        "Radar",
        "Compare",
        "What-if",
        "Journal",
    ])

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    with overview:
        st.subheader(symbol + " · Overview")

        a, b = st.columns(2)

        a.metric(
            "Latest observed price",
            f"${bar['price']:,.2f}",
        )

        b.metric(
            "~1-month change",
            f"{info['month_change']:+.2f}%",
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
            config={"displayModeBar": False},
        )

        explanation = {
            "RESEARCH ENTRY":
                "Several price-trend indicators are "
                "positive. Check fundamentals and "
                "valuation before considering an entry.",

            "WATCH":
                "Price-trend indicators are mixed. "
                "Continue monitoring.",

            "REVIEW RISK":
                "Recent technical indicators are "
                "weaker. Reassess your thesis "
                "and exposure.",
        }[info["signal"]]

        st.markdown(
            '<div class="signal">'
            '<div class="eyebrow">'
            'TECHNICAL RESEARCH OUTLOOK'
            '</div>'
            '<h3>'
            + html.escape(info["signal"])
            + '</h3><p>'
            + html.escape(explanation)
            + '</p></div>',
            unsafe_allow_html=True,
        )

        with st.expander("Why this outlook?"):
            labels = [
                "Above 20-day average",
                "Above 50-day average",
                "Positive ~1-month momentum",
                "RSI between 40 and 65",
            ]

            for label, passed in zip(
                labels,
                info["checks"],
            ):
                st.write(
                    ("✓ " if passed else "○ ")
                    + label
                )

            st.caption(
                "Technical conditions describe "
                "historical prices; they do not "
                "predict returns."
            )

        st.subheader("Latest headlines")

        try:
            articles = news(symbol)

            if not articles:
                st.info("No headlines available.")

            for article in articles[:6]:
                with st.container(border=True):
                    st.write(
                        "**" + article["title"] + "**"
                    )

                    st.caption(
                        article["tone"]
                        + " headline wording · "
                        + article["published"]
                    )

                    st.link_button(
                        "Read source",
                        article["url"],
                    )

        except Exception:
            st.info(
                "News feed temporarily unavailable."
            )

    # --------------------------------------------------------
    # OPPORTUNITY RADAR
    # --------------------------------------------------------

    with radar:
        st.subheader("Opportunity radar")

        rows = []

        for ticker in WATCHLIST:
            try:
                result = analyze(prices(ticker))

                if result:
                    rows.append({
                        "Ticker": ticker,
                        "Close ($)": round(
                            result["price"], 2
                        ),
                        "Positive signs": result["score"],
                        "Month %": round(
                            result["month_change"], 2
                        ),
                        "Outlook": result["signal"],
                    })

            except Exception:
                pass

        if rows:
            table = pd.DataFrame(rows).sort_values(
                ["Positive signs", "Month %"],
                ascending=False,
            )

            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Export radar",
                table.to_csv(index=False),
                "market_radar.csv",
                "text/csv",
            )

        else:
            st.info("Radar data unavailable.")

    # --------------------------------------------------------
    # STOCK COMPARISON
    # --------------------------------------------------------

    with compare:
        st.subheader("Compare stocks")

        a, b = st.columns(2)

        with a:
            first = st.selectbox(
                "First",
                WATCHLIST,
                key="first",
            )

        with b:
            second = st.selectbox(
                "Second",
                WATCHLIST,
                index=min(
                    1,
                    len(WATCHLIST) - 1,
                ),
                key="second",
            )

        if first == second:
            st.info(
                "Choose two different tickers."
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
                    [first_prices, second_prices],
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
                        (first, GOLD),
                        (second, GREEN),
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

                    fig.update_layout(
                        height=320,
                        paper_bgcolor=PANEL,
                        plot_bgcolor=PANEL,
                        font=dict(
                            color="#B4C0CF"
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
                        "Both series are normalized "
                        "to 100 at the starting date."
                    )

            except Exception:
                st.info(
                    "Comparison data unavailable."
                )

    # --------------------------------------------------------
    # INVESTMENT SIMULATOR
    # --------------------------------------------------------

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

        a, b = st.columns(2)

        a.metric(
            "Potential value",
            f"${value:,.2f}",
        )

        b.metric(
            "Gain / loss",
            f"${value - amount:+,.2f}",
        )

        st.caption(
            "Illustrative arithmetic only; "
            "excludes taxes, dividends and fees."
        )

    # --------------------------------------------------------
    # RESEARCH JOURNAL
    # --------------------------------------------------------

    with journal:
        st.subheader("Research journal")

        st.caption(
            "On free Streamlit hosting, "
            "local notes may disappear after "
            "a restart. Export regularly."
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

            submitted = st.form_submit_button(
                "Save note"
            )

        if submitted:
            if thesis.strip():
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
                                symbol,
                                stance,
                                bar["price"],
                                thesis.strip(),
                            ),
                        )

                    st.success("Note saved.")

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
                    SELECT created, ticker, stance,
                           price, thesis
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


dashboard()

with st.expander("Email alerts"):
    st.write(
        "GitHub Actions checks your watchlist "
        "separately. Configure thresholds and "
        "keywords in alert_config.json. "
        "Emails use repository secrets and "
        "do not require Streamlit to stay open."
    )

st.caption(
    "Market Intelligence · Educational research "
    "only · No automatic trades or guaranteed signals."
)
