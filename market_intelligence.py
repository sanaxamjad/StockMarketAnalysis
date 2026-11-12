
import json
import re
import sqlite3
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_core import analyze, format_central, get_latest_bar, get_news, get_prices, now_central

st.set_page_config(page_title="Market Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent
GREEN, RED, AMBER, BLUE = "#39E6A5", "#FF7185", "#FFC778", "#85AEFF"
BG, MUTED = "#090E16", "#A4B3C7"
try:
    CONFIG = json.loads((ROOT / "alert_config.json").read_text())
except (OSError, ValueError):
    CONFIG = {}
WATCHLIST = CONFIG.get("watchlist", ["NVDA", "CRDO", "VOO", "AAPL", "MSFT", "AMD"])
if not isinstance(WATCHLIST, list) or not WATCHLIST:
    WATCHLIST = ["NVDA", "CRDO", "VOO", "AAPL", "MSFT", "AMD"]
WATCHLIST = list(dict.fromkeys(str(x).upper() for x in WATCHLIST))
VALID = re.compile(r"^[A-Z0-9^][A-Z0-9.^=-]{0,19}$")

st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 85% 0%,#17273b 0%,transparent 35%),radial-gradient(ellipse at 0% 35%,#112b27 0%,transparent 36%),#090e16;color:#f4f7fc}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1190px;padding-top:1.1rem;padding-bottom:4rem}
h1,h2,h3{letter-spacing:-.04em!important} h1{font-weight:850!important}
p,li{color:#c4cfdd}
[data-testid="stMetric"]{background:linear-gradient(140deg,#192637,#111b29);border:1px solid #304156;border-radius:17px;padding:17px}
[data-testid="stMetricLabel"]{color:#a4b3c7} [data-testid="stMetricValue"]{color:#f7f9fd}
[data-testid="stMetricDelta"]{font-weight:750}
.stButton button,.stDownloadButton button,[data-testid="stLinkButton"] a{border-radius:12px!important;min-height:43px;font-weight:750;border-color:#33465a;background:#172434;color:#edf4ff}
.stButton button[kind="primary"]{background:#c9fa75;color:#0a1510;border:none}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:22px;overflow-x:auto;border-bottom:1px solid #304156}
[data-testid="stTabs"] button{font-weight:800;color:#9eb0c5;white-space:nowrap}
[data-testid="stTabs"] [aria-selected="true"]{color:#39e6a5!important}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:#39e6a5}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stTextArea"] textarea,[data-baseweb="select"]>div{background:#142132!important;color:#f6f9fc!important;border-radius:12px!important;border-color:#304156!important}
[data-testid="stDataFrame"], [data-testid="stForm"], [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"]{border-radius:17px!important}
[data-testid="stAlert"]{border-radius:14px}
[data-testid="stRadio"] label{color:#d8e4f0}
.mi-hero{background:linear-gradient(115deg,#1b3036,#17243a 65%,#26233d);border:1px solid #3b5062;border-radius:25px;padding:28px;margin:15px 0 22px}
.mi-eyebrow{color:#a6bbcf;font-size:12px;font-weight:850;letter-spacing:1.5px;text-transform:uppercase}
.mi-price{font-size:clamp(40px,6vw,67px);letter-spacing:-3px;color:#fff;font-weight:850;margin:7px 0}
.mi-chip{display:inline-block;border-radius:999px;padding:7px 13px;font-weight:850;font-size:12px;letter-spacing:.6px}
.mi-card{background:linear-gradient(145deg,#1b2a3b,#121d2a);border:1px solid #34465a;border-radius:20px;padding:20px;margin:6px 0 15px;min-height:190px}
.mi-card-title{font-size:12px;font-weight:850;letter-spacing:1.2px}
.mi-card-price{font-size:clamp(21px,2.5vw,30px);font-weight:850;letter-spacing:-1px;margin:15px 0 8px}
.mi-card-body{font-size:13px;color:#b7c6d7;line-height:1.65}
.mi-panel{background:linear-gradient(125deg,#172637,#101b29);border:1px solid #304156;border-radius:18px;padding:19px;margin:12px 0 17px}
.mi-panel h4{color:#f4f7fc;margin:0 0 8px;font-size:17px}
.mi-panel p{color:#b7c6d7;font-size:14px;margin:0;line-height:1.65}
.mi-mini{color:#9fb0c5;font-size:12px}
@media(max-width:650px){.block-container{padding:.7rem .65rem 3rem}.mi-hero{padding:19px}.mi-card{min-height:0}.mi-price{letter-spacing:-1px}[data-testid="stMetric"]{padding:11px}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def prices(symbol):
    return get_prices(symbol)

@st.cache_data(ttl=300, show_spinner=False)
def latest_price(symbol):
    return get_latest_bar(symbol)

@st.cache_data(ttl=900, show_spinner=False)
def headlines(symbol):
    return get_news(symbol)

@st.cache_resource(show_spinner=False)
def finbert():
    from transformers import pipeline
    return pipeline("text-classification", model="ProsusAI/finbert", device=-1)

@st.cache_data(ttl=1800, show_spinner=False)
def classify_news(titles):
    if not titles:
        return None
    results = finbert()(list(titles), truncation=True, max_length=256, batch_size=8)
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    items = []
    for title, prediction in zip(titles, results):
        label = prediction["label"].lower()
        if label not in counts:
            label = "neutral"
        counts[label] += 1
        items.append({"title": title, "label": label, "confidence": float(prediction["score"])})
    shares = {k: v / len(items) for k, v in counts.items()}
    net = shares["positive"] - shares["negative"]
    direction = "POSITIVE" if net > .2 else "NEGATIVE" if net < -.2 else "NEUTRAL"
    return {"direction": direction, "shares": shares, "items": items, "count": len(items)}

def news_analysis(articles):
    titles = tuple(str(a.get("title", "")).strip() for a in articles[:10] if a.get("title"))
    try:
        return classify_news(titles), None
    except Exception:
        return None, "FinBERT unavailable; technical analysis still works."

def research_signal(info, news):
    adjustment = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}.get(news["direction"], 0) if news else 0
    total = max(0, min(5, int(info["score"]) + adjustment))
    if total >= 3 and float(info["rsi"]) < 70:
        return "BULLISH", total
    if total <= 1:
        return "BEARISH", total
    return "MIXED", total

def price_levels(df):
    """Calculate historical support and resistance reference bands."""
    if df is None or len(df) < 65:
        return None
    data = df.dropna(subset=["High", "Low", "Close"]).copy()
    if len(data) < 65:
        return None
    high, low, close = (data[c].astype(float) for c in ("High", "Low", "Close"))
    previous = close.shift(1)
    tr = pd.concat([high-low, (high-previous).abs(), (low-previous).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    support = float(low.tail(60).min())
    resistance = float(high.tail(60).max())
    if not all(np.isfinite(v) for v in (atr, support, resistance)) or atr <= 0 or resistance <= support:
        return None
    width = min(atr * .5, (resistance-support) * .2)
    buy_low, buy_high = support, support + width
    sell_low, sell_high = resistance-width, resistance
    if buy_high >= sell_low:
        return None
    risk = max(.01, support - atr)
    return {
        "buy_low": buy_low, "buy_high": buy_high,
        "sell_low": sell_low, "sell_high": sell_high,
        "risk": risk, "support": support, "resistance": resistance,
        "atr": atr, "ma20": float(close.tail(20).mean()),
        "ma50": float(close.tail(50).mean())
    }

def money(value):
    return f"${value:,.2f}" if value is not None else "Unavailable"

def zone_range(a, b):
    return f"{money(a)} – {money(b)}"

def classify_price(current, levels):
    if levels is None:
        return "UNAVAILABLE", "Not enough history to estimate meaningful levels.", AMBER
    if current < levels["risk"]:
        return "BREAKDOWN", "Price is below the historical risk reference. A lower price alone is NOT a buy signal; support may have failed.", RED
    if current <= levels["buy_high"]:
        return "BUY ZONE", "Price is near the recent 60-day low. This is a potential entry area to research, not proof the stock is undervalued.", GREEN
    if current >= levels["sell_low"]:
        if current > levels["sell_high"]:
            return "BREAKOUT", "Price is above recent 60-day resistance. This could be a breakout; a high price alone does NOT mean sell.", BLUE
        return "SELL ZONE", "Price is near the recent 60-day high. Existing holders may review profit-taking; this is not a mandatory sell.", RED
    return "HOLD / WAIT", "Price is between the estimated entry and profit-taking zones. Monitor for a clearer setup.", AMBER

def action_explanation(price_state, trend):
    if price_state == "BUY ZONE":
        return "Price is low relative to its recent range. " + ("Trend indicators are supportive, but check company fundamentals." if trend == "BULLISH" else "Trend indicators are not clearly supportive, so investigate why the stock fell.")
    if price_state == "SELL ZONE":
        return "Price is near historical resistance. If you already own it, compare the target with your purchase price and time horizon."
    if price_state == "BREAKDOWN":
        return "Support may have failed. Avoid interpreting the lower price as an automatic bargain."
    if price_state == "BREAKOUT":
        return "The stock exceeded its recent high. Reassess momentum and valuation; the old sell target may no longer be useful."
    if price_state == "HOLD / WAIT":
        return "For a new purchase, you could monitor for a move toward support. Existing holders should consider their own cost basis and goals."
    return "No price-based decision is available for this ticker."

def html_card(title, value, description, color):
    return f"""<div class='mi-card' style='border-top:3px solid {color}'>
    <div class='mi-card-title' style='color:{color}'>{escape(title)}</div>
    <div class='mi-card-price' style='color:{color}'>{escape(value)}</div>
    <div class='mi-card-body'>{escape(description)}</div></div>"""

def chart(df, levels):
    close = df["Close"].astype(float)
    color = GREEN if close.iloc[-1] >= close.iloc[0] else RED
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=close, mode="lines", name="Price",
        line=dict(color=color, width=3),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>"
    ))
    if levels:
        fig.add_hrect(y0=levels["buy_low"], y1=levels["buy_high"], fillcolor=GREEN, opacity=.15, line_width=0)
        fig.add_hrect(y0=levels["sell_low"], y1=levels["sell_high"], fillcolor=RED, opacity=.15, line_width=0)
        fig.add_hline(y=levels["risk"], line_color=AMBER, line_dash="dot", opacity=.7)
    fig.update_layout(
        height=370, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED), margin=dict(l=0,r=5,t=10,b=5), showlegend=False,
        hovermode="x unified", xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#25364A", tickprefix="$", side="right")
    )
    return fig

def comparison_chart(df):
    fig = go.Figure()
    for i, symbol in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[symbol], name=symbol,
            line=dict(color=[GREEN,BLUE][i%2], width=3)
        ))
    fig.update_layout(
        height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED), margin=dict(l=0,r=0,t=25,b=0),
        legend=dict(orientation="h",y=1.15), xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#25364A")
    )
    return fig

def journal_db():
    conn = sqlite3.connect(ROOT / "market_journal.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS journal(
        id INTEGER PRIMARY KEY, created TEXT, ticker TEXT,
        stance TEXT, price REAL, thesis TEXT
    )""")
    conn.commit()
    return conn

st.title("📈 Market Intelligence")
st.caption("Your personal market desk · Free financial AI · Clear price decisions")
left, right = st.columns([4,1], vertical_alignment="bottom")
with left:
    a,b = st.columns([2,1])
    with a:
        search = st.text_input("Search any stock ticker", placeholder="NVDA, CRDO, TSLA, PLTR, BRK-B...").strip().upper()
    with b:
        selected = st.selectbox("Your watchlist", WATCHLIST)
symbol = search or selected
with right:
    refresh = st.button("↻ Refresh", type="primary", use_container_width=True)
if refresh:
    prices.clear()
    latest_price.clear()
    headlines.clear()
    classify_news.clear()
if not VALID.fullmatch(symbol):
    st.error("Enter a valid Yahoo Finance ticker symbol.")
    st.stop()

@st.fragment(run_every=300)
def dashboard(ticker):
    try:
        daily = prices(ticker)
        if daily is None or daily.empty:
            st.warning(f"No market data found for {ticker}.")
            return
        info = analyze(daily)
        if info is None:
            st.warning("Not enough price history for this stock.")
            return
        latest = latest_price(ticker) or get_latest_bar(ticker, daily)
        if latest is None:
            st.warning("Latest price unavailable.")
            return
        checked = now_central()
    except Exception as exc:
        st.error(f"Market data error: {exc}")
        return

    current = float(latest["price"])
    change = float(info["month_change"])
    try:
        articles = headlines(ticker)
    except Exception:
        articles = []
    news, news_error = news_analysis(articles)
    trend, trend_score = research_signal(info, news)
    levels = price_levels(daily)
    state, state_reason, state_color = classify_price(current, levels)
    explanation = action_explanation(state, trend)

    st.caption(
        f"↻ Dashboard: {format_central(checked)} · "
        f"◷ Market: {format_central(latest['timestamp'], daily=latest['daily'])}"
    )
    home, discover, research = st.tabs(["⌂ Home", "◈ Discover", "▤ Research"])

    with home:
        movement = GREEN if change >= 0 else RED
        arrow = "▲" if change >= 0 else "▼"
        st.markdown(f"""<div class='mi-hero'>
        <div class='mi-eyebrow'>{escape(ticker)} · MARKET OVERVIEW</div>
        <div class='mi-price'>{money(current)}</div>
        <div style='color:{movement};font-weight:850;margin-bottom:16px'>{arrow} {change:+.2f}% · Past month</div>
        <span class='mi-chip' style='background:{state_color}25;color:{state_color}'>{escape(state)}</span>
        <span class='mi-chip' style='background:#26384a;color:#cbd9ec;margin-left:7px'>{escape(trend)} TREND</span>
        <p style='margin:17px 0 0;color:#c4d2e0;line-height:1.75'>{escape(explanation)}</p>
        </div>""", unsafe_allow_html=True)

        st.subheader("When to buy, hold, or sell")
        st.caption("These are historical price reference zones, not automatic trading instructions.")
        if levels:
            a,b,c = st.columns(3)
            with a:
                st.markdown(html_card("↓ BUY ZONE · NEAR RECENT LOW",zone_range(levels["buy_low"],levels["buy_high"]),"If the stock falls into this range, it is near recent support. Research a possible entry; first check whether the business or trend has deteriorated.",GREEN),unsafe_allow_html=True)
            with b:
                st.markdown(html_card("◷ HOLD / WAIT · BETWEEN LEVELS",zone_range(levels["buy_high"],levels["sell_low"]),"Between the buy and sell zones, the price is not at either extreme. New buyers may wait; existing holders should review their own investment plan.",AMBER),unsafe_allow_html=True)
            with c:
                st.markdown(html_card("↑ SELL ZONE · NEAR RECENT HIGH",zone_range(levels["sell_low"],levels["sell_high"]),"If the stock rises into this range, it is near recent resistance. Existing holders may research taking profits, based on their purchase price and goals.",RED),unsafe_allow_html=True)
            st.markdown(f"""<div class='mi-panel'><h4>What the dashboard says right now: <span style='color:{state_color}'>{escape(state)}</span></h4>
            <p>{escape(state_reason)}<br><br><b>Risk reference:</b> Below {money(levels['risk'])}, where the support idea may have failed.
            <br>Zones use a 14-session volatility buffer (ATR {money(levels['atr'])}). They are historical reference ranges, not fair-value estimates or guaranteed future prices.</p></div>""", unsafe_allow_html=True)
        else:
            st.info("Not enough reliable history to calculate price zones.")

        st.subheader("Price chart")
        period = st.radio("Chart period", ["1M", "3M", "6M", "1Y", "ALL"], index=2, horizontal=True, label_visibility="collapsed")
        days = {"1M":22,"3M":66,"6M":132,"1Y":252,"ALL":len(daily)}[period]
        st.plotly_chart(chart(daily.tail(days), levels), use_container_width=True, config={"displayModeBar":False})
        st.caption("Green band = potential buy area · Red band = potential sell area · Amber dotted line = support breakdown reference. Levels can fall outside the selected chart view.")

        st.subheader("Technical & AI outlook")
        a, b, c = st.columns(3)
        a.metric("Trend assessment", trend)
        b.metric("Technical checks", f"{info['score']} / 4")
        c.metric("RSI", f"{info['rsi']:.1f}")
        st.markdown(f"""<div class='mi-panel'><h4>How to read this</h4><p>
        The <b>price zone</b> answers “Is the stock near its recent low or high?” The <b>{escape(trend.lower())} trend</b> answers “Do momentum and recent headline indicators support the setup?” They can disagree. Neither predicts the future.</p></div>""", unsafe_allow_html=True)
        st.subheader("Financial news sentiment")
        if news:
            a, b, c = st.columns(3)
            a.metric("Positive headlines", f"{news['shares']['positive']:.0%}")
            b.metric("Negative headlines", f"{news['shares']['negative']:.0%}")
            c.metric("Neutral headlines", f"{news['shares']['neutral']:.0%}")
            st.caption(f"FinBERT analyzed {news['count']} headlines, not full articles. Classification is not a probability of stock returns.")
        else:
            st.info(news_error or "No recent headlines available.")
        st.metric("Annualized historical volatility", f"{info['volatility']:.1f}%")

    with discover:
        st.subheader("Watchlist intelligence")
        st.markdown("<div class='mi-panel'><h4>Your market radar</h4><p>Compare recent prices, price-based zones, and technical momentum. Search any ticker above to open its full analysis.</p></div>", unsafe_allow_html=True)
        rows = []
        for stock in WATCHLIST:
            try:
                data = prices(stock)
                result = analyze(data)
                if result is None:
                    continue
                z = price_levels(data)
                p = float(result["price"])
                s = classify_price(p, z)[0]
                rows.append({"Stock":stock,"Daily close":p,"1M change":float(result["month_change"]),"Price zone":s,"Trend":research_signal(result,None)[0],"Buy range":zone_range(z["buy_low"],z["buy_high"]) if z else "—","Sell range":zone_range(z["sell_low"],z["sell_high"]) if z else "—"})
            except Exception:
                continue
        if rows:
            table = pd.DataFrame(rows).sort_values("1M change", ascending=False)
            def cell_color(value):
                colors = {"BUY ZONE":GREEN,"SELL ZONE":RED,"HOLD / WAIT":AMBER,"BREAKDOWN":RED,"BREAKOUT":BLUE,"BULLISH":GREEN,"BEARISH":RED,"MIXED":AMBER}
                return f"color:{colors.get(value, '#dce6f3')};font-weight:800"
            styled = table.style.map(cell_color, subset=["Price zone","Trend"]).format({"Daily close":"${:,.2f}","1M change":"{:+.2f}%"})
            st.dataframe(styled, hide_index=True, use_container_width=True)
            st.download_button("↓ Export watchlist", table.to_csv(index=False), "watchlist.csv", "text/csv")
            st.caption("Watchlist uses last daily close and technical-only trend; the Home tab can use a more recent intraday quote and FinBERT headlines.")
        st.subheader("Compare any two stocks")
        a, b = st.columns(2)
        first = a.text_input("Stock A", value=WATCHLIST[0], key="compare_a").strip().upper()
        second = b.text_input("Stock B", value=WATCHLIST[min(1,len(WATCHLIST)-1)], key="compare_b").strip().upper()
        if first == second:
            st.info("Choose two different symbols.")
        elif VALID.fullmatch(first) and VALID.fullmatch(second):
            try:
                x = prices(first)["Close"].rename(first)
                y = prices(second)["Close"].rename(second)
                combined = pd.concat([x,y], axis=1).dropna().tail(126)
                if len(combined) > 1:
                    st.plotly_chart(comparison_chart(combined.div(combined.iloc[0])*100), use_container_width=True, config={"displayModeBar":False})
                    st.caption("Both stocks start at 100 so you can compare relative performance.")
                else:
                    st.info("Insufficient overlapping history.")
            except Exception:
                st.info("Comparison unavailable. Check both ticker symbols.")
        else:
            st.info("Enter valid Yahoo Finance ticker symbols.")

    with research:
        st.subheader(f"{ticker} · Research lab")
        st.markdown("<div class='mi-panel'><h4>Under the hood</h4><p>Inspect the assumptions behind the price zones, technical trend, and financial headline analysis.</p></div>", unsafe_allow_html=True)
        st.subheader("Technical indicator breakdown")
        labels = ["Price above 20-day average","Price above 50-day average","Positive monthly momentum","RSI between 40 and 65"]
        for label, passed in zip(labels, info["checks"]):
            if passed:
                st.success(f"✓ {label}")
            else:
                st.warning(f"— {label} not met")
        if levels:
            st.subheader("Price level methodology")
            a, b, c = st.columns(3)
            a.metric("20-day average", money(levels["ma20"]))
            b.metric("50-day average", money(levels["ma50"]))
            c.metric("14-day ATR", money(levels["atr"]))
            st.markdown(f"""<div class='mi-panel'><h4>Fixed historical reference bands</h4><p>
            <b>Buy:</b> 60-session low ({money(levels['support'])}) plus a volatility buffer.<br>
            <b>Sell:</b> 60-session high ({money(levels['resistance'])}) minus a volatility buffer.<br>
            <b>Hold / wait:</b> the price interval between those bands.<br>
            <b>Breakdown:</b> price below {money(levels['risk'])}; <b>breakout:</b> price above {money(levels['resistance'])}.<br>
            These reference levels are recalculated as historical data updates. They do not establish intrinsic value or whether you personally should trade.</p></div>""", unsafe_allow_html=True)
        st.subheader("AI-analyzed financial headlines")
        lookup = {item["title"]:item for item in news["items"]} if news else {}
        if not articles:
            st.info("No recent headlines available.")
        for article in articles[:10]:
            title = article.get("title", "Untitled")
            with st.container(border=True):
                st.markdown(f"**{title}**")
                item = lookup.get(title)
                if item:
                    color = {"positive":"🟢","negative":"🔴","neutral":"🟡"}.get(item["label"], "⚪")
                    st.caption(f"{color} {item['label'].title()} · {item['confidence']:.0%} classification confidence")
                if article.get("published"):
                    st.caption(article["published"])
                if article.get("url"):
                    st.link_button("Read source", article["url"])
        st.caption("FinBERT analyzes headline wording only. News sources can be duplicated or irrelevant to the ticker.")
        st.subheader("Scenario simulator")
        amount = st.number_input("Hypothetical investment ($)", min_value=0.0, value=500.0, step=50.0)
        move = st.slider("Hypothetical price change (%)", -80, 150, 10)
        future = amount*(1+move/100)
        a, b = st.columns(2)
        a.metric("Hypothetical future value", f"${future:,.2f}")
        b.metric("Hypothetical gain / loss", f"${future-amount:+,.2f}")
        st.caption("Scenario math only. Excludes taxes, fees and dividends.")
        st.subheader("Research journal")
        with st.form("journal_form"):
            stance = st.selectbox("Research stance", ["Research buying","Watch","Review holding","Research selling"])
            thesis = st.text_area("Your notes")
            save = st.form_submit_button("Save research note")
        if save:
            if not thesis.strip():
                st.warning("Write a note before saving.")
            else:
                try:
                    with journal_db() as conn:
                        conn.execute("INSERT INTO journal (created,ticker,stance,price,thesis) VALUES (?,?,?,?,?)", (now_central().isoformat(),ticker,stance,current,thesis.strip()))
                    st.success("Research note saved.")
                except Exception as exc:
                    st.error(f"Could not save note: {exc}")
        try:
            with journal_db() as conn:
                notes = pd.read_sql_query("SELECT created,ticker,stance,price,thesis FROM journal ORDER BY id DESC LIMIT 100", conn)
            if not notes.empty:
                st.dataframe(notes, hide_index=True, use_container_width=True)
                st.download_button("↓ Export journal", notes.to_csv(index=False), "research_journal.csv", "text/csv")
        except Exception:
            st.info("Journal unavailable.")
        st.caption("Journal entries stored on free Streamlit hosting may reset after app restarts. Export regularly.")

dashboard(symbol)
with st.expander("Automated email alerts"):
    st.write("Your GitHub Actions email alerts are separate and unchanged.")
st.divider()
st.caption("Market Intelligence · Historical technical levels, not personalized trading advice · No automatic trading")
