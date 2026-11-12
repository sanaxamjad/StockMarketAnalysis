
import hashlib
import html
import re
from datetime import datetime
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CENTRAL = ZoneInfo("America/Chicago")
SENTIMENT = SentimentIntensityAnalyzer()


def now_central():
    return datetime.now(CENTRAL)


def format_central(value, daily=False):
    if value is None:
        return "Unavailable"

    stamp = pd.Timestamp(value)

    if daily:
        return stamp.strftime("%b %d, %Y (daily session)")

    if stamp.tzinfo is None:
        return stamp.strftime(
            "%b %d, %Y (timezone unknown)"
        )

    return stamp.tz_convert(CENTRAL).strftime(
        "%b %d, %Y · %I:%M %p %Z"
    )


def get_prices(symbol, period="2y", interval="1d"):
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
        prepost=False,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna(subset=["Close"]).copy()


def get_latest_bar(symbol, daily=None):
    try:
        bars = get_prices(symbol, "5d", "5m")

        if not bars.empty:
            return {
                "price": float(bars["Close"].iloc[-1]),
                "timestamp": bars.index[-1],
                "kind": "5-minute bar start",
                "daily": False,
            }
    except Exception:
        pass

    if daily is not None and not daily.empty:
        return {
            "price": float(daily["Close"].iloc[-1]),
            "timestamp": daily.index[-1],
            "kind": "Daily session date",
            "daily": True,
        }

    return None


def analyze(data):
    if data is None or len(data) < 55:
        return None

    close = data["Close"].astype(float)
    price = float(close.iloc[-1])

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    if loss.iloc[-1] == 0:
        rsi = 100.0 if gain.iloc[-1] > 0 else 50.0
    else:
        rsi = float(
            100 - 100 / (
                1 + gain.iloc[-1] / loss.iloc[-1]
            )
        )

    month_change = (
        price / float(close.iloc[-21]) - 1
    ) * 100

    checks = [
        price > float(ma20.iloc[-1]),
        price > float(ma50.iloc[-1]),
        month_change > 0,
        40 <= rsi <= 65,
    ]

    score = sum(checks)

    if score >= 3 and rsi < 70:
        signal = "RESEARCH ENTRY"
    elif score <= 1:
        signal = "REVIEW RISK"
    else:
        signal = "WATCH"

    returns = close.pct_change().dropna()

    volatility = float(
        returns.tail(20).std() * np.sqrt(252) * 100
    )

    return {
        "price": price,
        "month_change": month_change,
        "ma20": ma20,
        "ma50": ma50,
        "rsi": rsi,
        "checks": checks,
        "score": score,
        "signal": signal,
        "volatility": volatility,
    }


def get_news(symbol, limit=12):
    query = quote_plus(
        f"{symbol} stock company earnings when:2d"
    )

    url = (
        "https://news.google.com/rss/search?"
        f"q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    response.raise_for_status()

    articles = []

    entries = feedparser.parse(
        response.content
    ).entries[:limit]

    for entry in entries:
        title = html.unescape(
            re.sub(
                r"<[^>]+>",
                "",
                entry.get("title", ""),
            )
        ).strip()

        link = entry.get("link", "")

        if not title or not link.startswith("https://"):
            continue

        sentiment = SENTIMENT.polarity_scores(
            title
        )["compound"]

        if sentiment > 0.15:
            tone = "Positive"
        elif sentiment < -0.15:
            tone = "Negative"
        else:
            tone = "Neutral"

        articles.append({
            "id": hashlib.sha256(
                f"{symbol}|{link}".encode()
            ).hexdigest(),
            "title": title,
            "url": link,
            "tone": tone,
            "published": entry.get("published", ""),
        })

    return articles
