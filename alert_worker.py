
import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from market_core import (
    analyze,
    get_news,
    get_prices,
    now_central,
)

ROOT = Path(__file__).resolve().parent

CONFIG = json.loads(
    (ROOT / "alert_config.json").read_text()
)

STATE_FILE = ROOT / "alert_state.json"


def load_state():
    if STATE_FILE.exists():
        state = json.loads(
            STATE_FILE.read_text()
        )
    else:
        state = {}

    for key in ("signals", "conditions", "seen_news"):
        state.setdefault(key, {})

    state.setdefault("pending", [])

    return state


def save_state(state):
    temp = ROOT / "alert_state.tmp"

    temp.write_text(
        json.dumps(
            state,
            indent=2,
            sort_keys=True,
        )
    )

    temp.replace(STATE_FILE)


def send_email(subject, body):
    message = EmailMessage()

    message["From"] = os.environ["ALERT_EMAIL_FROM"]
    message["To"] = os.environ["ALERT_EMAIL_TO"]
    message["Subject"] = subject

    message.set_content(body)

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465,
        context=ssl.create_default_context(),
        timeout=30,
    ) as smtp:
        smtp.login(
            os.environ["ALERT_EMAIL_FROM"],
            os.environ["ALERT_EMAIL_PASSWORD"],
        )

        smtp.send_message(message)


def check_events(state):
    events = []
    successful = 0

    for symbol in CONFIG["watchlist"]:
        symbol = symbol.strip().upper()

        try:
            info = analyze(
                get_prices(symbol, "1y", "1d")
            )

            if info is None:
                print(
                    f"{symbol}: insufficient price history"
                )
                continue

            successful += 1

            old = state["signals"].get(symbol)

            # Technical trend change alerts
            if (
                CONFIG.get("alert_on_signal_change")
                and old is not None
                and old != info["signal"]
            ):
                events.append(
                    f"{symbol} — Trend change: "
                    f"{old} → {info['signal']}\n"
                    f"Daily bar close: "
                    f"${info['price']:,.2f}\n"
                    f"Positive indicators: "
                    f"{info['score']}/4"
                )

            state["signals"][symbol] = info["signal"]

            # Price threshold alerts
            for direction, field in [
                ("above", "price_above"),
                ("below", "price_below"),
            ]:
                target = CONFIG.get(
                    field, {}
                ).get(symbol)

                if target is None:
                    continue

                target = float(target)

                if direction == "above":
                    triggered = info["price"] >= target
                else:
                    triggered = info["price"] <= target

                key = (
                    f"{symbol}:{direction}:{target}"
                )

                previous = state[
                    "conditions"
                ].get(key)

                if previous is False and triggered:
                    events.append(
                        f"{symbol} — Price crossed "
                        f"{direction} ${target:,.2f}\n"
                        f"Latest daily bar close: "
                        f"${info['price']:,.2f}"
                    )

                state["conditions"][key] = triggered

        except Exception as exc:
            print(
                f"{symbol}: price check failed: {exc}"
            )
            continue

        # Financial news alerts
        if CONFIG.get("alert_on_news"):
            try:
                articles = get_news(symbol)

                old_ids = state[
                    "seen_news"
                ].get(symbol)

                first_news_check = old_ids is None
                seen = set(old_ids or [])

                keywords = [
                    word.lower()
                    for word in CONFIG.get(
                        "news_keywords", []
                    )
                ]

                for article in articles:
                    if article["id"] in seen:
                        continue

                    matches = any(
                        word in article["title"].lower()
                        for word in keywords
                    )

                    if not first_news_check and matches:
                        events.append(
                            f"{symbol} — New headline\n"
                            f"{article['title']}\n"
                            f"Tone: {article['tone']}\n"
                            f"{article['url']}"
                        )

                    seen.add(article["id"])

                # Preserve insertion order for history
                state["seen_news"][symbol] = list(
                    dict.fromkeys(
                        (old_ids or [])
                        + [
                            article["id"]
                            for article in articles
                        ]
                    )
                )[-500:]

            except Exception as exc:
                print(
                    f"{symbol}: news check failed: {exc}"
                )

    return events, successful


def main():
    # Manual test from GitHub Actions
    if (
        os.getenv("SEND_TEST_EMAIL", "").lower()
        == "true"
    ):
        send_email(
            "Market Intelligence — Test Successful",
            "Your stock alert email is configured.\n"
            + now_central().strftime(
                "%b %d, %Y · %I:%M %p %Z"
            ),
        )

        print(
            "Test email accepted by SMTP server."
        )
        return

    state = load_state()

    events, successful = check_events(state)

    if not successful:
        raise RuntimeError(
            "All stock price checks failed; "
            "alert state was not changed."
        )

    state["pending"].extend(events)
    save_state(state)

    if not state["pending"]:
        print("No new alerts.")
        return

    limit = max(
        1,
        int(CONFIG.get(
            "max_events_per_email", 15
        )),
    )

    selected = state["pending"][:limit]

    body = (
        "MARKET INTELLIGENCE\n"
        + now_central().strftime(
            "%b %d, %Y · %I:%M %p %Z"
        )
        + "\n\n"
        + (
            "\n\n"
            + "-" * 36
            + "\n\n"
        ).join(selected)
        + "\n\nAutomated research cues only. "
          "Verify before trading."
    )

    send_email(
        f"Market Intelligence — "
        f"{len(selected)} New Alerts",
        body,
    )

    state["pending"] = state["pending"][limit:]
    save_state(state)

    print(
        f"Sent {len(selected)} events."
    )


if __name__ == "__main__":
    main()
