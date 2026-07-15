import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import feedparser
import requests
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

DATA_DIR = Path("/app/data") if Path("/app").exists() else Path("data")
SENT_ARTICLES_FILE = DATA_DIR / "sent_articles.json"
TELEGRAM_API_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"
SUMMARY_SYSTEM_PROMPT = (
    "You summarize news articles for a Telegram audience. "
    "Always respond in English with exactly 3 concise bullet points."
)


@dataclass
class Settings:
    telegram_bot_token: str
    telegram_chat_id: str
    rss_feed_urls: List[str]
    stock_tickers: List[str]
    lm_studio_base_url: str
    lm_studio_model: str
    lm_studio_api_key: str
    check_interval_seconds: int
    yfinance_timeout_seconds: int
    http_timeout_seconds: int


def load_settings() -> Settings:
    load_dotenv()

    rss_feed_urls = [url.strip() for url in os.getenv("RSS_FEED_URLS", "").split(",") if url.strip()]
    stock_tickers = [ticker.strip() for ticker in os.getenv("STOCK_TICKERS", "").split(",") if ticker.strip()]

    return Settings(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        rss_feed_urls=rss_feed_urls,
        stock_tickers=stock_tickers,
        lm_studio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://host.docker.internal:1234/v1").strip(),
        lm_studio_model=os.getenv("LM_STUDIO_MODEL", "local-model").strip(),
        lm_studio_api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio").strip(),
        check_interval_seconds=int(os.getenv("CHECK_INTERVAL_SECONDS", "900")),
        yfinance_timeout_seconds=int(os.getenv("YFINANCE_TIMEOUT_SECONDS", "10")),
        http_timeout_seconds=int(os.getenv("HTTP_TIMEOUT_SECONDS", "15")),
    )


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SENT_ARTICLES_FILE.exists():
        SENT_ARTICLES_FILE.write_text("[]", encoding="utf-8")


def load_sent_article_ids() -> Set[str]:
    try:
        raw_data = SENT_ARTICLES_FILE.read_text(encoding="utf-8")
        article_ids = json.loads(raw_data)
        if isinstance(article_ids, list):
            return {str(article_id) for article_id in article_ids}
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        logging.warning("Could not load sent article IDs: %s", error)
    return set()


def save_sent_article_ids(article_ids: Set[str]) -> None:
    SENT_ARTICLES_FILE.write_text(json.dumps(sorted(article_ids), indent=2), encoding="utf-8")


def article_identifier(entry: Dict) -> str:
    raw_id = entry.get("id") or entry.get("guid") or entry.get("link") or entry.get("title") or ""
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def fetch_new_articles(feed_urls: List[str], sent_article_ids: Set[str]) -> List[Dict[str, str]]:
    new_articles: List[Dict[str, str]] = []

    for feed_url in feed_urls:
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                logging.warning("RSS feed parsing issue for %s: %s", feed_url, feed.bozo_exception)

            for entry in feed.entries:
                article_id = article_identifier(entry)
                if article_id in sent_article_ids:
                    continue

                title = entry.get("title", "Untitled article").strip()
                link = entry.get("link", "").strip()
                if not link:
                    logging.warning("Skipping article with missing link: %s", title)
                    continue

                source_text = (entry.get("summary") or entry.get("description") or "").strip()
                new_articles.append(
                    {
                        "id": article_id,
                        "title": title,
                        "link": link,
                        "source_text": source_text,
                    }
                )
        except Exception as error:  # noqa: BLE001
            logging.error("Failed to fetch RSS feed %s: %s", feed_url, error)

    return new_articles


def summarize_article(client: OpenAI, settings: Settings, article: Dict[str, str]) -> str:
    user_prompt = (
        f"Title: {article['title']}\n"
        f"Article excerpt: {article.get('source_text', '')}\n"
        f"Link: {article['link']}"
    )

    try:
        completion = client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout=settings.http_timeout_seconds,
        )

        content = completion.choices[0].message.content
        if not content:
            raise ValueError("LM Studio returned an empty summary")
        return content.strip()
    except Exception as error:  # noqa: BLE001
        logging.error("LM Studio summary failed for '%s': %s", article['title'], error)
        return "• Summary unavailable because LM Studio is unreachable.\n• Check LM Studio server status.\n• Original article link is provided below."


def send_telegram_message(settings: Settings, text: str) -> bool:
    endpoint = TELEGRAM_API_URL_TEMPLATE.format(token=settings.telegram_bot_token)

    try:
        response = requests.post(
            endpoint,
            json={
                "chat_id": settings.telegram_chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=settings.http_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            logging.error("Telegram API returned failure: %s", payload)
            return False
        return True
    except requests.RequestException as error:
        logging.error("Telegram message failed: %s", error)
        return False
    except ValueError as error:
        logging.error("Telegram response parsing failed: %s", error)
        return False


def build_news_message(article: Dict[str, str], summary: str) -> str:
    return (
        f"📰 {article['title']}\n\n"
        f"Summary:\n{summary}\n\n"
        f"Link: {article['link']}"
    )


def fetch_single_ticker_digest(ticker_symbol: str) -> str:
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="2d", interval="1d", auto_adjust=False)

    if history.empty:
        raise ValueError("No market data returned")

    current_price = float(history["Close"].iloc[-1])
    previous_close = float(history["Close"].iloc[-2]) if len(history) > 1 else current_price

    if previous_close == 0:
        change_percentage = 0.0
    else:
        change_percentage = ((current_price - previous_close) / previous_close) * 100

    indicator = "🟢" if change_percentage >= 0 else "🔴"
    return f"{indicator} {ticker_symbol}: {current_price:.2f} ({change_percentage:+.2f}%)"


def generate_market_digest(tickers: List[str], timeout_seconds: int) -> str:
    lines = ["📈 Market Digest"]

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(tickers)))) as executor:
        future_map = {executor.submit(fetch_single_ticker_digest, ticker): ticker for ticker in tickers}

        for future, ticker in [(future, future_map[future]) for future in future_map]:
            try:
                lines.append(future.result(timeout=timeout_seconds))
            except TimeoutError:
                lines.append(f"⚠️ {ticker}: data request timed out")
                logging.error("yfinance timeout for ticker %s", ticker)
            except Exception as error:  # noqa: BLE001
                lines.append(f"⚠️ {ticker}: data unavailable")
                logging.error("yfinance request failed for ticker %s: %s", ticker, error)

    return "\n".join(lines)


def process_news_cycle(settings: Settings, client: OpenAI, sent_article_ids: Set[str]) -> None:
    if not settings.rss_feed_urls:
        logging.warning("RSS_FEED_URLS is empty. Skipping news cycle.")
        return

    new_articles = fetch_new_articles(settings.rss_feed_urls, sent_article_ids)
    if not new_articles:
        logging.info("No new RSS articles found.")
        return

    for article in new_articles:
        summary = summarize_article(client, settings, article)
        message = build_news_message(article, summary)

        if send_telegram_message(settings, message):
            sent_article_ids.add(article["id"])
            save_sent_article_ids(sent_article_ids)


def process_market_cycle(settings: Settings) -> None:
    if not settings.stock_tickers:
        logging.warning("STOCK_TICKERS is empty. Skipping market cycle.")
        return

    digest = generate_market_digest(settings.stock_tickers, settings.yfinance_timeout_seconds)
    send_telegram_message(settings, digest)


def validate_required_settings(settings: Settings) -> bool:
    missing = []
    if not settings.telegram_bot_token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not settings.telegram_chat_id:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        logging.error("Missing required environment variables: %s", ", ".join(missing))
        return False
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    settings = load_settings()
    if not validate_required_settings(settings):
        return

    ensure_storage()
    sent_article_ids = load_sent_article_ids()

    client = OpenAI(base_url=settings.lm_studio_base_url, api_key=settings.lm_studio_api_key)

    while True:
        process_news_cycle(settings, client, sent_article_ids)
        process_market_cycle(settings)
        logging.info("Cycle completed. Sleeping for %s seconds.", settings.check_interval_seconds)
        time.sleep(settings.check_interval_seconds)


if __name__ == "__main__":
    main()
