"""
Steam Reviews Collector
=======================
AML Final Project - Dataset Collection
Collects game reviews from the Steam public API (no API key required).

Output columns:
  - review_text        : the full review body (main document for NLP)
  - sentiment          : classification target (positive / negative)
  - voted_up           : original boolean from Steam (True = recommended)
  - votes_helpful      : how many users found the review helpful
  - votes_funny        : how many users found the review funny
  - playtime_forever   : total playtime of the reviewer (in hours)
  - playtime_at_review : playtime at the time of writing the review (in hours)
  - review_date        : date the review was written (ISO format)
  - language           : language of the review
  - game_name          : name of the game being reviewed
  - game_id            : Steam app ID of the game
  - game_genre         : genre(s) of the game (fetched from store page)

Usage:
    pip install requests pandas
    python steam_reviews_collector.py

    Adjust GAMES and REVIEWS_PER_GAME to control dataset size.
"""

import requests
import pandas as pd
import time
import random
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

# Popular games across different genres for diversity.
# Format: {"name": "...", "id": <steam_app_id>}
# You can find a game's app ID in its Steam store URL:
# https://store.steampowered.com/app/570/Dota_2/  →  app ID = 570
GAMES = [
    # Action
    {"name": "Counter-Strike 2",          "id": 730},
    {"name": "DOOM Eternal",              "id": 782330},
    {"name": "Sekiro",                    "id": 814380},
    {"name": "Elden Ring",                "id": 1245620},
    {"name": "The Witcher 3",             "id": 292030},
    {"name": "Cyberpunk 2077",            "id": 1091500},
    {"name": "Sid Meier's Civilization VI","id": 289070},
    {"name": "Total War Warhammer III",   "id": 1142710},
    {"name": "Stardew Valley",            "id": 413150},
    {"name": "Cities Skylines",           "id": 255710},
    {"name": "Hollow Knight",             "id": 367520},
    {"name": "Hades",                     "id": 1145360},
    {"name": "Dota 2",                    "id": 570},
    {"name": "Team Fortress 2",           "id": 440},
    {"name": "Resident Evil 4 Remake",    "id": 2050650},
    {"name": "Disco Elysium",             "id": 632470},
    {"name": "Marvel Rivals",             "id": 2767030},
    {"name": "PRAGMATA",                  "id": 3357650},
    {"name": "R.E.P.O.",                  "id": 3241660},
    {"name": "Black Myth: Wukong",        "id": 2358720},
]

REVIEWS_PER_GAME    = 500    # reviews to collect per game
                             # increase for a bigger dataset (max ~3000-5000/game)
LANGUAGE            = "english"   # "all" to get reviews in all languages
DELAY_BETWEEN_CALLS = (0.5, 1.5)  # seconds between API requests (Steam is generous)
OUTPUT_FILE         = "steam_reviews.csv"

# ─── Sentiment label ──────────────────────────────────────────────────────────

def get_sentiment(voted_up: bool) -> str:
    """
    Steam uses a binary recommendation: Recommended / Not Recommended.
    This becomes our classification target.
    """
    return "positive" if voted_up else "negative"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def polite_get(url: str, params: dict = None) -> dict | None:
    """GET with delay and error handling. Returns parsed JSON or None."""
    time.sleep(random.uniform(*DELAY_BETWEEN_CALLS))
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.warning(f"Request failed: {e}")
        return None


def get_game_genre(app_id: int) -> str:
    """
    Fetches genre tags for a game from the Steam Store API.
    Returns a comma-separated string of genres (e.g. 'Action,RPG,Open World').
    """
    url = f"https://store.steampowered.com/api/appdetails"
    data = polite_get(url, params={"appids": app_id, "filters": "genres"})

    if not data or not data.get(str(app_id), {}).get("success"):
        return "Unknown"

    genres = data[str(app_id)].get("data", {}).get("genres", [])
    return ",".join(g["description"] for g in genres) if genres else "Unknown"

# ─── Core scraping function ───────────────────────────────────────────────────

def get_reviews_for_game(game: dict, max_reviews: int) -> list[dict]:
    """
    Calls the Steam GetReviews API in a cursor-based pagination loop.
    Docs: https://partner.steamgames.com/doc/store/getreviews

    Returns a list of review dicts.
    """
    reviews   = []
    cursor    = "*"           # Steam uses cursor-based pagination
    app_id    = game["id"]
    game_name = game["name"]

    log.info(f"  Fetching genre for '{game_name}'...")
    genre = get_game_genre(app_id)

    url = f"https://store.steampowered.com/appreviews/{app_id}"

    while len(reviews) < max_reviews:
        params = {
            "json":          1,
            "language":      LANGUAGE,
            "filter":        "recent",      # 'recent' | 'updated' | 'all'
            "review_type":   "all",         # 'all' | 'positive' | 'negative'
            "purchase_type": "all",         # 'all' | 'steam' | 'non_steam_purchase'
            "num_per_page":  100,           # max allowed by Steam API
            "cursor":        cursor,
        }

        data = polite_get(url, params=params)

        if not data or data.get("success") != 1:
            log.warning(f"  API error for '{game_name}', stopping.")
            break

        batch = data.get("reviews", [])
        if not batch:
            log.info(f"  No more reviews available for '{game_name}'.")
            break

        for r in batch:
            review_text = r.get("review", "").strip()
            if not review_text:
                continue  # skip empty reviews

            voted_up          = r.get("voted_up", False)
            votes_helpful     = r.get("votes_helpful", 0)
            votes_funny       = r.get("votes_funny", 0)

            # Playtime is given in minutes — convert to hours (rounded to 1 decimal)
            playtime_forever  = round(r.get("author", {}).get("playtime_forever", 0) / 60, 1)
            playtime_review   = round(r.get("author", {}).get("playtime_at_review", 0) / 60, 1)

            # Timestamp → ISO date string
            ts = r.get("timestamp_created", 0)
            review_date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""

            language = r.get("language", "")

            reviews.append({
                "review_text":          review_text,
                "sentiment":            get_sentiment(voted_up),
                "voted_up":             voted_up,
                "votes_helpful":        votes_helpful,
                "votes_funny":          votes_funny,
                "playtime_forever_hrs": playtime_forever,
                "playtime_at_review_hrs": playtime_review,
                "review_date":          review_date,
                "language":             language,
                "game_name":            game_name,
                "game_id":              app_id,
                "game_genre":           genre,
            })

            if len(reviews) >= max_reviews:
                break

        # Update cursor for next page
        new_cursor = data.get("cursor", "")
        if not new_cursor or new_cursor == cursor:
            break  # no more pages
        cursor = new_cursor

        log.info(f"    {len(reviews)}/{max_reviews} reviews collected for '{game_name}'")

    return reviews

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_reviews = []

    for i, game in enumerate(GAMES, 1):
        log.info(f"[{i}/{len(GAMES)}] === Game: {game['name']} (id={game['id']}) ===")
        reviews = get_reviews_for_game(game, REVIEWS_PER_GAME)
        all_reviews.extend(reviews)
        log.info(f"  ✓ {len(reviews)} reviews. Total so far: {len(all_reviews)}")

    # ── Save ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_reviews)

    # Remove duplicates (same review text for same game)
    df.drop_duplicates(subset=["review_text", "game_id"], inplace=True)

    # Remove very short reviews (less than 10 words) — low quality for NLP
    df = df[df["review_text"].str.split().str.len() >= 10]

    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info(f"\n✅ Done! Saved {len(df)} reviews to '{OUTPUT_FILE}'")
    print("\n── Dataset summary ──────────────────────────────────────────")
    print(f"Total reviews      : {len(df)}")
    print(f"Unique games       : {df['game_name'].nunique()}")
    print(f"Date range         : {df['review_date'].min()}  →  {df['review_date'].max()}")
    print(f"\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    print(f"\nReviews per game:")
    print(df["game_name"].value_counts())
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
