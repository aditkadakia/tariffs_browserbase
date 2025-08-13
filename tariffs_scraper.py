
import os, sys, re, csv, math, asyncio
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel

from browserbase import Browserbase
from stagehand import Stagehand, StagehandConfig

# ---- FREE sentiment (VADER) ----
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

POS_TARIFF_HINTS = ["protect","protection","protecting","fair trade","bring back jobs",
    "onshore","reshore","domestic","manufacturing","counter china","stand up to china",
    "level playing field","good","win","works"]
NEG_TARIFF_HINTS = ["inflation","prices","price hike","cost","costs","tax on consumers",
    "trade war","retaliation","tariff war","smoot","hawley","inefficient","jobs lost",
    "hurts","burden","bad","stupid"]
THEMES = {
    "inflation/prices": ["inflation","prices","price","expensive","cost","costs"],
    "china/geopolitics": ["china","beijing","ccp","xi","chinese"],
    "jobs/manufacturing": ["jobs","manufacturing","factory","factories","onshore","reshore","made in"],
    "trade war/retaliation": ["trade war","tariff war","retaliation","retaliate","tit for tat"],
    "farmers/agriculture": ["farmers","agriculture","soy","corn","wheat","ranchers"],
    "consumers/smb": ["consumers","consumer","small business","smb","main street"],
    "national security": ["national security","security","strategic","defense","supply chain"],
}

class Tweet(BaseModel):
    handle: str
    text: str

def must(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"[CONFIG] Missing {name}. Put it in .env", file=sys.stderr); sys.exit(1)
    return v

async def ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

def classify_stance(text: str, sia: SentimentIntensityAnalyzer) -> str:
    t = text.lower()
    comp = sia.polarity_scores(text)["compound"]
    if any(k in t for k in POS_TARIFF_HINTS): comp += 0.20
    if any(k in t for k in NEG_TARIFF_HINTS): comp -= 0.20
    return "positive" if comp >= 0.2 else "negative" if comp <= -0.2 else "neutral"

def top_themes(texts: List[str], k: int = 2) -> List[str]:
    counts: Dict[str,int] = {k:0 for k in THEMES}
    for tx in texts:
        low = tx.lower()
        for theme, keys in THEMES.items():
            if any(key in low for key in keys):
                counts[theme] += 1
    ranked = [t for t,_ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True) if counts[t] > 0]
    return ranked[:k] if ranked else []

async def wait_for_login(page, seconds=25):
    print("[LOGIN] Waiting for login/cookies… (use the Live viewer for THIS session if needed)")
    for _ in range(seconds):
        try:
            cookies = await page.context.cookies()
            if any(c.get("name") in ("auth_token","ct0") and "x.com" in (c.get("domain") or "") for c in cookies):
                print("✅ Detected login via cookies.")
                return True
        except Exception:
            pass
        for sel in [
            'div[data-testid="SideNav_AccountSwitcher_Button"]',
            'div[data-testid="AppTabBar_Home_Link"]',
            'a[aria-label="Home"][href*="/home"]',
            'a[aria-label="Profile"]',
            'a[aria-label="Post"]',
        ]:
            if await page.locator(sel).count() > 0:
                print("✅ Detected logged-in UI.")
                return True
        await asyncio.sleep(1)
    print("⏳ Didn’t see a clear login signal; continuing anyway…")
    return False

async def load_results(page):
    desktop = "https://x.com/search?q=%23Tariffs&src=typed_query&f=live"
    mobile  = "https://mobile.twitter.com/search?q=%23Tariffs&src=typed_query&f=live"
    await page.goto(desktop, wait_until="domcontentloaded")
    await page.wait_for_timeout(1800)
    if await page.locator("div[lang]").count() == 0:
        await page.goto(mobile, wait_until="domcontentloaded")
        await page.wait_for_timeout(1800)
    for _ in range(12):
        await page.keyboard.press("PageDown")
        await asyncio.sleep(0.7)

async def scrape_tweets(page) -> List[Tweet]:
    tweets: List[Tweet] = []
    seen = set()

    articles = page.locator("article:has(div[lang])")
    count = await articles.count()

    if count == 0:
        text_nodes = page.locator("div[lang]")
        n = min(await text_nodes.count(), 200)
        for i in range(n):
            text = (await text_nodes.nth(i).inner_text()).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) < 20: continue
            key = text[:80]
            if key in seen: continue
            seen.add(key)
            tweets.append(Tweet(handle="@unknown", text=text))
        return tweets

    for i in range(min(count, 180)):
        art = articles.nth(i)
        chunks = await art.locator("div[lang]").all_inner_texts()
        if not chunks: continue
        text = " ".join(c.strip() for c in chunks)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 20: continue

        # robust handle extraction (avoid strict-mode)
        handle = "@unknown"
        try:
            user = art.locator("div[data-testid='User-Name']").first
            if await user.count() > 0:
                raw = (await user.inner_text()).strip()
                m = re.search(r"@\w{1,15}", raw)
                if m: handle = m.group(0)
            if handle == "@unknown":
                raw_all = (await art.inner_text()).strip()
                m2 = re.search(r"@\w{1,15}", raw_all)
                if m2: handle = m2.group(0)
        except Exception:
            pass

        key = f"{handle}::{text[:80]}"
        if key in seen: continue
        seen.add(key)
        tweets.append(Tweet(handle=handle, text=text))

    return tweets

async def main():
    load_dotenv()
    await ensure_vader()
    sia = SentimentIntensityAnalyzer()

    BB_API  = must("BROWSERBASE_API_KEY")
    BB_PROJ = must("BROWSERBASE_PROJECT_ID")
    BB_CTX  = os.getenv("BB_CONTEXT_ID")  # optional; will create if missing

    bb = Browserbase(api_key=BB_API)

    if not BB_CTX:
        ctx = bb.contexts.create(project_id=BB_PROJ)
        BB_CTX = ctx.id
        print(f"[CTX] Created Context ID: {BB_CTX}  (add this to .env as BB_CONTEXT_ID)")

    sess = bb.sessions.create(
        project_id=BB_PROJ,
        browser_settings={
            "context": {"id": BB_CTX, "persist": True},
            "fingerprint": {"devices": ["mobile"], "browsers": ["chrome"], "operatingSystems": ["android"]},
            "viewport": {"width": 390, "height": 844},
        },
    )
    print("[BB] Session ID:", sess.id, "(open in Dashboard → Sessions → Live if you need to interact)")

    try:
        try:
            cfg = StagehandConfig(env="BROWSERBASE", api_key=BB_API, project_id=BB_PROJ,
                                  browserbase_session_id=sess.id)
        except TypeError:
            cfg = StagehandConfig(env="BROWSERBASE", api_key=BB_API, project_id=BB_PROJ,
                                  browserbaseSessionId=sess.id)
        sh = Stagehand(cfg)
        await sh.init()

        page = sh.page
        await page.goto("https://x.com/home", wait_until="domcontentloaded")
        await wait_for_login(page, seconds=25)

        await load_results(page)
        if await page.locator("div[lang]").count() == 0:
            raise RuntimeError("Still gated by X after login — refresh Live or ensure you’re fully logged in, then re-run.")

        tweets = await scrape_tweets(page)
        print(f"[SCRAPE] Collected {len(tweets)} tweets")

        classified, counts = [], {"positive":0,"negative":0,"neutral":0}
        for t in tweets:
            stance = classify_stance(t.text, sia)
            counts[stance] += 1
            classified.append({"handle": t.handle, "text": t.text, "stance": stance})

        total = max(1, sum(counts.values()))
        pos = math.floor(100*counts["positive"]/total)
        neg = math.floor(100*counts["negative"]/total)
        neu = 100 - pos - neg

        themes = top_themes([c["text"] for c in classified], k=2)
        skew = "negative" if counts["negative"]>max(counts["positive"],counts["neutral"]) else \
               "positive" if counts["positive"]>max(counts["negative"],counts["neutral"]) else "mixed"
        themes_str = " and ".join(themes) if themes else "mixed themes"
        summary = f"On #Tariffs, tweets skew {skew} ({neg}% negative, {pos}% positive, {neu}% neutral), with frequent mentions of {themes_str}."
        print("\n=== One-sentence summary (no LLM) ===\n" + summary)
        print("\n=== Counts ===\n", counts)

        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out = f"tariffs_tweets_{ts}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["handle","text","stance"])
            for r in classified: w.writerow([r["handle"], r["text"], r["stance"]])
        print(f"\n[FILE] Saved: {out}")

    finally:
        # Close Stagehand (this closes the Browserbase-controlled browser)
        try:
            await sh.close()
        except Exception:
            pass
        print("[BB] Session closed via Stagehand. If a Live tab is still open, hit Stop there too.")

if __name__ == "__main__":
    asyncio.run(main())
