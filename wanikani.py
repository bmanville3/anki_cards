import sys
from zoneinfo import ZoneInfo
import requests
from datetime import datetime, timedelta, timezone

# ── Config ────────────────────────────────────────────────────────────────────

API_TOKEN = "2a7edc87-a177-4a3b-b0fc-00262d74d1fe"
# Any gap longer than this between reviews is capped at this value (seconds).
# Prevents a bathroom break / distraction from inflating your total.
MAX_GAP_SECONDS = 60
MAX_TIME_FOR_ONE_OFF = 10  # seconds added for a session with only a single review

LOCAL_TZ = ZoneInfo("America/New_York")

# ── Helpers ───────────────────────────────────────────────────────────────────

BASE_URL = "https://api.wanikani.com/v2"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Wanikani-Revision": "20170710",
}


def fetch_all_pages(url: str, params: dict) -> list:
    """Fetch all pages of a paginated WaniKani collection."""
    results = []
    next_url = url

    while next_url:
        resp = requests.get(next_url, headers=HEADERS, params=params)
        resp.raise_for_status()
        body = resp.json()

        if "error" in body:
            print(f"API error: {body['error']} (code {body.get('code')})")
            sys.exit(1)

        results.extend(body.get("data", []))

        next_url = body.get("pages", {}).get("next_url")
        params = {}  # next_url already has params baked in

    return results


def parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if API_TOKEN == "YOUR_API_TOKEN_HERE":
        print("Set your API token via the WANIKANI_TOKEN env var or edit the script.")
        sys.exit(1)

    # Yesterday: midnight → midnight UTC
    now = datetime.now(LOCAL_TZ)
    yesterday_start = (now - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    yesterday_end = yesterday_start + timedelta(days=1)

    print(f"Fetching assignments updated on {yesterday_start.date()} UTC …\n")

    assignments = fetch_all_pages(
        f"{BASE_URL}/assignments",
        {"updated_after": iso(yesterday_start)},
    )

    # Keep only those updated strictly within yesterday
    reviewed = [
        a for a in assignments
        if a.get("data_updated_at")
        and yesterday_start <= parse_ts(a["data_updated_at"]) < yesterday_end
    ]

    if not reviewed:
        print("No assignments were updated yesterday — looks like a rest day! 🎉")
        return

    # Build sessions first, then derive total from them so everything is consistent
    SESSION_BREAK = timedelta(minutes=5)
    sessions = []
    timestamps = sorted(parse_ts(a["data_updated_at"]) for a in reviewed)
    session_start = timestamps[0]
    session_end   = timestamps[0]
    session_secs  = 0
    session_count = 1

    for prev, curr in zip(timestamps, timestamps[1:]):
        gap = curr - prev
        if gap > SESSION_BREAK:
            tail = MAX_TIME_FOR_ONE_OFF if session_count == 1 else MAX_GAP_SECONDS
            sessions.append((session_start, session_end, session_secs + tail))
            session_start = curr
            session_secs  = 0
            session_count = 1
        else:
            session_secs += min(gap.total_seconds(), MAX_GAP_SECONDS)
            session_count += 1
        session_end = curr

    tail = MAX_TIME_FOR_ONE_OFF if session_count == 1 else MAX_GAP_SECONDS
    sessions.append((session_start, session_end, session_secs + tail))

    total_seconds = sum(secs for _, _, secs in sessions)

    # ── Type breakdown ────────────────────────────────────────────────────────

    type_counts: dict[str, int] = {}
    for a in reviewed:
        stype = a["data"].get("subject_type", "unknown")
        type_counts[stype] = type_counts.get(stype, 0) + 1

    # ── Output ────────────────────────────────────────────────────────────────

    tz_abbr = now.strftime("%Z")  # "EST" or "EDT"

    print(f"📅  Date:            {yesterday_start.date()} ({tz_abbr})")
    print(f"✅  Items reviewed:  {len(reviewed)}")
    print()
    print("  By type:")
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {stype:<20} {count}")
    print()
    print(f"⏱️  Estimated time:  ~{fmt(int(total_seconds))}")
    print(f"   (gaps between reviews capped at {MAX_GAP_SECONDS}s each)")
    print()
    print(f"🗓️  Sessions ({len(sessions)}):")
    for i, (start, end, secs) in enumerate(sessions, 1):
        local_start = start.astimezone(LOCAL_TZ)
        local_end   = end.astimezone(LOCAL_TZ)
        print(
            f"   {i}. {local_start.strftime('%H:%M')} – {local_end.strftime('%H:%M')} {tz_abbr}"
            f"  · {fmt(int(secs))}"
        )


if __name__ == "__main__":
    main()