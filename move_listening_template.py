"""
Anki Migration Script
=====================
1. Ensures Japanese-Listening top-level deck exists
2. For every Master-Listening card, finds its sibling Master-Reading card,
   reads its deck path, swaps Japanese-Reading → Japanese-Listening,
   and moves the Listening card to the mirrored deck path.

Result: Japanese-Listening mirrors the full tree of Japanese-Reading.
Reading cards and their FSRS history are completely untouched.

Requires: AnkiConnect addon running in Anki (default port 8765)
"""

import json
import urllib.request
import sys
from datetime import datetime
from collections import defaultdict

ANKICONNECT = "http://localhost:8765"
READING_TEMPLATE  = "Master-Reading"
LISTENING_TEMPLATE = "Master-Listening"
READING_ROOT  = "Japanese-Reading"
LISTENING_ROOT = "Japanese-Listening"


# ---------------------------------------------------------------------------
# AnkiConnect helpers
# ---------------------------------------------------------------------------

def ac(action, **params):
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode()
    req = urllib.request.Request(ANKICONNECT, payload)
    resp = json.loads(urllib.request.urlopen(req).read())
    if resp.get("error"):
        raise RuntimeError(f"AnkiConnect error [{action}]: {resp['error']}")
    return resp["result"]


def find_cards(query):
    return ac("findCards", query=query)


def cards_info(card_ids):
    if not card_ids:
        return []
    return ac("cardsInfo", cards=card_ids)


def change_deck(card_ids, deck_name):
    ac("changeDeck", cards=card_ids, deck=deck_name)


def create_deck(name):
    ac("createDeck", deck=name)


# ---------------------------------------------------------------------------
# Step 1 – Ensure Japanese-Listening root exists
# ---------------------------------------------------------------------------

def ensure_root_deck():
    print("\n[1/2] Ensuring Japanese-Listening root deck exists...")
    create_deck(LISTENING_ROOT)
    print(f"    ✓ {LISTENING_ROOT} ready")


# ---------------------------------------------------------------------------
# Step 2 – Mirror Reading deck tree into Listening
# ---------------------------------------------------------------------------

def mirror_listening_cards():
    print("\n[2/2] Mirroring Listening cards into Japanese-Listening tree...")

    # Fetch all Listening cards
    listening_ids = find_cards(f'card:"{LISTENING_TEMPLATE}"')
    if not listening_ids:
        print("    ! No Listening cards found.")
        print("      Add the Master-Listening template in Anki first:")
        print("      Tools → Manage Note Types → Cards → Add Card Type")
        print("      Then re-run this script.")
        return

    print(f"    Found {len(listening_ids)} Listening cards")

    # Fetch all Reading cards so we can build a note_id → deck map
    reading_ids = find_cards(f'card:"{READING_TEMPLATE}"')
    if not reading_ids:
        print("    ! No Reading cards found — cannot determine deck paths.")
        return

    print(f"    Found {len(reading_ids)} Reading cards — building note→deck map...")

    # Build {note_id: deck_name} from Reading cards
    reading_info = cards_info(reading_ids)
    note_to_deck = {r["note"]: r["deckName"] for r in reading_info}

    # Now process Listening cards in batches
    listening_info = cards_info(listening_ids)

    # Group Listening card IDs by their target deck
    target_deck_to_cards = defaultdict(list)
    skipped = 0

    for lcard in listening_info:
        note_id = lcard["note"]
        reading_deck = note_to_deck.get(note_id)

        if not reading_deck:
            skipped += 1
            continue

        # Swap the root prefix
        if reading_deck.startswith(READING_ROOT):
            target_deck = LISTENING_ROOT + reading_deck[len(READING_ROOT):]
        else:
            # Reading card isn't under Japanese-Reading — put it at root
            target_deck = LISTENING_ROOT
            print(f"      ! Note {note_id} is in '{reading_deck}' (not under {READING_ROOT}), "
                  f"placing in {LISTENING_ROOT}")

        target_deck_to_cards[target_deck].append(lcard["cardId"])

    # Create decks and move cards
    print(f"\n    Moving cards into {len(target_deck_to_cards)} mirrored decks...")
    total_moved = 0

    for deck_name, card_ids in sorted(target_deck_to_cards.items()):
        create_deck(deck_name)
        change_deck(card_ids, deck_name)
        print(f"      ✓ {deck_name}  ({len(card_ids)} cards)")
        total_moved += len(card_ids)

    print(f"\n    ✓ Moved {total_moved} Listening cards")
    if skipped:
        print(f"    ! Skipped {skipped} cards (no matching Reading card found)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  Anki Japanese Deck Migration")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    try:
        version = ac("version")
        print(f"\nAnkiConnect version: {version}")
    except Exception:
        print("\n✗ Cannot reach AnkiConnect. Make sure Anki is open and the addon is enabled.")
        sys.exit(1)

    ensure_root_deck()
    mirror_listening_cards()

    print("\n" + "=" * 55)
    print("  Done. Japanese-Listening now mirrors")
    print("  the Japanese-Reading deck tree.")
    print("  Reading cards and FSRS history untouched.")
    print("=" * 55)


if __name__ == "__main__":
    main()
