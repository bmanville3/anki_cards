import argparse
import copy
import json
import os
import sqlite3

import requests

ANKI_CONNECT = "http://localhost:8765"

COLLECTION_PATH = os.path.expanduser(
    "/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.anki2"
)


def invoke(action, params=None):
    body = {"action": action, "version": 6}
    if params is not None:
        body["params"] = params

    response = requests.post(ANKI_CONNECT, json=body).json()

    if response["error"] is not None:
        raise RuntimeError(f"{action}: {response['error']}")

    return response["result"]


###############################################################################
# Helpers
###############################################################################

def get_all_decks():
    return invoke("deckNames")


def get_deck_config(deck):
    return invoke("getDeckConfig", {"deck": deck})


###############################################################################
# Cleanup: delete configs used by 0 decks
###############################################################################

def get_all_config_ids_from_db():
    uri = f"file:{COLLECTION_PATH}?immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}

        if "deck_config" in tables:
            cur.execute("SELECT id, name FROM deck_config")
            return {row[0]: row[1] for row in cur.fetchall()}

        # legacy fallback
        cur.execute("SELECT dconf FROM col")
        dconf = json.loads(cur.fetchone()[0])
        return {int(cid): cfg["name"] for cid, cfg in dconf.items()}
    finally:
        conn.close()


def cleanup_unused_configs(dry_run):
    all_configs = get_all_config_ids_from_db()

    used_ids = set()
    for deck in get_all_decks():
        used_ids.add(get_deck_config(deck)["id"])

    orphaned = {
        cid: name for cid, name in all_configs.items()
        if cid not in used_ids and cid != 1  # never touch "Default"
    }

    if not orphaned:
        print("No unused configs found.")
        return

    for cid, name in orphaned.items():
        if name == "Default":
            print(f"Skipping '{name}' deck...")
            continue
        if dry_run:
            print(f"[dry run] would delete unused config: {name} (id={cid})")
        else:
            print(f"Deleting unused config: {name} (id={cid})")
            invoke("removeDeckConfigId", {"configId": cid})


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without changing anything in Anki.",
    )
    args = parser.parse_args()
    dry_run = args.dry_run

    if dry_run:
        print("=== DRY RUN: no changes will be made ===")
    cleanup_unused_configs(dry_run)


if __name__ == "__main__":
    main()