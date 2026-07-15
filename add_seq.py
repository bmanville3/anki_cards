import json
from typing import Any
import urllib.request

from structure_all_cards import Card

ANKICONNECT = "http://localhost:8765"
SEQ_LENGTH = 13
OVERWRITE_SEQ_EXISTING = False

MASTER = "Master-"


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

def update_note(note_id: str, note_fields: dict[str, Any]):
     ac("updateNoteFields", note={"id": note_id, "fields": note_fields})

def find_decks() -> list[str]:
    return ac("deckNames")



def add_sequence_to_core2000() -> None:
    def _make_seq(raw_index: str, suffix: int, width: int = SEQ_LENGTH) -> str:
        raw_index = raw_index.strip()
        if not raw_index.isdigit():
            print(f"WARNING: non-numeric index value {raw_index!r}")
            return ""
        combined = int(raw_index) * 100 + suffix
        return str(combined).zfill(width)

    all_decks = find_decks()
    for deck in all_decks:
        if "Core 2000" not in deck:
            continue
        print(f"Adding sequence to {deck}")
        note_ids = ac("findNotes", query=f'deck:"{deck}"')
        print(f"Found {len(note_ids)} notes")
        notes = ac("notesInfo", notes=note_ids)

        for note in notes:
            id_ = note["noteId"]
            fields = note["fields"]
            if not OVERWRITE_SEQ_EXISTING and "Sequence" in fields and fields["Sequence"]["value"]:
                continue
            if note["modelName"] == "InfoNote":
                continue

            tags = note.get("tags", [])
            if "type::vocab" in tags:
                idx_field, suffix = "Optimized-Voc-Index", "0"
            elif "type::sentence" in tags:
                idx_field, suffix = "Optimized-Sent-Index", "1"
            else:
                print(f"{id_}: no type tag found ({tags}), skipping")
                continue

            previous_version = fields["Previous Version"]["value"]
            if not previous_version:
                print(f"{id_} has no previous version")
                continue
            try:
                best_attempt = Card.from_pretty_string(previous_version, "skip-verifier")
                if int(id_) != int(best_attempt.noteId):
                    raise ValueError(f"Mismatched id {id_} != {best_attempt.noteId}")
                raw_value = best_attempt.get_field(idx_field)
                seq = _make_seq(raw_value, suffix=0 if idx_field == "Optimized-Voc-Index" else 1)
                update_note(id_, {"Sequence": seq})
                print(f"Updated {id_} -> Sequence={seq} (from {idx_field})")
            except Exception as e:
                print(f"ERROR: Could not update {id_}: {e}")


def add_sequence_by_date_to_decks() -> None:
    """Sequence = creation order. Anki noteIds are epoch-ms creation
    timestamps, so sorting by noteId ascending gives creation order."""
    all_decks = find_decks()
    for deck in all_decks:
        if "Genki I" not in deck and "In the Wild" not in deck:
            continue
        print(f"Adding sequence to {deck}")
        card_ids = find_cards(f'deck:"{deck}"')
        print(f"Found {len(card_ids)} cards")
        cards = cards_info(card_ids)

        # dedupe by note, since a note can have multiple cards
        by_note = {}
        for card in cards:
            by_note.setdefault(card["note"], card)

        ordered = sorted(by_note.items(), key=lambda kv: int(kv[0]))
        for rank, (note_id, card) in enumerate(ordered, start=1):
            if not OVERWRITE_SEQ_EXISTING and "Sequence" in card["fields"] and card["fields"]["Sequence"]["value"]:
                continue
            if card["modelName"] == "InfoNote":
                continue
            seq = str(rank).zfill(SEQ_LENGTH)
            update_note(note_id, {"Sequence": seq})
            print(f"Set {note_id} sequence to {seq}")


def add_sequence_to_all_jlab() -> None:
    all_decks = find_decks()
    for deck in all_decks:
        if "Jlab's beginner course" not in deck:
            continue
        print(f"Adding sequence to {deck}")
        cards_ids = find_cards(f'deck:"{deck}"')
        print(f"Found {len(cards_ids)} cards")
        cards = cards_info(cards_ids)
        for card in cards:
            id_ = card["note"]
            if "Sequence" in card['fields'] and card['fields']["Sequence"]['value']:
                continue
            if card["modelName"] == "InfoNote":
                continue
            previous_version = card['fields']["Previous Version"]['value']
            if not previous_version:
                print(f"{id_} has not previous version")
                continue
            try:
                best_attempt = Card.from_pretty_string(previous_version, "skip-verifier")
                if int(id_) != int(best_attempt.noteId):
                    raise ValueError(f"Mismatched id {id_} != {best_attempt.noteId}")
                update_note(id_, {"Sequence": best_attempt.get_field("Sequence")})
                print(f"Updated {id_}")
            except Exception:
                print(f"ERROR: Could not update {id_}")

def add_sequence_to_all_video() -> None:
    all_decks = find_decks()
    i = 0
    deck_to_id = {}
    for deck in all_decks:
        if "Japanese Video Deck" not in deck:
            continue
        if deck not in deck_to_id:
            deck_to_id[deck] = i
            i += 1
        print(f"Adding sequence to {deck}")
        cards_ids = find_cards(f'deck:"{deck}"')
        print(f"Found {len(cards_ids)} cards")
        cards = cards_info(cards_ids)

        for card in cards:
            id_ = card["note"]
            if "Sequence" in card['fields'] and card['fields']["Sequence"]['value']:
                continue
            if card["modelName"] == "InfoNote":
                continue
            previous_version = card['fields']["Previous Version"]['value']
            if not previous_version:
                print(f"{id_} has not previous version")
                continue
            try:
                best_attempt = Card.from_pretty_string(previous_version, "skip-verifier")
                if int(id_) != int(best_attempt.noteId):
                    raise ValueError(f"Mismatched id {id_} != {best_attempt.noteId}")
                timecode = best_attempt.get_field("TimeCode")
                mm, rest = timecode.split(":")
                ss, ms = rest.split(".")
                timecode = f"{mm.zfill(2)}{ss.zfill(2)}{ms.zfill(3)}"
                deck_prefix = str(deck_to_id[deck]).zfill(2)
                timecode = deck_prefix + timecode
                seq = timecode.zfill(SEQ_LENGTH)
                update_note(id_, {"Sequence": seq})
                print(f"Set {id_} sequence to {seq}")
            except Exception:
                print(f"ERROR: Could not update {id_}")

def duplicate_notes_to_deck(source_query: str, target_deck: str) -> None:
    note_ids = ac("findNotes", query=source_query)
    print(f"Found {len(note_ids)} notes to duplicate")
    notes_info = ac("notesInfo", notes=note_ids)

    for note in notes_info:
        model_name = note["modelName"]
        fields = {k: v["value"] for k, v in note["fields"].items()}
        tags = note.get("tags", [])

        new_note = {
            "deckName": target_deck,
            "modelName": model_name,
            "fields": fields,
            "tags": tags,
            # uncoment to run this
            # "options":   {"allowDuplicate": True}
        }
        try:
            new_id = ac("addNote", note=new_note)
            print(f"Duplicated {note['noteId']} -> {new_id}")
        except RuntimeError as e:
            print(f"Failed to duplicate {note['noteId']}: {e}")


def main():
    add_sequence_to_core2000()
    add_sequence_by_date_to_decks()
    add_sequence_to_all_video()


if __name__ == "__main__":
    main()
