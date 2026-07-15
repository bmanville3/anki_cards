import copy
from pprint import pprint
import requests

ANKI_CONNECT = "http://localhost:8765"


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

def get_name(deck):
    return deck.split("::")[-1]


def get_card_count(deck):
    stats = invoke("getDeckStats", {"decks": [deck]})
    # keyed by deck id, not name — find the entry whose "name" matches
    for entry in stats.values():
        if entry["name"] == get_name(deck):
            return entry["total_in_deck"]
    raise RuntimeError(f"No stats returned for deck: {deck}")


def save_config(config):
    invoke("saveDeckConfig", {"config": config})


def assign_config(deck, config_id):
    invoke("setDeckConfigId", {
        "decks": [deck],
        "configId": config_id,
    })


def clone_config(name):
    return invoke("cloneDeckConfigId", {"name": name})


###############################################################################
# Config Creation
###############################################################################

def configure(cfg):
    """
    Modify a deck config in-place.
    Keeps existing FSRS weights.
    """

    cfg["name"] = cfg["name"]

    cfg["desiredRetention"] = 0.85
    cfg["autoplay"] = True
    cfg["replayq"] = False        # never auto-replay front audio

    cfg["new"]["perDay"] = 10
    cfg["rev"]["perDay"] = 9999
    cfg['weightSearch'] = ''

    return cfg


def get_all_deck_configs():
    """
    Returns every config group currently in use by any real deck,
    keyed by config id (dedup'd, since many decks can share one config).
    """
    configs = {}
    for deck in get_all_decks():
        cfg = get_deck_config(deck)
        configs[cfg["id"]] = cfg
    return configs


def ensure_config(name, template_cfg):
    """
    Creates the config if it doesn't exist.
    Otherwise updates the existing one.
    """

    all_configs = get_all_deck_configs()

    for cfg in all_configs.values():
        if cfg["name"] == name:
            cfg = copy.deepcopy(cfg)
            configure(cfg)
            save_config(cfg)
            return cfg["id"]

    config_id = clone_config(name)

    cfg = copy.deepcopy(template_cfg)
    cfg["id"] = config_id
    cfg["name"] = name

    configure(cfg)
    save_config(cfg)

    return config_id


###############################################################################
# Main
###############################################################################

# for action in ("Reading", "Listening"):

#     root = f"Japanese-{action}"
#     video_root = f"{root}::Japanese Video Deck"

#     print(f"\nProcessing {root}")

#     decks = sorted(
#         d for d in get_all_decks()
#         if d == root or d.startswith(root + "::")
#     )

#     ###########################################################################
#     # Parent shell
#     ###########################################################################

#     # print(f"Leaving parent shell alone: {root}")

#     ###########################################################################
#     # Shared Video Deck config
#     ###########################################################################

#     template = get_deck_config(root)

#     shared_video_config = ensure_config(
#         f"Japanese Video Deck-{action}",
#         template,
#     )

#     ###########################################################################
#     # Process every deck
#     ###########################################################################

#     for deck in decks:

#         # ------------------------------------------------------------------
#         # Japanese Video Deck shell
#         # ------------------------------------------------------------------

#         # if deck == video_root:
#         #     print(f"Leaving shell: {deck}")
#         #     continue

#         # ------------------------------------------------------------------
#         # Video deck children -> one shared config
#         # ------------------------------------------------------------------

#         if deck.startswith(video_root + "::"):
#             print(f"{deck}")
#             assign_config(deck, shared_video_config)
#             continue

#         # ------------------------------------------------------------------
#         # Ignore empty shell decks
#         # ------------------------------------------------------------------

#         # if get_card_count(deck) == 0:
#         #     print(f"Skipping empty shell: {deck}")
#         #     continue

#         # ------------------------------------------------------------------
#         # Normal deck
#         # ------------------------------------------------------------------

#         cfg_name = f"{deck}-{action}"

#         cfg_id = ensure_config(
#             cfg_name,
#             get_deck_config(deck),
#         )

#         print(f"{deck} -> {cfg_name}")

#         assign_config(deck, cfg_id)

pprint({d: get_deck_config(d) for d in get_all_decks()})

