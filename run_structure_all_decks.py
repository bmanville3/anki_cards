"""
Run the export → transform → import pipeline one step at a time.

Usage:
    python run_all_decks.py export
    python run_all_decks.py transform
    python run_all_decks.py import
    python run_all_decks.py export --decks genki_i core2000
    python run_all_decks.py transform --sample 5
    python run_all_decks.py import --dry-run
"""

import argparse
import logging
import pprint
import subprocess
import sys
from pathlib import Path

from attrs import define, field
from cattrs import unstructure
import requests

logger = logging.getLogger(__name__)

MEDIA_SRC = Path("/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media")
WORK_DIR  = Path("./pipeline")
SCRIPT    = Path("structure_all_cards.py")


@define
class DeckConfig:
    deck:     str
    slug:     str
    subdecks: list[str] = field(factory=list)

    def __attrs_post_init__(self):
        if not self.subdecks:
            return
        if self.subdecks == ["auto"]:
            resp = requests.post(
                "http://localhost:8765",
                json={"action": "deckNames", "version": 6, "params": {}},
            ).json()
            all_decks = resp.get("result") or []
            prefix = self.deck + "::"
            self.subdecks = sorted(d for d in all_decks if d.startswith(prefix))
            if not self.subdecks:
                self.subdecks = []


DECKS = [
    DeckConfig(deck="Core 2000",               slug="core2000"),
    DeckConfig(deck="Genki I",                 slug="genki_i"),
    DeckConfig(deck="In the Wild",             slug="in_the_wild"),
    DeckConfig(deck="Japanese Video Deck",     slug="japanese_video", subdecks=["auto"]),
    DeckConfig(deck="Jlab's beginner course",  slug="jlab"),
]


def slug_for_subdeck(parent_slug: str, deck_name: str) -> str:
    parts = deck_name.split("::")
    child_parts = parts[1:] if len(parts) > 1 else parts
    return "__".join(p.lower().replace(" ", "_") for p in child_parts)


def run(cmd: list[str], desc: str) -> bool:
    logger.info("▶ %s", desc)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error("✗ Failed: %s (exit %d)", desc, result.returncode)
        return False
    return True


def iter_leaf_dirs(cfg: DeckConfig, parent_dir: Path) -> list[tuple[str, Path]]:
    """
    Returns a list of (deck_name, deck_dir) pairs — one per leaf to process.
    Flat decks:    [(cfg.deck, parent_dir)]
    Subdecks:      [("Deck::Sub1", parent_dir/sub1), ("Deck::Sub2", parent_dir/sub2), ...]
    """
    if not cfg.subdecks:
        return [(cfg.deck, parent_dir)]
    return [
        (subdeck, parent_dir / slug_for_subdeck(cfg.slug, subdeck))
        for subdeck in cfg.subdecks
    ]


def export_deck(cfg: DeckConfig, parent_dir: Path, sample: int | None) -> bool:
    for deck_name, deck_dir in iter_leaf_dirs(cfg, parent_dir):
        deck_dir.mkdir(parents=True, exist_ok=True)
        raw_csv   = deck_dir / "raw_cards.csv"
        media_out = deck_dir / "media"
        cmd = [
            sys.executable, str(SCRIPT), "export",
            "--deck",      deck_name,
            "--out",       str(raw_csv),
            "--media-src", str(MEDIA_SRC),
            "--media-out", str(media_out),
        ]
        if sample:
            cmd += ["--sample", str(sample)]
        if not run(cmd, f"export '{deck_name}' → {raw_csv}"):
            return False
    return True


def transform_deck(cfg: DeckConfig, parent_dir: Path, sample: int | None) -> bool:
    ok = True
    for _, deck_dir in iter_leaf_dirs(cfg, parent_dir):
        raw_csv         = deck_dir / "raw_cards.csv"
        transformed_csv = deck_dir / "transformed_cards.csv"
        media_out       = deck_dir / "media"
        if not raw_csv.exists():
            logger.warning("No raw CSV at '%s', skipping", raw_csv)
            ok = False
            continue
        cmd = [
            sys.executable, str(SCRIPT), "transform",
            "--in",    str(raw_csv),
            "--out",   str(transformed_csv),
            "--media", str(media_out),
        ]
        if sample:
            cmd += ["--sample", str(sample)]
        if not run(cmd, f"transform '{raw_csv}'"):
            ok = False
    return ok


def import_deck(cfg: DeckConfig, parent_dir: Path, dry_run: bool) -> bool:
    ok = True
    for _, deck_dir in iter_leaf_dirs(cfg, parent_dir):
        transformed_csv = deck_dir / "transformed_cards.csv"
        if not transformed_csv.exists():
            logger.warning("No transformed CSV at '%s', skipping", transformed_csv)
            ok = False
            continue
        cmd = [
            sys.executable, str(SCRIPT), "import",
            "--in", str(transformed_csv),
        ]
        if not dry_run:
            cmd.append("--no-dry-run")
        if not run(cmd, f"import '{transformed_csv}'" + (" (dry run)" if dry_run else "")):
            ok = False
    return ok


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--decks",  nargs="+", default=None, metavar="SLUG",
                   help="Only process these deck slugs (e.g. genki_i core2000)")
    p.add_argument("--sample", type=int, default=None, help="Sample N cards per deck")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="step", required=True)

    p_export = sub.add_parser("export", help="Export decks from Anki to raw CSV")
    add_common_args(p_export)

    p_transform = sub.add_parser("transform", help="Transform raw CSV to import-ready CSV")
    add_common_args(p_transform)

    p_import = sub.add_parser("import", help="Import transformed CSV into target app")
    add_common_args(p_import)
    p_import.add_argument("--dry-run", action="store_true", help="Dry-run without writing")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    decks = DECKS
    if args.decks:
        decks = [d for d in DECKS if d.slug in args.decks]
        if not decks:
            logger.error("No matching decks found for slugs: %s", args.decks)
            sys.exit(1)

    logger.info("Processing decks:\n%s", pprint.pformat(unstructure(decks)))

    failures: list[str] = []

    for cfg in decks:
        parent_dir = WORK_DIR / cfg.slug
        parent_dir.mkdir(parents=True, exist_ok=True)
        logger.info("━━━ %s ━━━", cfg.deck)

        if args.step == "export":
            if not export_deck(cfg, parent_dir, args.sample):
                failures.append(f"{cfg.slug}:export")
        elif args.step == "transform":
            if not transform_deck(cfg, parent_dir, args.sample):
                failures.append(f"{cfg.slug}:transform")
        elif args.step == "import":
            if not import_deck(cfg, parent_dir, args.dry_run):
                failures.append(f"{cfg.slug}:import")

    logger.info("━━━ Done ━━━")
    if failures:
        logger.error("Failures: %s", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
