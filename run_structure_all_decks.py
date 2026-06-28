"""
Run the full export → transform → import pipeline across all decks.

Usage:
    # Full pipeline (export + transform + import)
    python run_all_decks.py

    # Just transform + import (if you already exported)
    python run_all_decks.py --skip-export

    # Just export (no transform, no import)
    python run_all_decks.py --export-only

    # Dry-run import (see what would happen without writing)
    python run_all_decks.py --skip-export --dry-run

    # Sample N cards per deck (useful for testing)
    python run_all_decks.py --sample 5
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

MEDIA_SRC   = Path("/Users/bmanville3/Library/Application Support/Anki2/User 1/collection.media")
WORK_DIR    = Path("./pipeline")   # all CSVs and media land here
SCRIPT      = Path("structure_all_cards.py")

@define
class DeckConfig:
    # The Anki deck name passed to --deck
    deck:      str
    # Stem for raw_cards and transformed_cards CSVs (no extension)
    slug:      str
    # If the deck has subdecks, export each separately and combine
    subdecks:  list[str] = field(factory=list)

    def __attrs_post_init__(self):
        if not self.subdecks:
            return
        # If subdecks was explicitly set to ["auto"], discover them via AnkiConnect
        if self.subdecks == ["auto"]:
            resp = requests.post(
                "http://localhost:8765",
                json={"action": "deckNames", "version": 6, "params": {}},
            ).json()
            all_decks = resp.get("result") or []
            prefix    = self.deck + "::"
            self.subdecks = sorted(d for d in all_decks if d.startswith(prefix))
            if not self.subdecks:
                # No subdecks found, treat as flat deck
                self.subdecks = []

DECKS = [
    DeckConfig(deck="Core 2000",               slug="core2000"),
    DeckConfig(deck="Genki I",                 slug="genki_i"),
    DeckConfig(deck="In the Wild",             slug="in_the_wild"),
    DeckConfig(deck="Japanese Video Deck",     slug="japanese_video", subdecks=["auto"]),
    DeckConfig(deck= "Jlab's beginner course", slug= "jlab"),
]


def run(cmd: list[str], desc: str) -> bool:
    """Run a subprocess command, return True on success."""
    logger.info("▶ %s", desc)
    logger.debug("  cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error("✗ Failed: %s (exit %d)", desc, result.returncode)
        return False
    return True


def export_deck(cfg: DeckConfig, media_out: Path, raw_csv: Path, sample: int | None) -> bool:
    """Export a single deck (or its subdecks) to raw CSV."""
    decks_to_export = cfg.subdecks if cfg.subdecks else [cfg.deck]

    for deck_name in decks_to_export:
        cmd = [
            sys.executable, str(SCRIPT), "export",
            "--deck",      deck_name,
            "--out",       str(raw_csv),
            "--media-src", str(MEDIA_SRC),
            "--media-out", str(media_out),
        ]
        if sample:
            cmd += ["--sample", str(sample)]
        if not run(cmd, f"export '{deck_name}'"):
            return False
    return True


def transform_deck(raw_csv: Path, transformed_csv: Path, media_out: Path, sample: int | None) -> bool:
    cmd = [
        sys.executable, str(SCRIPT), "transform",
        "--in",    str(raw_csv),
        "--out",   str(transformed_csv),
        "--media", str(media_out),
    ]
    if sample:
        cmd += ["--sample", str(sample)]
    return run(cmd, f"transform '{raw_csv.name}'")


def import_deck(transformed_csv: Path, dry_run: bool) -> bool:
    cmd = [
        sys.executable, str(SCRIPT), "import",
        "--in", str(transformed_csv),
    ]
    if not dry_run:
        cmd.append("--no-dry-run")
    return run(cmd, f"import '{transformed_csv.name}'" + (" (dry run)" if dry_run else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-export",  action="store_true", help="Skip export step (use existing raw CSVs)")
    parser.add_argument("--export-only",  action="store_true", help="Only export, skip transform and import")
    parser.add_argument("--skip-transform", action="store_true", help="Skip transform step (use existing transformed CSVs)")
    parser.add_argument("--dry-run",      action="store_true", help="Dry-run the import step")
    parser.add_argument("--sample",       type=int, default=None, help="Sample N cards per deck (for testing)")
    parser.add_argument("--decks",        nargs="+", default=None, metavar="SLUG",
                        help="Only process these deck slugs (e.g. genki_i core2000)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    decks = DECKS
    if args.decks:
        decks = [d for d in DECKS if d.slug in args.decks]
        if not decks:
            logger.error("No matching decks found for slugs: %s", args.decks)
            sys.exit(1)
    
    logger.info("All decks:\n%s", pprint.pformat(unstructure(decks)))

    failures: list[str] = []

    for cfg in decks:
        media_out       = WORK_DIR / cfg.slug / "media"
        raw_csv         = WORK_DIR / cfg.slug / "raw_cards.csv"
        transformed_csv = WORK_DIR / cfg.slug / "transformed_cards.csv"
        (WORK_DIR / cfg.slug).mkdir(parents=True, exist_ok=True)

        logger.info("━━━ %s ━━━", cfg.deck)

        if not args.skip_export:
            if not export_deck(cfg, media_out, raw_csv, args.sample):
                failures.append(f"{cfg.slug}:export")
                continue

        if args.export_only:
            continue

        if not args.skip_transform:
            if not raw_csv.exists():
                logger.warning("No raw CSV for '%s', skipping transform", cfg.slug)
                failures.append(f"{cfg.slug}:transform:no_raw_csv")
                continue
            if not transform_deck(raw_csv, transformed_csv, media_out, args.sample):
                failures.append(f"{cfg.slug}:transform")
                continue

        if not transformed_csv.exists():
            logger.warning("No transformed CSV for '%s', skipping import", cfg.slug)
            failures.append(f"{cfg.slug}:import:no_transformed_csv")
            continue
        if not import_deck(transformed_csv, args.dry_run):
            failures.append(f"{cfg.slug}:import")

    logger.info("━━━ Done ━━━")
    if failures:
        logger.error("Failures: %s", ", ".join(failures))
        sys.exit(1)
    else:
        logger.info("All decks completed successfully.")


if __name__ == "__main__":
    main()
