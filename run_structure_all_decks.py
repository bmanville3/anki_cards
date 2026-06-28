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


def run(cmd: list[str], desc: str) -> bool:
    logger.info("▶ %s", desc)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error("✗ Failed: %s (exit %d)", desc, result.returncode)
        return False
    return True


def export_deck(cfg: DeckConfig, media_out: Path, raw_csv: Path, sample: int | None) -> bool:
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
        deck_dir        = WORK_DIR / cfg.slug
        media_out       = deck_dir / "media"
        raw_csv         = deck_dir / "raw_cards.csv"
        transformed_csv = deck_dir / "transformed_cards.csv"
        deck_dir.mkdir(parents=True, exist_ok=True)

        logger.info("━━━ %s ━━━", cfg.deck)

        if args.step == "export":
            if not export_deck(cfg, media_out, raw_csv, args.sample):
                failures.append(f"{cfg.slug}:export")

        elif args.step == "transform":
            if not raw_csv.exists():
                logger.warning("No raw CSV for '%s', skipping", cfg.slug)
                failures.append(f"{cfg.slug}:transform:no_raw_csv")
                continue
            if not transform_deck(raw_csv, transformed_csv, media_out, args.sample):
                failures.append(f"{cfg.slug}:transform")

        elif args.step == "import":
            if not transformed_csv.exists():
                logger.warning("No transformed CSV for '%s', skipping", cfg.slug)
                failures.append(f"{cfg.slug}:import:no_transformed_csv")
                continue
            if not import_deck(transformed_csv, args.dry_run):
                failures.append(f"{cfg.slug}:import")

    logger.info("━━━ Done ━━━")
    if failures:
        logger.error("Failures: %s", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
