from pathlib import Path

FONTS_DIR     = Path("./fonts")

JAPANESE_FONTS = [
    "NotoSansJP-Regular",
    "NotoSerifJP-Regular",
    "MPLUSRounded1c-Regular",
    "MPLUS1p-Regular",
    "KosugiMaru-Regular",
    "SawarabiGothic-Regular",
    "SawarabiMincho-Regular",
    "ZenKakuGothicNew-Regular",
    "ZenAntique-Regular",
    "ShipporiMincho-Regular",
]

def build_font_face_css(fonts_dir: Path) -> str:
    if not fonts_dir.exists():
        print(f"  Warning: {fonts_dir} not found — no local fonts will be embedded.")
        return ""
    css_blocks = []
    for ttf in sorted(fonts_dir.glob("*.ttf")):
        family_name = ttf.stem.replace("-", " ").replace("_", " ")
        css_blocks.append(
            f"@font-face {{ font-family: '{family_name}'; src: url('{ttf.name}'); }}"
        )
    if not css_blocks:
        print(f"  Warning: no .ttf files found in {fonts_dir}.")
    return "\n".join(css_blocks)


def pos_to_class(pos_str: str) -> str:
    p = pos_str.lower()
    if any(x in p for x in ("verb", "vt", "vi", "vs", "vk", "v1", "v5")):
        return "pos-verb"
    if any(x in p for x in ("noun", "counter", "temporal")):
        return "pos-noun"
    if any(x in p for x in ("adjective", "adj")):
        return "pos-adj"
    if any(x in p for x in ("adverb", "adv")):
        return "pos-adv"
    if any(x in p for x in ("expression", "idiomatic")):
        return "pos-expr"
    return "pos-other"


def font_for_index(index: int) -> str:
    return JAPANESE_FONTS[index % len(JAPANESE_FONTS)]
