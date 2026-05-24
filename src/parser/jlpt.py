import json
from pathlib import Path


JLPT_DATA_DIR = Path("./yomitan-jlpt-vocab")

def load_jlpt_data(directory: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    files = sorted(directory.glob("term_meta_bank_*.json"))
    if not files:
        print(f"  Warning: no term_meta_bank_*.json files found in {directory} — JLPT badges disabled.")
        return {}
    for f in files:
        try:
            entries = json.loads(f.read_text(encoding="utf-8"))
            for entry in entries:
                if not (isinstance(entry, list) and len(entry) >= 3):
                    continue
                word    = entry[0]
                meta    = entry[2]
                display = meta.get("frequency", {}).get("displayValue", "")
                if display.startswith("N") and display[1:].isdigit():
                    level = int(display[1:])
                    if word not in mapping or level > mapping[word]:
                        mapping[word] = level
        except Exception as e:
            print(f"  Failed to load {f.name}: {e}")
    print(f"  JLPT data loaded: {len(mapping)} entries.")
    return mapping


_JLPT_MAP: dict[str, int] = load_jlpt_data(JLPT_DATA_DIR)


def jlpt_for_result(result) -> str:
    if not _JLPT_MAP:
        return ""
    best: int | None = None
    for entry in result.entries:
        for form in list(entry.kanji_forms) + list(entry.kana_forms):
            level = _JLPT_MAP.get(form.text)
            if level is not None and (best is None or level > best):
                best = level
    return f"N{best}" if best is not None else ""


def kata_to_moras(kana: str) -> list[str]:
    SMALL = set("ァィゥェォャュョヮヵヶぁぃぅぇぉゃゅょゎ")
    moras, i = [], 0
    while i < len(kana):
        if i + 1 < len(kana) and kana[i + 1] in SMALL:
            moras.append(kana[i : i + 2])
            i += 2
        else:
            moras.append(kana[i])
            i += 1
    return moras


def pitch_html(reading_kata: str, accent_type: str) -> str:
    if not reading_kata or not accent_type:
        return ""
    try:
        n = int(accent_type.split(",")[0])
    except ValueError:
        return ""
    moras = kata_to_moras(reading_kata)
    if not moras:
        return ""
    parts = []
    for i, mora in enumerate(moras):
        mora_pos = i + 1
        if n == 0:
            high = mora_pos > 1
        elif n == 1:
            high = mora_pos == 1
        else:
            high = 2 <= mora_pos <= n
        cls = "pa-high" if high else "pa-low"
        parts.append(f'<span class="{cls}">{mora}</span>')
        if n != 0 and mora_pos == n:
            parts.append('<span class="pa-drop">↘</span>')
    pattern_label = "平板" if n == 0 else f"型{n}"
    return (
        f'<span class="pa-wrap">'
        f'{"".join(parts)}'
        f'<span class="pa-label">{pattern_label}</span>'
        f'</span>'
    )