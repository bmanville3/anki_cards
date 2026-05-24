import fugashi
from jamdict import Jamdict
from jamdict.util import LookupResult
import unidic_lite

from common.types import Sense

MAX_ENTRIES           = 10
MAX_SENSES            = 10
MAX_GLOSSES_PER_SENSE = 2

JMD = Jamdict()

def jamdict_senses_for(surface: str, lemma: str) -> tuple[list[Sense], LookupResult]:
    if JMD is None:
        return [], None
    result = JMD.lookup(lemma) or JMD.lookup(surface)
    if not result or not result.entries:
        return [], result
    senses: list[Sense] = []
    seen: set[tuple[str, str]] = set()
    idx = 0
    for entry in result.entries[:MAX_ENTRIES]:
        for jmd_sense in entry.senses[:MAX_SENSES]:
            glosses = [g.text for g in jmd_sense.gloss[:MAX_GLOSSES_PER_SENSE]]
            if not glosses:
                continue
            meaning = ", ".join(glosses)
            pos_str = "/".join(sorted(str(p) for p in jmd_sense.pos))
            key = (meaning, pos_str)
            if key in seen:
                continue
            seen.add(key)
            senses.append(Sense(index=idx, meaning=meaning, pos=pos_str))
            idx += 1
    return senses, result
