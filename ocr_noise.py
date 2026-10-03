"""Synthetic OCR / PDF-extraction noise for robustness tests and noise augmentation.

Operations are word-local and deterministic given the word and the seed, so a term or a definition is corrupted exactly like the
same words inside the sentence and the annotation stays aligned. Noise types (all scaled by `level`, roughly the share of affected words):
  confusions   rn<->m, cl<->d, l<->1/I, O<->0, e<->c, vv<->w, ... (typical OCR character confusions)
  ligatures    fi/fl/ff -> single ligature characters
  hyphen       a space after a hyphen at line breaks ("defini- tion") or a broken word ("defin ition")
  junk         stray symbols and formula-like fragments inserted at random word positions (also inside terms/definitions, so they never mark span boundaries)

Example:
    python ocr_noise.py --demo
"""
import argparse
import hashlib
import random
import re
from typing import Dict, List, Tuple

CONFUSIONS = [("rn", "m"), ("m", "rn"), ("cl", "d"), ("d", "cl"), ("vv", "w"), ("w", "vv"), ("l", "1"), ("l", "I"), ("I", "l"),
              ("O", "0"), ("0", "O"), ("e", "c"), ("c", "e"), ("h", "b"), ("b", "h"), ("ii", "u"), ("a", "o"), ("i", "j")]
LIGATURES = {"fi": "ﬁ", "fl": "ﬂ", "ff": "ﬀ", "ffi": "ﬃ"}
JUNK = ["~", "•", "|", "_", "^", "ø", "§", "Σ", "∫", "( x ) =", "p ( y | x )", "f_i ( x , y )", "{ w1 , w2 }", "x^2 + y^2",
        "[ 1 ]", "( 3 )", "∑ i = 1 n", "log P ( w )", "a_i b_j"]


def _rng(word: str, seed: int) -> random.Random:
    return random.Random(int(hashlib.md5(f"{seed}|{word}".encode()).hexdigest()[:12], 16))


def corrupt_word(word: str, level: float, seed: int = 0) -> str:
    if len(word) < 3 or level <= 0:
        return word
    rng = _rng(word, seed)
    out = word
    for src, dst in CONFUSIONS:  # each eligible confusion applied with probability `level`
        if src in out and rng.random() < level:
            idx = [m.start() for m in re.finditer(re.escape(src), out)]
            i = rng.choice(idx)
            out = out[:i] + dst + out[i + len(src):]
    for src, dst in LIGATURES.items():
        if src in out and rng.random() < level * 2:
            out = out.replace(src, dst, 1)
    if len(out) > 6 and rng.random() < level * 0.6:  # broken word / line-break hyphenation
        cut = rng.randint(3, len(out) - 3)
        out = out[:cut] + ("- " if rng.random() < 0.5 else " ") + out[cut:]
    return out


def corrupt_text(text: str, level: float, seed: int = 0) -> str:
    return " ".join(corrupt_word(w, level, seed) for w in text.split())


def corrupt_record(rec: Dict, level: float, seed: int = 0) -> Dict:
    """Corrupt a unified-format record (text + optional term/definition substrings), keeping the annotation aligned."""
    from training_bilstm_term import find_definition_span, find_term_span

    text = rec["text"].strip()
    definition, term = rec.get("definition"), rec.get("term")
    d_span = find_definition_span(text, definition) if definition else None
    t_span = find_term_span(text, term, d_span) if term else None
    spans: List[Tuple[int, int, str]] = []
    if d_span:
        spans.append((d_span[0], d_span[1], "definition"))
    if t_span:
        spans.append((t_span[0], t_span[1], "term"))
    spans.sort()
    # segments: gaps between spans and the spans themselves; junk goes to a RANDOM word position of a random segment (inside
    # terms/definitions too), so that it carries no information about the span boundaries
    segments, cursor = [], 0
    for start, end, name in spans:
        if start < cursor:  # overlapping spans: leave the rest as is
            continue
        segments.append((None, text[cursor:start]))
        segments.append((name, text[start:end]))
        cursor = end
    segments.append((None, text[cursor:]))
    seg_words = [corrupt_text(t, level, seed).split() for _, t in segments]
    rng = random.Random(f"{seed}|{text}")
    n_junk = 0
    if level > 0:
        n_junk = (rng.random() < min(1.0, level * 6)) + (rng.random() < min(1.0, level * 4))
    total = sum(len(w) for w in seg_words)
    for _ in range(n_junk if total else 0):
        k = rng.randrange(total + 1)  # global word position
        for words in seg_words:
            if k <= len(words):
                words.insert(k, rng.choice(JUNK))
                break
            k -= len(words)
        total += 1
    fields = {name: " ".join(w) for (name, _), w in zip(segments, seg_words) if name}
    pieces = [" ".join(w) for w in seg_words]
    new = dict(rec)
    new["text"] = " ".join(p for p in pieces if p).strip()
    for key in ("term", "definition"):
        if key in fields:
            new[key] = fields[key]
    return new


def augment_records(records: List[Dict], fraction: float, max_level: float = 0.12, seed: int = 0) -> List[Dict]:
    """Return `records` plus noisy copies of a random `fraction` of them (noise level uniform in [0.02, max_level])."""
    rng = random.Random(seed)
    extra = []
    for i, rec in enumerate(records):
        if rng.random() < fraction:
            extra.append(corrupt_record(rec, rng.uniform(0.02, max_level), seed=i))
    return records + extra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--level", type=float, default=0.08)
    args = ap.parse_args()
    if args.demo:
        rec = {"text": "A hidden Markov model is a statistical model in which the system being modeled is assumed to be a Markov process.",
               "term": "hidden Markov model", "definition": "a statistical model in which the system being modeled is assumed to be a Markov process"}
        for level in (0.0, 0.03, args.level, 0.15):
            out = corrupt_record(rec, level, seed=3)
            print(f"[{level:.2f}] {out['text']}\n       term={out.get('term')!r}")


if __name__ == "__main__":
    main()
