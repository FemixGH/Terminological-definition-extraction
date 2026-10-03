"""Extract terms and their definitions from text with a trained checkpoint.

Examples:
    python extract.py --model-dir experiments/transformer_crf_v3/model --text "An algorithm is a finite sequence of instructions to solve a problem."
    python extract.py --model-dir experiments/transformer_crf_v3/model --input-file paragraphs.txt --output result.jsonl

Input is split into sentences (one per line with --one-sentence-per-line, otherwise by punctuation). The model
labels words (whitespace-delimited) as TERM / DEF with IOB2 tags; every predicted span is printed as text.
Russian and English are handled by the same model.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import logging as hf_logging
from seqeval.metrics.sequence_labeling import get_entities
from transformers import AutoConfig, AutoTokenizer

import training_bilstm_term as base

hf_logging.set_verbosity_error()
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ\"«(\[])")


def split_sentences(text: str):
    sentences = []
    for paragraph in re.split(r"\n+", text.strip()):
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(s.strip() for s in SENTENCE_SPLIT.split(paragraph) if s.strip())
    return sentences


class Extractor:
    def __init__(self, model_dir: str, model_name: str = base.TRAINING_CONFIG["model_name"], max_length: int = 512,
                 batch_size: int = 16, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        config._name_or_path = model_name  # encoder architecture from the hub name, weights from the checkpoint
        config.load_encoder_weights = False
        self.id2label = {int(k): v for k, v in config.id2label.items()}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = base.BertCRFForTokenClassification.from_pretrained(model_dir, config=config).to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size

    def tag(self, sentences):
        """Word-level IOB2 tags (first subword of each whitespace-delimited word); truncated words get O."""
        words_per_sentence = [s.split() for s in sentences]
        order = sorted(range(len(sentences)), key=lambda i: len(words_per_sentence[i]))
        tags = [None] * len(sentences)
        for start in range(0, len(order), self.batch_size):
            ids = order[start:start + self.batch_size]
            batch = [words_per_sentence[i] or ["."] for i in ids]
            enc = self.tokenizer(batch, is_split_into_words=True, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                pred = self.model.predict(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            for row, i in enumerate(ids):
                row_tags, seen = ["O"] * len(words_per_sentence[i]), set()
                for pos, w in enumerate(enc.word_ids(row)):
                    if w is None or w in seen or w >= len(row_tags) or pos >= len(pred[row]):
                        continue
                    seen.add(w)
                    label = int(pred[row][pos])
                    row_tags[w] = self.id2label.get(label, "O") if label >= 0 else "O"
                tags[i] = row_tags
        return words_per_sentence, tags

    def extract(self, sentences):
        words_per_sentence, tags = self.tag(sentences)
        results = []
        for sentence, words, tg in zip(sentences, words_per_sentence, tags):
            spans = {"TERM": [], "DEF": []}
            for kind, start, end in get_entities(tg):
                if kind in spans:
                    spans[kind].append(" ".join(words[start:end + 1]))
            results.append({"sentence": sentence, "terms": spans["TERM"], "definitions": spans["DEF"]})
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-name", default=base.TRAINING_CONFIG["model_name"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--input-file")
    ap.add_argument("--one-sentence-per-line", action="store_true")
    ap.add_argument("--only-with-definitions", action="store_true", help="Skip sentences without a predicted definition.")
    ap.add_argument("--output", help="JSONL output file (default: print to stdout)")
    args = ap.parse_args()

    text = args.text if args.text is not None else Path(args.input_file).read_text(encoding="utf-8")
    sentences = [l.strip() for l in text.splitlines() if l.strip()] if args.one_sentence_per_line else split_sentences(text)
    results = Extractor(args.model_dir, args.model_name).extract(sentences)
    if args.only_with_definitions:
        results = [r for r in results if r["definitions"]]
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    for r in results:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.output:
        out.close()


if __name__ == "__main__":
    main()
