"""Evaluate a trained checkpoint on held-out test files (word-level spans, seqeval IOB2).

Reports, per test file and per language: entity micro P/R/F1 and span F1 for DEF and TERM.
"Entity Macro-F1" is the mean of the DEF and TERM span-level F1.

Example:
    python evaluate_model.py --model-dir experiments/<run>/model \
        --test-files output/test_deft.json output/test_wcl.json output/test_w00.json
"""
import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments

import training_bilstm_term as base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--test-files", nargs="+", required=True)
    ap.add_argument("--model-name", default=base.TRAINING_CONFIG["model_name"],
                    help="Hub name of the encoder the checkpoint was trained from.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", default=None, help="Optional JSON file for the results.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    # Build the encoder architecture from the original hub name; the trained weights are loaded below.
    config._name_or_path = args.model_name
    config.load_encoder_weights = False
    model = base.BertCRFForTokenClassification.from_pretrained(model_dir, config=config)
    trainer = base.CRFTrainerWithLLRD(
        model=model,
        args=TrainingArguments(output_dir=str(model_dir / "_eval_tmp"), per_device_eval_batch_size=args.batch_size,
                               report_to=[], fp16=True, save_strategy="no"),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=base.compute_metrics,
        word_start_ids=base.build_word_start_ids(tokenizer),
    )
    results = base.evaluate_test_files(trainer, tokenizer, args.test_files)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
