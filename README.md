# Multilingual definition extraction (EN/RU)

XLM-RoBERTa Large + CRF tagger that finds terms and their definitions in English and Russian sentences
(IOB2 labels: `O`, `B-DEF`, `I-DEF`, `B-TERM`, `I-TERM`).

## Install

```bash
pip install -r requirements.txt
```

Training expects a GPU with about 16 GB of memory (batch size 10 with gradient accumulation 5).

## Data

`output/train.json` and `output/test_{deft,wcl,w00}.json` (one JSON object per line: `text`, `term`, `definition`,
`language`, `corpus`, `group`). They merge three English corpora and their Russian machine translations:

- DEFT (Spala et al., SemEval-2020 Task 6), CC BY-NC-SA 4.0
- WCL (Navigli and Velardi, 2010)
- W00 (Jin et al., 2013)

Translation pairs and duplicate texts share a `group`; the train/test split is done by group, so no sentence or
its translation appears on both sides. **The data is for non-commercial research use only** (DEFT licence);
please follow the licences of the original corpora.

## Train

```bash
python training_bilstm_term.py --preset transformer_crf --no-cross-validation --no-torch-compile \
    --batch-size 10 --gradient-accumulation-steps 5
```

Other presets: `transformer_softmax`, `transformer_bilstm`, `transformer_bilstm_crf`, `bilstm_crf_no_bert`,
`crf_no_bert`, `bilstm_no_bert`. Results and the model are written to `experiments/<name>/`.

## Test

```bash
python evaluate_model.py --model-dir experiments/transformer_crf/model \
    --test-files output/test_deft.json output/test_wcl.json output/test_w00.json
```

Metrics are word-level span metrics (seqeval); `Entity Macro-F1` is the mean of the DEF and TERM span F1.

## Use on text

```bash
python extract.py --model-dir experiments/transformer_crf/model --text "A graph is a set of vertices and edges."
```

## Licence

Code: not chosen yet (add a LICENSE file before publishing). Data: see above.
