import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PretrainedConfig,
)
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_scheduler
from torchcrf import CRF
from evaluate import load
from sklearn.metrics import precision_recall_fscore_support

metric = load("seqeval")

TRAINING_CONFIG = {
    "model_name": "FacebookAI/xlm-roberta-large",
    "data_path": "output/train.json",
    "labels": ["O", "B-DEF", "I-DEF", "B-TERM", "I-TERM"],
    "preset_name": None,

    "use_bert_encoder": True,
    "embedding_dim": 256,
    "embedding_dropout": 0.1,
    "embedding_learning_rate": 0.0005,

    "use_llrd": True,
    "llrd_decay_rate": 0.95,

    "use_bilstm": True,
    "bilstm_hidden_size": 256,
    "bilstm_num_layers": 1,
    "bilstm_dropout": 0.3,

    "use_term_only_examples": True,
    "use_oversampling": False,
    "oversampling_max_repeat": 2,

    "use_cross_validation": True,
    "n_folds": 5,
    "cv_seed": 42,

    "bert_base_learning_rate": 0.00002,
    "classifier_learning_rate": 0.0001,
    "crf_learning_rate": 0.0005,
    "bilstm_learning_rate": 0.0002,

    "lr_scheduler_type": "cosine_with_restarts",
    "num_cycles": 0.5,
    "warmup_ratio": 0.1,
    "warmup_steps": 0,

    "batch_size": 25,
    "gradient_accumulation_steps": 2,

    "num_train_epochs": 20,
    "weight_decay": 0.002,
    "max_length": 192,
    "split_ratio": 0.15,
    "max_grad_norm": 1.0,

    "fp16": True,
    "fp16_opt_level": "O1",
    "fp16_full_eval": True,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 2,

    "gradient_checkpointing": False,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,

    "base_output_dir": "./experiments",
    "logging_steps": 50,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_entity_macro_f1",
    "save_total_limit": 2,

    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.0005,

    "seed": 42,
    "experiment_name": "article_bert_crf",
    "show_validation_samples": 5,

    "collect_errors": True,
    "collect_errors_per_epoch": False,

    "use_crf": True,

    # Prefix every sentence with its source corpus tag, e.g. "[DEFT] " (lets the model learn corpus-specific conventions).
    "corpus_tag": False,

    # OCR-noise augmentation: add noisy copies (character confusions, ligatures, broken words, formula junk) of this share
    # of the training records; copies keep the original's group, so validation stays leak-free.
    "ocr_noise_fraction": 0.0,
    "ocr_noise_max_level": 0.12,

    # Auxiliary "sentence contains a definition" head (multi-task regularizer, off by default).
    "use_sentence_head": False,
    "sentence_head_weight": 0.3,

    # Token-level class weighting: O vs entity labels.
    "use_token_class_weights": True,
    # Recompute O/entity weights from the current train split for each fold/run.
    "auto_token_class_weights": True,
    "token_class_weight_power": 0.5,
    "token_class_weight_max": 6.0,
    "token_class_weights": {
        "O": 1.0,
        "ENTITY": 3.0,
    },
    # With CRF enabled, add a weighted emission CE term: loss = CRF + alpha * CE.
    "crf_emission_loss_weight": 0.25,

    # Sequence-level reweighting (helps with corpus/class imbalance).
    "use_weighted_sampler": True,
    "weighted_sampler_mode": "corpus_binary",  # binary | corpus | corpus_binary
    "weighted_sampler_max_weight": 5.0,

    "torch_compile": True,
    "torch_compile_mode": "default",

    "validation_data_path": None,

    # Held-out files evaluated (word-level) after training; missing files are skipped.
    "test_data_paths": [
        "output/test_deft.json",
        "output/test_wcl.json",
        "output/test_w00.json",
    ],
}

TRAINING_PRESETS = {
    # The four architectures compared in the paper (XLM-RoBERTa Large encoder).
    "transformer_softmax": {"use_bilstm": False, "use_crf": False, "crf_emission_loss_weight": 0.0,
                            "experiment_name": "transformer_softmax"},
    "transformer_bilstm": {"use_bilstm": True, "use_crf": False, "crf_emission_loss_weight": 0.0,
                           "experiment_name": "transformer_bilstm"},
    "transformer_crf": {"use_bilstm": False, "use_crf": True, "experiment_name": "transformer_crf"},
    "transformer_bilstm_crf": {"use_bilstm": True, "use_crf": True, "experiment_name": "transformer_bilstm_crf"},

    # Pure BiLSTM + CRF mode without transformer encoder.
    "bilstm_crf_no_bert": {
        "use_bert_encoder": False,
        "use_bilstm": True,
        "use_crf": True,
        "use_llrd": False,
        "embedding_dim": 256,
        "embedding_dropout": 0.1,
        "embedding_learning_rate": 5e-4,
        "bilstm_hidden_size": 256,
        "bilstm_num_layers": 1,
        "bilstm_dropout": 0.3,
        "bilstm_learning_rate": 2e-4,
        "classifier_learning_rate": 1e-4,
        "crf_learning_rate": 5e-4,
    },
    # Encoder-free baselines of the ablation: trainable token embeddings only.
    "crf_no_bert": {
        "use_bert_encoder": False, "use_bilstm": False, "use_crf": True, "use_llrd": False,
        "embedding_dim": 256, "embedding_dropout": 0.1, "embedding_learning_rate": 5e-4,
        "classifier_learning_rate": 1e-4, "crf_learning_rate": 5e-4,
    },
    "bilstm_no_bert": {
        "use_bert_encoder": False, "use_bilstm": True, "use_crf": False, "crf_emission_loss_weight": 0.0, "use_llrd": False,
        "embedding_dim": 256, "embedding_dropout": 0.1, "embedding_learning_rate": 5e-4,
        "bilstm_hidden_size": 256, "bilstm_num_layers": 1, "bilstm_dropout": 0.3, "bilstm_learning_rate": 2e-4,
        "classifier_learning_rate": 1e-4,
    },
    # BERT + Softmax baseline with LLRD and immediate 5-fold cross-validation.
    "bert_softmax_llrd_balanced_cv5_v1": {
        "use_bert_encoder": True,
        "use_bilstm": False,
        "use_crf": False,
        "crf_emission_loss_weight": 0.0,
        "use_llrd": True,
        "llrd_decay_rate": 0.95,
        "use_cross_validation": True,
        "n_folds": 5,
        "bert_base_learning_rate": 2e-5,
        "classifier_learning_rate": 1e-4,
        "lr_scheduler_type": "cosine_with_restarts",
        "num_cycles": 0.5,
        "warmup_ratio": 0.08,
        "weight_decay": 0.01,
        "batch_size": 25,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 20,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 5e-4,
        "use_weighted_sampler": True,
        "weighted_sampler_mode": "corpus_binary",
        "weighted_sampler_max_weight": 5.0,
        "use_token_class_weights": True,
        "auto_token_class_weights": True,
        "token_class_weight_power": 0.5,
        "token_class_weight_max": 6.0,
        "experiment_name": "article_bert_softmax_cv5_v1",
    },
}


def apply_training_preset(config: Dict, presets: Dict[str, Dict]) -> None:
    preset_name = config.get("preset_name")
    if not preset_name:
        return

    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys())) or "<none>"
        raise ValueError(f"Unknown preset_name='{preset_name}'. Available presets: {available}")

    config.update(presets[preset_name])


apply_training_preset(TRAINING_CONFIG, TRAINING_PRESETS)

LABELS = TRAINING_CONFIG["labels"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
DATA_PATH = Path(TRAINING_CONFIG["data_path"])
BASE_OUTPUT_DIR = Path(TRAINING_CONFIG["base_output_dir"])
MODEL_NAME = TRAINING_CONFIG["model_name"]



class BertCRFForTokenClassification(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.use_bert_encoder = bool(getattr(config, "use_bert_encoder", True))
        self.use_bilstm = getattr(config, 'use_bilstm', False)
        self.use_crf = getattr(config, 'use_crf', True)  # Default to True for backwards compatibility

        if self.use_bert_encoder:
            if getattr(config, "load_encoder_weights", True):
                self.bert = AutoModel.from_pretrained(config._name_or_path, config=config)
            else:
                # Inference from a saved checkpoint: the encoder weights come from the checkpoint itself,
                # so neither the hub model nor a download is needed.
                self.bert = AutoModel.from_config(config)
            encoder_output_size = config.hidden_size
        else:
            embedding_dim = int(getattr(config, "embedding_dim", 256))
            embedding_dropout = float(getattr(config, "embedding_dropout", 0.1))
            pad_token_id = getattr(config, "pad_token_id", 0)
            self.token_embeddings = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=pad_token_id,
            )
            self.embedding_dropout = nn.Dropout(embedding_dropout)
            encoder_output_size = embedding_dim

        if self.use_bilstm:
            bilstm_hidden_size = getattr(config, 'bilstm_hidden_size', 256)
            bilstm_num_layers = getattr(config, 'bilstm_num_layers', 1)
            bilstm_dropout = getattr(config, 'bilstm_dropout', 0.3)

            self.bilstm = nn.LSTM(
                input_size=encoder_output_size,
                hidden_size=bilstm_hidden_size,
                num_layers=bilstm_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=bilstm_dropout if bilstm_num_layers > 1 else 0.0
            )
            self.bilstm_dropout = nn.Dropout(bilstm_dropout)
            classifier_input_size = bilstm_hidden_size * 2
        else:
            classifier_input_size = encoder_output_size

        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(classifier_input_size, config.num_labels)

        if self.use_crf:
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Optional auxiliary head: does the sentence contain a definition? (multi-task regularizer)
        self.use_sentence_head = bool(getattr(config, "use_sentence_head", False))
        self.sentence_head_weight = float(getattr(config, "sentence_head_weight", 0.3))
        if self.use_sentence_head:
            self.sentence_classifier = nn.Linear(classifier_input_size, 2)

        self.use_token_class_weights = bool(getattr(config, "use_token_class_weights", False))
        self.crf_emission_loss_weight = float(getattr(config, "crf_emission_loss_weight", 0.0))
        self.emission_label_smoothing = float(getattr(config, "emission_label_smoothing", 0.0))

        token_weights = torch.ones(config.num_labels, dtype=torch.float)
        if self.use_token_class_weights:
            cfg_weights = getattr(config, "token_class_weights", {"O": 1.0, "ENTITY": 1.0})
            o_weight = float(cfg_weights.get("O", 1.0))
            entity_weight = float(cfg_weights.get("ENTITY", 1.0))
            id2label_cfg = getattr(config, "id2label", {})
            for label_id in range(config.num_labels):
                label_name = id2label_cfg.get(label_id, id2label_cfg.get(str(label_id), ""))
                token_weights[label_id] = o_weight if label_name == "O" else entity_weight
        self.register_buffer("token_class_weights", token_weights)

        self.init_weights()

    def _encode_sequence(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if self.use_bert_encoder:
            encoder_kwargs = dict(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if head_mask is not None:  # not every encoder (e.g. DeBERTa) accepts head_mask
                encoder_kwargs["head_mask"] = head_mask
            outputs = self.bert(input_ids, **encoder_kwargs)
            sequence_output = outputs[0]
            hidden_states = getattr(outputs, "hidden_states", None)
            attentions = getattr(outputs, "attentions", None)
        else:
            sequence_output = self.token_embeddings(input_ids)
            sequence_output = self.embedding_dropout(sequence_output)
            hidden_states = None
            attentions = None

        sequence_output = self.dropout(sequence_output)

        if self.use_bilstm:
            if attention_mask is not None:
                total_length = sequence_output.size(1)
                lengths = attention_mask.sum(dim=1).cpu()
                packed_sequence = nn.utils.rnn.pack_padded_sequence(
                    sequence_output, lengths, batch_first=True, enforce_sorted=False
                )
                packed_output, _ = self.bilstm(packed_sequence)
                sequence_output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output, batch_first=True, total_length=total_length
                )
            else:
                sequence_output, _ = self.bilstm(sequence_output)

            sequence_output = self.bilstm_dropout(sequence_output)

        return sequence_output, hidden_states, attentions

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, hidden_states, attentions = self._encode_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_weights = self.token_class_weights if self.use_token_class_weights else None

            if self.use_crf:
                mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(labels, dtype=torch.bool)

                crf_labels = labels.clone()
                crf_labels[crf_labels == -100] = 0

                crf_loss = -self.crf(logits, crf_labels, mask=mask, reduction="mean")

                if self.use_token_class_weights and self.crf_emission_loss_weight > 0.0:
                    emission_loss_fct = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=-100,
                                                            label_smoothing=self.emission_label_smoothing)
                    emission_loss = emission_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    loss = crf_loss + self.crf_emission_loss_weight * emission_loss
                else:
                    loss = crf_loss
            else:
                loss_fct = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.use_sentence_head and labels is not None and loss is not None:
            id2label_cfg = getattr(self.config, "id2label", {})
            def_ids = [int(i) for i, name in id2label_cfg.items() if str(name).endswith("-DEF")]
            has_def = torch.zeros_like(labels[:, 0])
            for def_id in def_ids:
                has_def = has_def | (labels == def_id).any(dim=1).long()
            sentence_logits = self.sentence_classifier(sequence_output[:, 0])
            loss = loss + self.sentence_head_weight * nn.functional.cross_entropy(sentence_logits, has_def)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def emissions(self, input_ids, attention_mask=None):
        """Classifier logits (CRF emission scores) without decoding."""
        with torch.no_grad():
            sequence_output, _, _ = self._encode_sequence(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            return self.classifier(sequence_output)

    def decode(self, logits, attention_mask=None, o_bias=0.0):
        """Viterbi/argmax decoding of emissions. `o_bias` is a float added to the score of label O, or a tuple of
        (label, bias) pairs added to the scores of those labels, before decoding."""
        with torch.no_grad():
            logits = logits.clone()
            for label, value in ((("O", o_bias),) if not isinstance(o_bias, tuple) else o_bias):
                if value:
                    logits[..., int(self.config.label2id[label])] += value
            mask = attention_mask.bool() if attention_mask is not None else None
            if self.use_crf:
                return self.crf.decode(logits, mask=mask)
            predictions = torch.argmax(logits, dim=-1)
            if mask is not None:
                predictions = predictions.masked_fill(~mask, -100)
            return predictions.tolist()

    def predict(self, input_ids, attention_mask=None):
        with torch.no_grad():
            sequence_output, _, _ = self._encode_sequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )

            logits = self.classifier(sequence_output)

            if self.use_crf:
                # Use CRF decoding
                if attention_mask is not None:
                    mask = attention_mask.bool()
                    predictions = self.crf.decode(logits, mask=mask)
                else:
                    predictions = self.crf.decode(logits)
            else:
                # Use argmax decoding
                predictions = torch.argmax(logits, dim=-1)
                if attention_mask is not None:
                    # Mask out padded positions
                    predictions = predictions.masked_fill(~attention_mask.bool(), -100)
                predictions = predictions.tolist()

            return predictions


def context_window_plan(lengths: List[int], k: int, max_words: int):
    """For sentences of one document (given by their word counts) choose the context sentences of each one.

    Returns, per sentence, (previous_indices_in_reading_order, next_indices). The whole centre sentence is kept; the nearest
    context sentences (alternating previous/next) are added while they fit into `max_words`.
    """
    plan = []
    for pos in range(len(lengths)):
        budget = max(0, max_words - lengths[pos])
        prev_idx, next_idx = [], []
        for step in range(1, k + 1):
            for side in (-1, 1):
                j = pos + side * step
                if 0 <= j < len(lengths) and lengths[j] <= budget:
                    budget -= lengths[j]
                    (prev_idx if side < 0 else next_idx).append(j)
        plan.append((sorted(prev_idx), sorted(next_idx)))
    return plan


def build_word_start_ids(tokenizer) -> torch.Tensor:
    """Ids of vocabulary pieces that start a whitespace-delimited word.

    Used to evaluate at word level: only the first subword of each word is scored.
    SentencePiece marks word starts with '▁', WordPiece marks continuations with '##'.
    """
    vocab = tokenizer.get_vocab()
    n_sentencepiece = sum(1 for piece in vocab if piece.startswith("\u2581"))
    n_wordpiece = sum(1 for piece in vocab if piece.startswith("##"))
    n_bpe = sum(1 for piece in vocab if piece.startswith("\u0120"))
    if n_bpe > max(n_sentencepiece, n_wordpiece):  # byte-level BPE (RoBERTa, GPT-2 style)
        ids = [i for piece, i in vocab.items() if piece.startswith("\u0120")]
    elif n_sentencepiece >= n_wordpiece:
        ids = [i for piece, i in vocab.items() if piece.startswith("\u2581")]
    else:
        ids = [i for piece, i in vocab.items() if not piece.startswith("##")]
    return torch.tensor(sorted(ids), dtype=torch.long)


class CRFTrainerWithLLRD(Trainer):
    """Custom Trainer with Layer-wise Learning Rate Decay (LLRD) and advanced scheduling.

    If `word_start_ids` is given, evaluation is word-level: labels of non-first subwords are
    masked (-100) in `prediction_step`, so every metric is computed over words, not subwords.
    """

    def __init__(self, *args, word_start_ids: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_start_ids = word_start_ids

    def _build_weighted_sampler(self):
        if self.train_dataset is None:
            return None

        mode = str(getattr(self.args, "weighted_sampler_mode", "corpus_binary"))
        max_weight = float(getattr(self.args, "weighted_sampler_max_weight", 5.0))
        o_label_id = LABEL2ID["O"]

        key_counts = Counter()
        keys = []

        for idx in range(len(self.train_dataset)):
            row = self.train_dataset[idx]
            labels = row.get("labels", [])
            valid_labels = [int(l) for l in labels if int(l) != -100]
            has_entity = any(label_id != o_label_id for label_id in valid_labels)
            binary_group = "pos" if has_entity else "neg"
            corpus_group = str(row.get("corpus", "UNK"))

            if mode == "binary":
                key = binary_group
            elif mode == "corpus":
                key = corpus_group
            else:
                key = f"{corpus_group}::{binary_group}"

            keys.append(key)
            key_counts[key] += 1

        if not key_counts:
            return None

        raw_weights = [1.0 / key_counts[key] for key in keys]
        mean_weight = float(np.mean(raw_weights)) if raw_weights else 1.0
        norm_weights = [(w / mean_weight) if mean_weight > 0 else 1.0 for w in raw_weights]
        clipped_weights = [min(max_weight, w) for w in norm_weights]

        seed = int(getattr(self.args, "weighted_sampler_seed", 42))
        generator = torch.Generator()
        generator.manual_seed(seed)

        print(
            f"[WeightedSampler] mode={mode}, groups={len(key_counts)}, "
            f"max_group={max(key_counts.values())}, min_group={min(key_counts.values())}, "
            f"max_weight={max(clipped_weights):.3f}"
        )

        return WeightedRandomSampler(
            weights=torch.tensor(clipped_weights, dtype=torch.double),
            num_samples=len(clipped_weights),
            replacement=True,
            generator=generator,
        )

    def get_train_dataloader(self):
        use_sampler = bool(getattr(self.args, "use_weighted_sampler", False))
        if not use_sampler:
            return super().get_train_dataloader()

        if getattr(self.args, "world_size", 1) > 1:
            print("[WeightedSampler] Distributed training detected; fallback to default sampler.")
            return super().get_train_dataloader()

        sampler = self._build_weighted_sampler()
        if sampler is None:
            return super().get_train_dataloader()

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Match default Trainer behavior: drop columns that are not consumed by model.forward.
        if isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def create_optimizer(self):
        """
        Create optimizer with Layer-wise Learning Rate Decay (LLRD).
        """
        if self.optimizer is None:
            use_llrd = getattr(self.args, 'use_llrd', False)
            model_has_bert = bool(getattr(self.model, "use_bert_encoder", False) and hasattr(self.model, "bert"))

            if use_llrd and model_has_bert:
                self._create_llrd_optimizer()
            else:
                if use_llrd and not model_has_bert:
                    print("[LLRD] use_llrd=True but no BERT encoder detected; falling back to standard optimizer.")
                # Fallback to standard differentiated LR
                self._create_standard_optimizer()

        return self.optimizer

    def _create_llrd_optimizer(self):
        """Create optimizer with layer-wise learning rate decay."""

        base_lr = getattr(self.args, 'bert_base_learning_rate', 2e-5)
        classifier_lr = getattr(self.args, 'classifier_learning_rate', 1e-4)
        crf_lr = getattr(self.args, 'crf_learning_rate', 1e-3)
        bilstm_lr = getattr(self.args, 'bilstm_learning_rate', 5e-4)
        decay_rate = getattr(self.args, 'llrd_decay_rate', 0.95)
        weight_decay = self.args.weight_decay

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias"]

        num_layers = self.model.bert.config.num_hidden_layers

        optimizer_grouped_parameters = []

        print("\n" + "=" * 100)
        print("LAYER-WISE LEARNING RATE DECAY (LLRD) CONFIGURATION")
        print("=" * 100)

        print(f"\nBERT Encoder Layers (with LLRD, decay_rate={decay_rate}):")
        print("-" * 100)

        for layer_num in range(num_layers):
            layer_lr = base_lr * (decay_rate ** (num_layers - layer_num - 1))

            optimizer_grouped_parameters.append({
                "params": [
                    p for n, p in self.model.bert.encoder.layer[layer_num].named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
                "lr": layer_lr,
                "name": f"bert.encoder.layer.{layer_num}_wd"
            })

            # Parameters without weight decay
            optimizer_grouped_parameters.append({
                "params": [
                    p for n, p in self.model.bert.encoder.layer[layer_num].named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": layer_lr,
                "name": f"bert.encoder.layer.{layer_num}_no_wd"
            })

            # Count parameters in this layer
            layer_params = sum(
                p.numel() for p in self.model.bert.encoder.layer[layer_num].parameters()
                if p.requires_grad
            )
            print(f"  Layer {layer_num:2d}: LR = {layer_lr:.2e} | Parameters: {layer_params:,}")

        embeddings_lr = base_lr * (decay_rate ** num_layers)
        print(f"\nBERT Embeddings:")
        print("-" * 100)
        print(f"  LR = {embeddings_lr:.2e}")

        optimizer_grouped_parameters.extend([
            {
                "params": [
                    p for n, p in self.model.bert.embeddings.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
                "lr": embeddings_lr,
                "name": "bert.embeddings_wd"
            },
            {
                "params": [
                    p for n, p in self.model.bert.embeddings.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": embeddings_lr,
                "name": "bert.embeddings_no_wd"
            }
        ])

        embeddings_params = sum(
            p.numel() for p in self.model.bert.embeddings.parameters()
            if p.requires_grad
        )
        print(f"  Parameters: {embeddings_params:,}")

        if hasattr(self.model.bert, 'pooler') and self.model.bert.pooler is not None:
            pooler_lr = base_lr
            print(f"\nBERT Pooler:")
            print("-" * 100)
            print(f"  LR = {pooler_lr:.2e}")

            optimizer_grouped_parameters.extend([
                {
                    "params": [
                        p for n, p in self.model.bert.pooler.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                    "lr": pooler_lr,
                    "name": "bert.pooler_wd"
                },
                {
                    "params": [
                        p for n, p in self.model.bert.pooler.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": pooler_lr,
                    "name": "bert.pooler_no_wd"
                }
            ])

            pooler_params = sum(
                p.numel() for p in self.model.bert.pooler.parameters()
                if p.requires_grad
            )
            print(f"  Parameters: {pooler_params:,}")

        bilstm_params = 0
        if hasattr(self.model, 'bilstm') and self.model.use_bilstm:
            print(f"\nBiLSTM Layer:")
            print("-" * 100)
            print(f"  LR = {bilstm_lr:.2e}")

            optimizer_grouped_parameters.extend([
                {
                    "params": [
                        p for n, p in self.model.bilstm.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                    "lr": bilstm_lr,
                    "name": "bilstm_wd"
                },
                {
                    "params": [
                        p for n, p in self.model.bilstm.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": bilstm_lr,
                    "name": "bilstm_no_wd"
                }
            ])

            bilstm_params = sum(
                p.numel() for p in self.model.bilstm.parameters()
                if p.requires_grad
            )
            print(f"  Parameters: {bilstm_params:,}")

        # === Classifier Head ===
        print(f"\nClassifier Head:")
        print("-" * 100)
        print(f"  LR = {classifier_lr:.2e}")

        optimizer_grouped_parameters.extend([
            {
                "params": [
                    p for n, p in self.model.classifier.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
                "lr": classifier_lr,
                "name": "classifier_wd"
            },
            {
                "params": [
                    p for n, p in self.model.classifier.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": classifier_lr,
                "name": "classifier_no_wd"
            }
        ])

        classifier_params = sum(
            p.numel() for p in self.model.classifier.parameters()
            if p.requires_grad
        )
        print(f"  Parameters: {classifier_params:,}")

        # === CRF Layer ===
        crf_params = 0
        if hasattr(self.model, 'crf') and self.model.use_crf:
            print(f"\nCRF Layer:")
            print("-" * 100)
            print(f"  LR = {crf_lr:.2e}")

            optimizer_grouped_parameters.append({
                "params": [p for p in self.model.crf.parameters() if p.requires_grad],
                "weight_decay": 0.0,  # No weight decay for CRF
                "lr": crf_lr,
                "name": "crf"
            })

            crf_params = sum(p.numel() for p in self.model.crf.parameters() if p.requires_grad)
            print(f"  Parameters: {crf_params:,}")

        # === Summary ===
        bert_params = embeddings_params + sum(
            p.numel() for layer in self.model.bert.encoder.layer
            for p in layer.parameters() if p.requires_grad
        )
        if hasattr(self.model.bert, 'pooler') and self.model.bert.pooler is not None:
            bert_params += sum(
                p.numel() for p in self.model.bert.pooler.parameters()
                if p.requires_grad
            )

        # Catch-all: encoder parameters outside `encoder.layer`/`embeddings`/`pooler` (e.g. DeBERTa relative-position
        # embeddings) would otherwise never be optimised.
        assigned = {id(p) for group in optimizer_grouped_parameters for p in group["params"]}
        leftover_decay = [p for n, p in self.model.named_parameters()
                          if id(p) not in assigned and p.requires_grad and not any(nd in n for nd in no_decay)]
        leftover_no_decay = [p for n, p in self.model.named_parameters()
                             if id(p) not in assigned and p.requires_grad and any(nd in n for nd in no_decay)]
        if leftover_decay or leftover_no_decay:
            print(f"[LLRD] {sum(p.numel() for p in leftover_decay + leftover_no_decay):,} extra parameters "
                  f"outside the layer groups get the base learning rate")
            optimizer_grouped_parameters.append({"params": leftover_decay, "weight_decay": weight_decay,
                                                 "lr": base_lr, "name": "other_wd"})
            optimizer_grouped_parameters.append({"params": leftover_no_decay, "weight_decay": 0.0,
                                                 "lr": base_lr, "name": "other_no_wd"})

        total_params = bert_params + bilstm_params + classifier_params + crf_params

        print(f"\n" + "=" * 100)
        print("PARAMETER SUMMARY:")
        print("=" * 100)
        print(f"BERT (all layers):     {bert_params:>12,} params | LR range: {embeddings_lr:.2e} - {base_lr:.2e}")
        if bilstm_params > 0:
            print(f"BiLSTM:                {bilstm_params:>12,} params | LR: {bilstm_lr:.2e}")
        print(f"Classifier:            {classifier_params:>12,} params | LR: {classifier_lr:.2e}")
        if crf_params > 0:
            print(f"CRF:                   {crf_params:>12,} params | LR: {crf_lr:.2e}")
        print(f"{'=' * 100}")
        print(f"TOTAL:                 {total_params:>12,} params")
        print(f"{'=' * 100}\n")

        # Create optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

    def _create_standard_optimizer(self):
        """Fallback to standard optimizer with uniform learning rates."""
        bert_lr = getattr(self.args, 'bert_base_learning_rate', 1e-5)
        embedding_lr = getattr(self.args, 'embedding_learning_rate', 5e-4)
        classifier_lr = getattr(self.args, 'classifier_learning_rate', 5e-5)
        crf_lr = getattr(self.args, 'crf_learning_rate', 1e-3)
        bilstm_lr = getattr(self.args, 'bilstm_learning_rate', 5e-4)
        weight_decay = self.args.weight_decay

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_grouped_parameters = []

        if hasattr(self.model, "bert") and bool(getattr(self.model, "use_bert_encoder", False)):
            optimizer_grouped_parameters.extend([
                # BERT parameters with weight decay
                {
                    "params": [
                        p for n, p in self.model.bert.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                    "lr": bert_lr,
                },
                # BERT parameters without weight decay
                {
                    "params": [
                        p for n, p in self.model.bert.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": bert_lr,
                },
            ])

        if hasattr(self.model, "token_embeddings") and not bool(getattr(self.model, "use_bert_encoder", True)):
            optimizer_grouped_parameters.append({
                "params": [p for p in self.model.token_embeddings.parameters() if p.requires_grad],
                "weight_decay": weight_decay,
                "lr": embedding_lr,
            })

        # Classifier parameters
        optimizer_grouped_parameters.append({
            "params": [p for p in self.model.classifier.parameters() if p.requires_grad],
            "weight_decay": weight_decay,
            "lr": classifier_lr,
        })

        # Add BiLSTM parameters if present
        if hasattr(self.model, 'bilstm') and self.model.use_bilstm:
            optimizer_grouped_parameters.append({
                "params": [p for p in self.model.bilstm.parameters() if p.requires_grad],
                "weight_decay": weight_decay,
                "lr": bilstm_lr,
            })

        # Add CRF parameters if present
        if hasattr(self.model, 'crf') and self.model.use_crf:
            optimizer_grouped_parameters.append({
                "params": [p for p in self.model.crf.parameters() if p.requires_grad],
                "weight_decay": 0.0,
                "lr": crf_lr,
            })

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        print("\n" + "=" * 80)
        print("STANDARD DIFFERENTIATED LEARNING RATES")
        print("=" * 80)
        if hasattr(self.model, "bert") and bool(getattr(self.model, "use_bert_encoder", False)):
            print(f"BERT:       {bert_lr:.2e}")
        if hasattr(self.model, "token_embeddings") and not bool(getattr(self.model, "use_bert_encoder", True)):
            print(f"Embedding:  {embedding_lr:.2e}")
        if hasattr(self.model, 'bilstm') and self.model.use_bilstm:
            print(f"BiLSTM:     {bilstm_lr:.2e}")
        print(f"Classifier: {classifier_lr:.2e}")
        if hasattr(self.model, 'crf') and self.model.use_crf:
            print(f"CRF:        {crf_lr:.2e}")
        print("=" * 80 + "\n")

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Create learning rate scheduler with warmup.
        """
        if self.lr_scheduler is None:
            warmup_steps = self.args.warmup_steps
            if warmup_steps is None:
                warmup_steps = self.args.get_warmup_steps(num_training_steps)

            print(f"\nLearning Rate Scheduler:")
            print(f"  Type: {self.args.lr_scheduler_type}")
            print(f"  Total training steps: {num_training_steps}")
            print(f"  Warmup steps: {warmup_steps} ({warmup_steps / num_training_steps * 100:.1f}%)")
            if self.args.lr_scheduler_type == "cosine_with_restarts":
                print(f"  Number of cycles: {getattr(self.args, 'num_cycles', 1)}")
            print()

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs={
                    "num_cycles": getattr(self.args, 'num_cycles', 1)
                } if self.args.lr_scheduler_type == "cosine_with_restarts" else {}
            )

        return self.lr_scheduler

    @staticmethod
    def _with_updated_eval_metrics(output, metrics: Dict):
        """Support both mutable and immutable EvalLoopOutput variants across transformers versions."""
        if hasattr(output, "_replace"):
            return output._replace(metrics=metrics)

        try:
            output.metrics = metrics
            return output
        except Exception:
            try:
                from dataclasses import replace as dataclass_replace
                return dataclass_replace(output, metrics=metrics)
            except Exception:
                return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to use CRF decoding."""
        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            if isinstance(outputs, dict):
                logits = outputs.get("logits")
                loss = outputs.get("loss")
            else:
                logits = outputs[1] if len(outputs) > 1 else outputs[0]
                loss = outputs[0] if len(outputs) > 1 else None

        if prediction_loss_only:
            return (loss, None, None)

        # Use CRF decoding for predictions
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        if hasattr(model, "module"):
            predictions = model.module.predict(input_ids, attention_mask)
        else:
            predictions = model.predict(input_ids, attention_mask)

        batch_size = logits.shape[0]
        seq_length = logits.shape[1]

        padded_predictions = torch.full(
            (batch_size, seq_length),
            fill_value=-100,
            dtype=torch.long,
            device=logits.device,
        )

        for i, pred_seq in enumerate(predictions):
            pred_len = min(len(pred_seq), seq_length)
            padded_predictions[i, :pred_len] = torch.tensor(
                pred_seq[:pred_len], dtype=torch.long, device=logits.device
            )

        labels = inputs.get("labels")
        if self.word_start_ids is not None and labels is not None:
            is_word_start = torch.isin(input_ids, self.word_start_ids.to(input_ids.device))
            labels = labels.masked_fill(~is_word_start, -100)
        return (loss, padded_predictions, labels)

    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only=None,
            ignore_keys=None,
            metric_key_prefix="eval",
    ):
        """Add per-corpus metrics and corpus-macro F1 to the default evaluation loop output."""
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        metrics = dict(output.metrics)
        base_f1 = float(metrics.get(f"{metric_key_prefix}_f1", 0.0))

        if output.predictions is None or output.label_ids is None:
            metrics[f"{metric_key_prefix}_macro_f1_by_corpus"] = base_f1
            return self._with_updated_eval_metrics(output, metrics)

        dataset = getattr(dataloader, "dataset", None)

        # HF Trainer can drop non-model columns (like `corpus`) in the eval dataloader.
        # Fallback to the original eval dataset to recover corpus mapping.
        has_corpus_in_loader = (
            dataset is not None
            and hasattr(dataset, "column_names")
            and "corpus" in dataset.column_names
        )
        if not has_corpus_in_loader:
            fallback_eval_dataset = getattr(self, "eval_dataset", None)
            if (
                fallback_eval_dataset is not None
                and hasattr(fallback_eval_dataset, "column_names")
                and "corpus" in fallback_eval_dataset.column_names
                and len(fallback_eval_dataset) == len(output.label_ids)
            ):
                dataset = fallback_eval_dataset
            else:
                metrics[f"{metric_key_prefix}_macro_f1_by_corpus"] = base_f1
                return self._with_updated_eval_metrics(output, metrics)

        corpus_values = [str(x) for x in dataset["corpus"]]
        if len(corpus_values) != len(output.label_ids):
            metrics[f"{metric_key_prefix}_macro_f1_by_corpus"] = base_f1
            return self._with_updated_eval_metrics(output, metrics)

        unique_corpora = sorted(set(corpus_values))
        corpus_f1_scores = []

        for corpus_name in unique_corpora:
            indices = [i for i, value in enumerate(corpus_values) if value == corpus_name]
            if not indices:
                continue

            subset_predictions = output.predictions[indices]
            subset_labels = output.label_ids[indices]
            subset_metrics = compute_metrics((subset_predictions, subset_labels))

            corpus_key = re.sub(r"[^a-zA-Z0-9]+", "_", corpus_name.strip().lower()).strip("_") or "unknown"
            for key, value in subset_metrics.items():
                if isinstance(value, (np.floating, np.integer)):
                    metrics[f"{metric_key_prefix}_{corpus_key}_{key}"] = value.item()
                else:
                    metrics[f"{metric_key_prefix}_{corpus_key}_{key}"] = value

            corpus_f1_scores.append(float(subset_metrics.get("f1", 0.0)))

        metrics[f"{metric_key_prefix}_macro_f1_by_corpus"] = (
            float(np.mean(corpus_f1_scores)) if corpus_f1_scores else base_f1
        )

        return self._with_updated_eval_metrics(output, metrics)


class ErrorCollectionCallback(TrainerCallback):
    """Callback to collect error examples after each validation epoch."""

    def __init__(self, tokenizer, val_dataset, output_dir: Path):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.epoch = 0
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Store reference to trainer at the beginning of training."""
        # Trainer is not passed in kwargs during on_train_begin,
        # so we'll rely on setting it manually after trainer creation

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Called after each validation evaluation."""
        # Only collect errors after evaluation (not during training start)
        if state.epoch is None or state.epoch == 0:
            return

        self.epoch = int(state.epoch)

        # Try to get trainer from different sources
        trainer = kwargs.get("trainer")
        if trainer is None and hasattr(self, 'trainer') and self.trainer is not None:
            trainer = self.trainer
        if trainer is None:
            print(f"Warning: Could not get trainer reference in ErrorCollectionCallback")
            return

        # Create epoch-specific output directory
        epoch_dir = self.output_dir / f"epoch_{self.epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCollecting error examples for epoch {self.epoch}...")
        # Collect errors for this epoch
        try:
            collect_error_examples(trainer, self.val_dataset, self.tokenizer, epoch_dir)
        except Exception as e:
            print(f"Error collecting error examples: {e}")
            import traceback
            traceback.print_exc()


class ValidationTextsCallback(TrainerCallback):
    """Callback to show text samples with predictions during validation."""

    def __init__(self, tokenizer, val_dataset, num_samples=5):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.num_samples = num_samples
        self.epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after each validation evaluation."""
        self.epoch += 1
        print(f"\n{'=' * 80}")
        print(f"VALIDATION - EPOCH {self.epoch}")
        print(f"{'=' * 80}")

        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        import random

        indices = random.sample(
            range(len(self.val_dataset)), min(self.num_samples, len(self.val_dataset))
        )

        for i, idx in enumerate(indices):
            print(f"\n--- Example {i + 1}/{len(indices)} ---")

            example = self.val_dataset[idx]
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            labels = example["labels"]

            # Make prediction with CRF
            with torch.no_grad():
                if hasattr(trainer.model, "module"):
                    predictions = trainer.model.module.predict(
                        torch.tensor([input_ids]).to(trainer.model.device),
                        torch.tensor([attention_mask]).to(trainer.model.device),
                    )[0]
                else:
                    predictions = trainer.model.predict(
                        torch.tensor([input_ids]).to(trainer.model.device),
                        torch.tensor([attention_mask]).to(trainer.model.device),
                    )[0]

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

            current_entity = []
            current_tokens = []
            result_lines = []

            min_len = min(len(tokens), len(predictions), len(labels))

            for j in range(min_len):
                token = tokens[j]
                pred_id = predictions[j] if j < len(predictions) else 0
                label_id = labels[j] if j < len(labels) else -100

                if label_id == -100:
                    continue

                pred_label = ID2LABEL[pred_id]
                true_label = ID2LABEL[label_id]

                if pred_label.startswith("B-") or pred_label == "O":
                    if current_entity:
                        entity_text = self.tokenizer.convert_tokens_to_string(current_tokens)
                        status = "✓" if current_entity[0] == true_label else "✗"
                        result_lines.append(
                            f"  Predicted: {current_entity[0]} -> '{entity_text}' {status}"
                        )
                        current_entity = []
                        current_tokens = []

                    if pred_label != "O":
                        if pred_label.startswith("B-"):
                            current_entity = [pred_label]
                            current_tokens = [token]
                elif pred_label.startswith("I-") and current_entity:
                    current_tokens.append(token)
                else:
                    current_tokens = [token]
                    current_entity = ["O"]

            print(f"Text: {text}")
            print("Markup:")
            for line in result_lines:
                print(line)

            pred_entities = sum(1 for p in predictions if ID2LABEL[p] != "O")
            true_entities = sum(1 for l in labels if ID2LABEL[l] != "O" and l != -100)
            print(f"Predicted entities: {pred_entities}, True entities: {true_entities}")

        print(f"{'=' * 80}\n")


class TensorBoardCallback(TrainerCallback):
    """Custom callback for TensorBoard logging with learning rate tracking."""

    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.global_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, self.global_step)

            if 'trainer' in kwargs and hasattr(kwargs['trainer'], 'optimizer'):
                optimizer = kwargs['trainer'].optimizer
                for i, param_group in enumerate(optimizer.param_groups):
                    lr = param_group['lr']
                    group_name = param_group.get('name', f'group_{i}')
                    self.writer.add_scalar(f'learning_rate/{group_name}', lr, self.global_step)

            self.global_step += 1

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback with configurable patience and threshold."""

    def __init__(self, patience=5, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.wait = 0
        self.best_epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        current_metric = metrics.get(args.metric_for_best_model, 0)
        current_epoch = state.epoch

        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_epoch = current_epoch
            print(f"\n[Early Stopping] Initial best metric: {current_metric:.4f}")
        elif current_metric > self.best_metric + self.threshold:
            improvement = current_metric - self.best_metric
            self.best_metric = current_metric
            self.best_epoch = current_epoch
            self.wait = 0
            print(f"\n[Early Stopping] New best metric: {current_metric:.4f} (improved by {improvement:.4f})")
        else:
            self.wait += 1
            print(
                f"\n[Early Stopping] No improvement for {self.wait} epoch(s). Best: {self.best_metric:.4f} (epoch {self.best_epoch})")

            if self.wait >= self.patience:
                control.should_training_stop = True
                print(f"\n[Early Stopping] Stopping training! No improvement for {self.patience} epochs.")
                print(f"[Early Stopping] Best metric: {self.best_metric:.4f} at epoch {self.best_epoch}")


def create_experiment_dir(base_dir: Path, experiment_name: str = None) -> Path:
    """Create experiment directory with timestamp."""
    if experiment_name:
        exp_dir = base_dir / experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base_dir / f"exp_{timestamp}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_config(output_dir: Path, config: Dict):
    """Save experiment configuration."""
    config_path = output_dir / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> List[Dict]:
    """Read JSON data from file (supports both JSON array and JSONL format)."""
    text = path.read_text(encoding="utf-8").strip()

    # Try to parse as JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        else:
            raise ValueError("Dataset must be a JSON array")
    except json.JSONDecodeError:
        # If fails, try JSONL format (one JSON object per line)
        data = []
        for line_num, line in enumerate(text.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

        if not data:
            raise ValueError("Could not parse JSON data (tried both JSON array and JSONL formats)")

        print(f"Loaded {len(data)} examples from JSONL format")
        return data


def read_json_any(path: Path) -> List[Dict]:
    """Read JSON array or JSONL; fallback to read_json."""
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Fallback to JSONL
    data = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except Exception:
            continue
    return data


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace in string."""
    return re.sub(r"\s+", " ", s).strip()


def find_definition_span(text: str, definition: str) -> Optional[Tuple[int, int]]:
    """Find definition span in text."""
    idx = text.find(definition)
    if idx != -1:
        return idx, idx + len(definition)

    ntext = normalize_whitespace(text)
    ndef = normalize_whitespace(definition)
    idx = ntext.find(ndef)

    if idx != -1:
        snippet = ndef[: min(30, len(ndef))]
        idx2 = text.find(snippet)
        if idx2 != -1:
            idx_start = idx2
            idx_end = idx_start + len(definition)
            return idx_start, idx_end

    idx = text.lower().find(definition.lower())
    if idx != -1:
        return idx, idx + len(definition)

    def_tokens = definition.split()
    for w in range(min(8, len(def_tokens)), 0, -1):
        window = " ".join(def_tokens[:w])
        idx = text.find(window)
        if idx != -1:
            return idx, idx + len(window)

    return None


def find_term_span(text: str, term: str, def_span: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Locate a term, preferring an occurrence outside the definition span.

    A term often re-occurs inside its own definition; taking the first occurrence would then
    label the wrong mention and lose the term tokens (the definition has priority).
    """
    if def_span is not None and term:
        start = 0
        while True:
            idx = text.find(term, start)
            if idx == -1:
                break
            end = idx + len(term)
            if end <= def_span[0] or idx >= def_span[1]:
                return idx, end
            start = idx + 1
    return find_definition_span(text, term)


def build_examples(raw: List[Dict]) -> List[Dict]:
    """Build examples from raw data with support for term-only examples."""
    examples = []
    skipped_no_text = 0
    skipped_no_definition = 0
    skipped_def_alignment = 0
    skipped_term_alignment = 0
    term_only_added = 0

    use_term_only = TRAINING_CONFIG.get("use_term_only_examples", True)

    for obj in raw:
        text = obj.get("text", "").strip()
        if not text:
            skipped_no_text += 1
            continue

        corpus = str(obj.get("corpus", "UNK"))
        language = str(obj.get("language", "unk"))
        if TRAINING_CONFIG.get("corpus_tag"):
            text = f"[{corpus}] {text}"

        def_span = None
        term_span = None

        # Process definitions
        if obj.get("has_definition"):
            definition = obj.get("definition")
            if not definition:
                skipped_no_definition += 1
                # Don't skip - there might be a term
            else:
                def_span = find_definition_span(text, definition)
                if def_span is None:
                    print(f"[WARN] Could not align definition; Text: {text[:80]}")
                    skipped_def_alignment += 1
                    # Don't skip - there might be a term

        # Process terms independently of definitions
        term = obj.get("term")
        if term:
            term_span = find_term_span(text, term, def_span)
            if term_span is None:
                print(f"[WARN] Could not align term '{term}'; Text: {text[:80]}")
                skipped_term_alignment += 1
            elif def_span is None and use_term_only:
                # This is a term-only example (no definition)
                term_only_added += 1

        # Add example if it has definition, term, or is a negative example
        if use_term_only:
            # Include all examples (with def, with term, or negative)
            examples.append({
                "text": text,
                "def_span": def_span,
                "term_span": term_span,
                "corpus": corpus,
                "language": language,
                "group": str(obj.get("group", "")),
            })
        else:
            # Original behavior: only include examples with definitions or negative examples
            if def_span is not None or (def_span is None and term_span is None):
                examples.append({
                    "text": text,
                    "def_span": def_span,
                    "term_span": None if def_span is None else term_span,
                    "corpus": corpus,
                    "language": language,
                    "group": str(obj.get("group", "")),
                })

    print(f"Built {len(examples)} examples")
    if skipped_no_text > 0:
        print(f"  Skipped (no text): {skipped_no_text}")
    if skipped_no_definition > 0:
        print(f"  Skipped (no definition when expected): {skipped_no_definition}")
    if skipped_def_alignment > 0:
        print(f"  Skipped (definition alignment failed): {skipped_def_alignment}")
    if skipped_term_alignment > 0:
        print(f"  Skipped (term alignment failed): {skipped_term_alignment}")
    if use_term_only and term_only_added > 0:
        print(f"  ✨ Added (term only, no definition): {term_only_added}")

    pos_examples = [ex for ex in examples if ex["def_span"] is not None]
    neg_examples = [ex for ex in examples if ex["def_span"] is None and ex["term_span"] is None]
    term_examples = [ex for ex in examples if ex["term_span"] is not None]
    term_only_examples = [ex for ex in examples if ex["def_span"] is None and ex["term_span"] is not None]

    print(f"Examples with definitions: {len(pos_examples)}")
    print(f"Examples with terms: {len(term_examples)}")
    if use_term_only and term_only_examples:
        print(f"  - Terms with definition: {len([ex for ex in examples if ex['def_span'] is not None and ex['term_span'] is not None])}")
        print(f"  - Terms WITHOUT definition: {len(term_only_examples)} ✨")

    # Oversampling (optional, controlled via TRAINING_CONFIG)
    use_oversampling = TRAINING_CONFIG.get("use_oversampling", False)

    if use_oversampling and pos_examples:
        pos_count = len(pos_examples)
        neg_count = len(neg_examples)

        if pos_count < neg_count:
            max_repeat = TRAINING_CONFIG.get("oversampling_max_repeat", 2)
            repeat_factor = min(max_repeat, max(1, neg_count // max(1, pos_count) - 1))

            if repeat_factor > 0:
                oversampled_pos = pos_examples * repeat_factor
                examples = examples + oversampled_pos
                print(
                    f"[Oversampling] ENABLED: positives: {pos_count} -> {len(pos_examples) * (repeat_factor + 1)} "
                    f"(repeat_factor={repeat_factor}), negatives: {neg_count}, total: {len(examples)}"
                )
            else:
                print(f"[Oversampling] ENABLED but no oversampling needed (balanced classes)")
        else:
            print(f"[Oversampling] ENABLED but no oversampling needed (positives >= negatives)")
    else:
        print(f"[Oversampling] DISABLED: Using original class distribution")

    print(f"[Final] Positive examples: {len([ex for ex in examples if ex['def_span'] is not None])}, "
          f"Negative examples: {len([ex for ex in examples if ex['def_span'] is None])}, "
          f"Total: {len(examples)}")

    return examples


def tokenize_and_align_labels(examples: Dict, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
    """Tokenize texts and align labels for both definitions and terms."""
    tokenized_inputs = tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )

    labels_all = []
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        word_labels = ["O"] * len(offsets)

        # Label definition tokens
        def_span = examples["def_span"][i]
        if def_span is not None:
            s, e = def_span
            first = True
            for j, (off_start, off_end) in enumerate(offsets):
                if off_start == off_end == 0:
                    continue
                if off_end <= s or off_start >= e:
                    continue

                if first:
                    word_labels[j] = "B-DEF"
                    first = False
                else:
                    word_labels[j] = "I-DEF"

        # Label term tokens
        term_span = examples["term_span"][i]
        if term_span is not None:
            s, e = term_span
            first = True
            for j, (off_start, off_end) in enumerate(offsets):
                if off_start == off_end == 0:
                    continue
                if off_end <= s or off_start >= e:
                    continue

                # Skip tokens already labeled as part of the definition
                if word_labels[j] == "O":
                    if first:
                        word_labels[j] = "B-TERM"
                        first = False
                    else:
                        word_labels[j] = "I-TERM"
                else:
                    # Token already labeled DEF: skip
                    # (the definition has priority when term and definition overlap)
                    pass

        label_ids = []
        for j, lab in enumerate(word_labels):
            off_start, off_end = offsets[j]
            if off_start == off_end == 0:
                label_ids.append(-100)
            else:
                label_ids.append(LABEL2ID[lab])

        labels_all.append(label_ids)

    tokenized_inputs.pop("offset_mapping", None)
    tokenized_inputs["labels"] = labels_all

    return tokenized_inputs


def infer_token_class_weights_from_train_split(train_dataset: Dataset) -> Dict[str, float]:
    """Infer O/entity token weights from train split only (ignores padding label -100)."""
    o_label_id = LABEL2ID["O"]

    o_count = 0
    entity_count = 0

    for i in range(len(train_dataset)):
        labels = train_dataset[i].get("labels", [])
        for label_id in labels:
            label_id = int(label_id)
            if label_id == -100:
                continue
            if label_id == o_label_id:
                o_count += 1
            else:
                entity_count += 1

    if entity_count == 0 or o_count == 0:
        return {"O": 1.0, "ENTITY": 1.0}

    ratio = o_count / entity_count
    power = float(TRAINING_CONFIG.get("token_class_weight_power", 0.5))
    max_w = float(TRAINING_CONFIG.get("token_class_weight_max", 6.0))

    entity_weight = float(min(max_w, max(1.0, ratio ** power)))
    return {"O": 1.0, "ENTITY": entity_weight}


def prepare_dataset(
        examples: List[Dict], tokenizer: PreTrainedTokenizerBase, split_ratio: float = 0.1,
        fold_indices: Tuple[List[int], List[int]] = None,
        val_examples: Optional[List[Dict]] = None
) -> DatasetDict:
    """Prepare dataset for training (supports cross-validation and explicit val set)."""
    ds = Dataset.from_list(examples)

    tokenized_train = ds.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, TRAINING_CONFIG["max_length"]),
        batched=True,
        remove_columns=["def_span", "term_span"],
    )

    max_len = TRAINING_CONFIG["max_length"]
    n_truncated = sum(1 for ids in tokenized_train["input_ids"] if len(ids) >= max_len)
    if n_truncated:
        print(f"[WARN] {n_truncated}/{len(tokenized_train)} examples reach max_length={max_len} "
              f"(labels beyond it are dropped)")

    if val_examples is not None:
        ds_val = Dataset.from_list(val_examples)
        tokenized_val = ds_val.map(
            lambda batch: tokenize_and_align_labels(batch, tokenizer, TRAINING_CONFIG["max_length"]),
            batched=True,
            remove_columns=["def_span", "term_span"],
        )
        return DatasetDict({"train": tokenized_train, "test": tokenized_val})

    if fold_indices is not None:
        train_indices, test_indices = fold_indices
        train_dataset = tokenized_train.select(train_indices)
        test_dataset = tokenized_train.select(test_indices)
        return DatasetDict({"train": train_dataset, "test": test_dataset})
    else:
        groups = tokenized_train["group"] if "group" in tokenized_train.column_names else []
        if groups and all(groups):
            # Keep a sentence, its translation and duplicates on the same side of the split.
            from sklearn.model_selection import GroupShuffleSplit

            splitter = GroupShuffleSplit(n_splits=1, test_size=split_ratio, random_state=TRAINING_CONFIG["seed"])
            train_idx, val_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
            return DatasetDict({
                "train": tokenized_train.select(train_idx.tolist()),
                "test": tokenized_train.select(val_idx.tolist()),
            })
        print("[WARN] No 'group' field in the data: validation split may leak translations. "
              "Use the provided output/train.json, which has a group field.")
        return tokenized_train.train_test_split(test_size=split_ratio, seed=TRAINING_CONFIG["seed"])


def official_style_macro_f1(label_ids: np.ndarray, pred_ids: np.ndarray) -> float:
    """Macro F1 over all non-O labels, as in the SemEval-2020 Task 6 Subtask 2 scorer."""
    eval_ids = [i for label, i in LABEL2ID.items() if label != "O"]
    if len(label_ids) == 0:
        return 0.0
    _, _, f1, _ = precision_recall_fscore_support(
        label_ids, pred_ids, labels=eval_ids, average="macro", zero_division=0
    )
    return float(f1)


def compute_metrics(pred):
    """Compute evaluation metrics including entity-level and token-level F1."""
    predictions, labels = pred

    if hasattr(predictions, "cpu"):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    if predictions.dtype == np.int64 or predictions.dtype == np.int32:
        predictions_decoded = predictions
    else:
        predictions_decoded = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    all_token_preds = []
    all_token_labels = []

    for pred_row, label_row in zip(predictions_decoded, labels):
        preds = []
        labs = []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            pred_label = ID2LABEL[int(p)]
            true_label = ID2LABEL[int(l)]

            preds.append(pred_label)
            labs.append(true_label)

            all_token_preds.append(int(p))
            all_token_labels.append(int(l))

        true_predictions.append(preds)
        true_labels.append(labs)

    entity_results = metric.compute(
        predictions=true_predictions,
        references=true_labels,
        scheme="IOB2"
    )

    # Token-level metrics
    all_token_preds = np.array(all_token_preds)
    all_token_labels = np.array(all_token_labels)

    # Token accuracy
    token_accuracy = (all_token_preds == all_token_labels).mean()

    # Weighted metrics (by class frequency)
    token_precision, token_recall, token_f1, token_support = precision_recall_fscore_support(
        all_token_labels,
        all_token_preds,
        average='weighted',
        zero_division=0
    )

    # Macro-averaged metrics (equal weight for each class)
    token_precision_macro, token_recall_macro, token_f1_macro, _ = precision_recall_fscore_support(
        all_token_labels,
        all_token_preds,
        average='macro',
        zero_division=0
    )

    # Per-class metrics for each token label
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_token_labels,
        all_token_preds,
        average=None,
        zero_division=0,
        labels=list(LABEL2ID.values())
    )

    # Binary classification for DEF vs non-DEF
    def_labels = (all_token_labels != LABEL2ID["O"]).astype(int)
    def_preds = (all_token_preds != LABEL2ID["O"]).astype(int)

    def_precision, def_recall, def_f1, _ = precision_recall_fscore_support(
        def_labels,
        def_preds,
        average='binary',
        zero_division=0
    )

    # Entity token metrics excluding O (micro over entity tags only)
    entity_label_ids = [label_id for label_name, label_id in LABEL2ID.items() if label_name != "O"]
    entity_token_precision, entity_token_recall, entity_token_f1, _ = precision_recall_fscore_support(
        all_token_labels,
        all_token_preds,
        labels=entity_label_ids,
        average='micro',
        zero_division=0
    )

    # Binary classification for Definition vs non-Definition (B-DEF/I-DEF only)
    definition_label_ids = [
        LABEL2ID[label_name] for label_name in ("B-DEF", "I-DEF") if label_name in LABEL2ID
    ]
    definition_labels = np.isin(all_token_labels, definition_label_ids).astype(int)
    definition_preds = np.isin(all_token_preds, definition_label_ids).astype(int)

    definition_precision, definition_recall, definition_f1, _ = precision_recall_fscore_support(
        definition_labels,
        definition_preds,
        average='binary',
        zero_division=0
    )

    # Binary classification for TERM vs non-TERM
    # TERM includes B-TERM and I-TERM
    term_label_ids = [LABEL2ID.get("B-TERM", -1), LABEL2ID.get("I-TERM", -1)]
    term_labels = np.isin(all_token_labels, term_label_ids).astype(int)
    term_preds = np.isin(all_token_preds, term_label_ids).astype(int)

    term_precision, term_recall, term_f1, _ = precision_recall_fscore_support(
        term_labels,
        term_preds,
        average='binary',
        zero_division=0
    )

    # Build result dictionary
    result = {
        # Entity-level metrics (primary metrics)
        "precision": entity_results["overall_precision"],
        "recall": entity_results["overall_recall"],
        "f1": entity_results["overall_f1"],
        "accuracy": entity_results.get("overall_accuracy", 0.0),

        # Span-level F1 per entity type and their mean ("Entity Macro-F1" in the paper)
        "entity_def_f1": entity_results.get("DEF", {}).get("f1", 0.0),
        "entity_term_f1": entity_results.get("TERM", {}).get("f1", 0.0),
        "entity_macro_f1": (
            entity_results.get("DEF", {}).get("f1", 0.0) + entity_results.get("TERM", {}).get("f1", 0.0)
        ) / 2,

        # DEFT-official-style score: macro-average of per-label F1 over the B-/I- labels (O excluded),
        # computed over words when the trainer masks non-first subwords.
        "official_style_macro_f1": official_style_macro_f1(all_token_labels, all_token_preds),

        # Token-level metrics (weighted)
        "token_accuracy": token_accuracy,
        "token_f1_weighted": token_f1,
        "token_precision_weighted": token_precision,
        "token_recall_weighted": token_recall,

        # Token-level metrics (macro)
        "token_f1_macro": token_f1_macro,
        "token_precision_macro": token_precision_macro,
        "token_recall_macro": token_recall_macro,

        # Entity token metrics excluding O
        "entity_token_f1_no_o": entity_token_f1,
        "entity_token_precision_no_o": entity_token_precision,
        "entity_token_recall_no_o": entity_token_recall,

        # DEF class metrics
        "def_f1": def_f1,
        "def_precision": def_precision,
        "def_recall": def_recall,

        # Definition-only metrics (B-DEF/I-DEF)
        "definition_f1": definition_f1,
        "definition_precision": definition_precision,
        "definition_recall": definition_recall,

        # TERM class metrics
        "term_f1": term_f1,
        "term_precision": term_precision,
        "term_recall": term_recall,
    }

    # Add per-class metrics for each token label
    for label_name, label_id in LABEL2ID.items():
        idx = label_id
        result[f"{label_name}_precision"] = float(per_class_precision[idx])
        result[f"{label_name}_recall"] = float(per_class_recall[idx])
        result[f"{label_name}_f1"] = float(per_class_f1[idx])
        result[f"{label_name}_support"] = int(per_class_support[idx])

    return result


def evaluate_test_files(trainer, tokenizer, paths) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Word-level span evaluation on held-out files, overall and per language.

    Returns {file: {"all"|lang: {n_examples, entity_micro_{precision,recall,f1}, def_f1, term_f1,
    entity_macro_f1}}}. `trainer` must be a CRFTrainerWithLLRD built with `word_start_ids`.
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for path in paths:
        examples = build_examples(read_json_any(Path(path)))
        subsets = {"all": examples}
        for lang in sorted({ex["language"] for ex in examples if ex.get("language") not in (None, "unk")}):
            subsets[lang] = [ex for ex in examples if ex.get("language") == lang]
        results[str(path)] = {}
        for name, subset in subsets.items():
            dataset = prepare_dataset(subset, tokenizer, val_examples=subset)["test"]
            m = trainer.predict(dataset, metric_key_prefix="test").metrics
            results[str(path)][name] = {
                "n_examples": len(subset),
                "entity_micro_precision": m["test_precision"],
                "entity_micro_recall": m["test_recall"],
                "entity_micro_f1": m["test_f1"],
                "def_f1": m["test_entity_def_f1"],
                "term_f1": m["test_entity_term_f1"],
                "entity_macro_f1": m["test_entity_macro_f1"],
                "official_style_macro_f1": m["test_official_style_macro_f1"],
            }
            r = results[str(path)][name]
            print(f"{Path(path).name:18s} {name:4s} n={r['n_examples']:5d} "
                  f"microF1={r['entity_micro_f1']:.4f} macroF1={r['entity_macro_f1']:.4f} "
                  f"officialStyleF1={r['official_style_macro_f1']:.4f} "
                  f"(DEF {r['def_f1']:.4f} / TERM {r['term_f1']:.4f})")
    return results


def collect_error_examples(trainer, eval_dataset, tokenizer, output_dir: Path):
    """Collect examples where model made errors during validation."""
    print("\nCollecting error examples...")

    model = trainer.model
    model.eval()
    device = model.device

    error_examples = []
    error_type_counts = {"false_positive": 0, "false_negative": 0, "wrong_boundary": 0}

    for idx, example in enumerate(eval_dataset):
        input_ids = torch.tensor([example["input_ids"]]).to(device)
        attention_mask = torch.tensor([example["attention_mask"]]).to(device)
        labels = example["labels"]

        # Get predictions
        with torch.no_grad():
            if hasattr(model, "module"):
                predictions = model.module.predict(input_ids, attention_mask)[0]
            else:
                predictions = model.predict(input_ids, attention_mask)[0]

        # Check if there are any errors
        has_error = False
        for pred_id, label_id in zip(predictions, labels):
            if label_id != -100 and pred_id != label_id:
                has_error = True
                break

        if has_error:
            # Decode text and labels
            tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
            text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)

            # Collect token-level predictions and labels
            token_errors = []
            for j, (token, pred_id, label_id) in enumerate(zip(tokens, predictions, labels)):
                if label_id == -100:
                    continue

                pred_label = ID2LABEL[pred_id]
                true_label = ID2LABEL[label_id]

                if pred_id != label_id:
                    token_errors.append({
                        "position": j,
                        "token": token,
                        "predicted": pred_label,
                        "true": true_label
                    })

                    # Count error types
                    if true_label == "O" and pred_label != "O":
                        error_type_counts["false_positive"] += 1
                    elif true_label != "O" and pred_label == "O":
                        error_type_counts["false_negative"] += 1
                    elif true_label != "O" and pred_label != "O":
                        error_type_counts["wrong_boundary"] += 1

            # Extract predicted and true entities
            def extract_entities(token_list, label_list, label_ids):
                entities = []
                current_entity = []
                current_tokens = []
                current_label = None

                for token, label_id in zip(token_list, label_ids):
                    if label_id == -100:
                        continue

                    label = ID2LABEL[label_id]

                    if label.startswith("B-"):
                        if current_entity:
                            entity_text = tokenizer.convert_tokens_to_string(current_tokens)
                            entities.append({
                                "text": entity_text,
                                "label": current_label
                            })
                        current_entity = [label]
                        current_tokens = [token]
                        current_label = label.replace("B-", "")
                    elif label.startswith("I-") and current_entity:
                        current_tokens.append(token)
                    else:  # "O"
                        if current_entity:
                            entity_text = tokenizer.convert_tokens_to_string(current_tokens)
                            entities.append({
                                "text": entity_text,
                                "label": current_label
                            })
                            current_entity = []
                            current_tokens = []
                            current_label = None

                # Handle last entity
                if current_entity:
                    entity_text = tokenizer.convert_tokens_to_string(current_tokens)
                    entities.append({
                        "text": entity_text,
                        "label": current_label
                    })

                return entities

            predicted_entities = extract_entities(tokens, labels, predictions)
            true_entities = extract_entities(tokens, labels, labels)

            # Determine error type for this example
            error_type = "other"
            if len(true_entities) == 0 and len(predicted_entities) > 0:
                error_type = "false_positive_entity"
            elif len(true_entities) > 0 and len(predicted_entities) == 0:
                error_type = "false_negative_entity"
            elif len(true_entities) > 0 and len(predicted_entities) > 0:
                error_type = "boundary_or_partial_error"

            error_example = {
                "example_id": idx,
                "text": text,
                "predicted_entities": predicted_entities,
                "true_entities": true_entities,
                "token_errors": token_errors,
                "num_errors": len(token_errors),
                "error_type": error_type
            }

            error_examples.append(error_example)

    # Calculate additional statistics
    total_token_errors = sum(ex["num_errors"] for ex in error_examples)

    # Save error examples to JSON
    errors_file = output_dir / "model_errors.json"
    with open(errors_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(eval_dataset),
            "examples_with_errors": len(error_examples),
            "error_rate": len(error_examples) / len(eval_dataset) if len(eval_dataset) > 0 else 0,
            "total_token_errors": total_token_errors,
            "error_type_counts": error_type_counts,
            "errors": error_examples
        }, f, indent=2, ensure_ascii=False)

    print(
        f"Collected {len(error_examples)} examples with errors ({len(error_examples) / len(eval_dataset) * 100:.1f}%)")
    print(f"Total token-level errors: {total_token_errors}")
    print(f"Error type distribution:")
    print(f"  False positives: {error_type_counts['false_positive']}")
    print(f"  False negatives: {error_type_counts['false_negative']}")
    print(f"  Wrong boundaries: {error_type_counts['wrong_boundary']}")
    print(f"Error examples saved to: {errors_file}")

    return error_examples


def save_fold_predictions(trainer, val_dataset, examples: List[Dict], indices: List[int], path: Path) -> None:
    """Write word-level gold and predicted tags of a held-out fold (out-of-fold predictions)."""
    out = trainer.predict(val_dataset)
    with open(path, "w", encoding="utf-8") as f:
        for row, (p_row, l_row) in enumerate(zip(out.predictions, out.label_ids)):
            keep = l_row != -100
            ex = examples[indices[row]]
            f.write(json.dumps({
                "index": int(indices[row]),
                "corpus": ex["corpus"],
                "language": ex["language"],
                "group": ex.get("group", ""),
                "text": ex["text"],
                "gold_tags": [ID2LABEL[int(l)] for l in l_row[keep]],
                "pred_tags": [ID2LABEL[int(p)] if p >= 0 else "O" for p in p_row[keep]],
            }, ensure_ascii=False) + "\n")
    print(f"Saved out-of-fold predictions to {path}")


def train_single_fold(
        fold_num: int,
        train_indices: List[int],
        test_indices: List[int],
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        experiment_dir: Path,
        config: PretrainedConfig
) -> Dict:
    """Train a single fold in cross-validation."""

    print(f"\n{'=' * 100}")
    print(f"TRAINING FOLD {fold_num + 1}/{TRAINING_CONFIG['n_folds']}")
    print(f"{'=' * 100}")
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")

    # Create fold-specific directory
    fold_dir = experiment_dir / f"fold_{fold_num + 1}"
    fold_dir.mkdir(exist_ok=True)

    model_output_dir = fold_dir / "model"
    model_output_dir.mkdir(exist_ok=True)

    tb_log_dir = fold_dir / "tensorboard_logs"
    tb_log_dir.mkdir(exist_ok=True)

    # Prepare dataset for this fold
    dataset = prepare_dataset(examples, tokenizer, fold_indices=(train_indices, test_indices))

    print(f"\nFold {fold_num + 1} dataset:")
    print(f"  Training samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['test'])}")

    if TRAINING_CONFIG.get("use_token_class_weights", False) and TRAINING_CONFIG.get("auto_token_class_weights", False):
        auto_weights = infer_token_class_weights_from_train_split(dataset["train"])
        config.token_class_weights = auto_weights
        print(f"  Auto token class weights (train split): {auto_weights}")

    # Initialize model
    model = BertCRFForTokenClassification(config)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Calculate effective batch size
    effective_batch_size = TRAINING_CONFIG["batch_size"] * TRAINING_CONFIG["gradient_accumulation_steps"]
    total_steps = (len(dataset["train"]) // effective_batch_size) * TRAINING_CONFIG["num_train_epochs"]

    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        eval_strategy=TRAINING_CONFIG["eval_strategy"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        learning_rate=TRAINING_CONFIG["bert_base_learning_rate"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        logging_dir=str(fold_dir / "logs"),
        logging_steps=TRAINING_CONFIG["logging_steps"],
        report_to=[],
        run_name=f"fold_{fold_num + 1}",
        logging_first_step=True,
        logging_strategy="steps",
        seed=TRAINING_CONFIG["seed"],
        overwrite_output_dir=True,
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        warmup_steps=TRAINING_CONFIG.get("warmup_steps"),
        load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAINING_CONFIG["metric_for_best_model"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"], save_only_model=True,
        fp16=TRAINING_CONFIG["fp16"],
        adam_beta1=TRAINING_CONFIG["adam_beta1"],
        adam_beta2=TRAINING_CONFIG["adam_beta2"],
        adam_epsilon=TRAINING_CONFIG["adam_epsilon"],
    )

    # Add custom attributes for LLRD
    training_args.use_llrd = TRAINING_CONFIG["use_llrd"]
    training_args.llrd_decay_rate = TRAINING_CONFIG["llrd_decay_rate"]
    training_args.bert_base_learning_rate = TRAINING_CONFIG["bert_base_learning_rate"]
    training_args.embedding_learning_rate = TRAINING_CONFIG.get("embedding_learning_rate", 5e-4)
    training_args.classifier_learning_rate = TRAINING_CONFIG["classifier_learning_rate"]
    training_args.crf_learning_rate = TRAINING_CONFIG["crf_learning_rate"]
    training_args.bilstm_learning_rate = TRAINING_CONFIG["bilstm_learning_rate"]
    training_args.num_cycles = TRAINING_CONFIG["num_cycles"]
    training_args.use_weighted_sampler = TRAINING_CONFIG.get("use_weighted_sampler", False)
    training_args.weighted_sampler_mode = TRAINING_CONFIG.get("weighted_sampler_mode", "corpus_binary")
    training_args.weighted_sampler_max_weight = TRAINING_CONFIG.get("weighted_sampler_max_weight", 5.0)
    training_args.weighted_sampler_seed = TRAINING_CONFIG["seed"] + int(fold_num)

    # Create callbacks
    tb_callback = TensorBoardCallback(tb_log_dir)
    validation_callback = ValidationTextsCallback(
        tokenizer=tokenizer,
        val_dataset=dataset["test"],
        num_samples=TRAINING_CONFIG["show_validation_samples"],
    )
    early_stopping_callback = EarlyStoppingCallback(
        patience=TRAINING_CONFIG["early_stopping_patience"],
        threshold=TRAINING_CONFIG["early_stopping_threshold"]
    )

    # Add error collection callback if enabled
    callbacks: List[TrainerCallback] = [tb_callback, validation_callback, early_stopping_callback]
    error_collection_callback = None
    if TRAINING_CONFIG.get("collect_errors_per_epoch", True):
        error_collection_callback = ErrorCollectionCallback(
            tokenizer=tokenizer,
            val_dataset=dataset["test"],
            output_dir=model_output_dir,
        )
        callbacks.append(error_collection_callback)
        print(f"\nError collection per epoch: ENABLED")
    else:
        print(f"\nError collection per epoch: DISABLED")

    # Create trainer
    trainer = CRFTrainerWithLLRD(
            word_start_ids=build_word_start_ids(tokenizer),
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train
    print(f"\nStarting training for fold {fold_num + 1}...")
    train_result = trainer.train()

    # Save model
    trainer.save_model(str(model_output_dir))

    # Evaluate
    print(f"\nEvaluating fold {fold_num + 1}...")
    metrics = trainer.evaluate()

    print(f"\nFold {fold_num + 1} Results:")
    print(f"  Precision:            {metrics.get('eval_precision', 0):.4f}")
    print(f"  Recall:               {metrics.get('eval_recall', 0):.4f}")
    print(f"  F1:                   {metrics.get('eval_f1', 0):.4f}")
    print(f"  Macro F1 by corpus:   {metrics.get('eval_macro_f1_by_corpus', 0):.4f}")
    print(f"  Entity F1 (no O):     {metrics.get('eval_entity_token_f1_no_o', 0):.4f}")
    print(f"  Definition recall:    {metrics.get('eval_definition_recall', 0):.4f}")

    # Save fold metrics
    with open(fold_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if TRAINING_CONFIG.get("save_fold_predictions", False):
        save_fold_predictions(trainer, dataset["test"], examples, test_indices, fold_dir / "oof_predictions.jsonl")
    if TRAINING_CONFIG.get("remove_fold_models", False):
        import shutil

        shutil.rmtree(model_output_dir, ignore_errors=True)

    return metrics


def train_with_cross_validation(
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        experiment_dir: Path,
        config: PretrainedConfig,
):
    """Train with k-fold cross-validation."""

    n_folds = TRAINING_CONFIG["n_folds"]
    cv_seed = TRAINING_CONFIG.get("cv_seed", 42)

    print(f"\n{'=' * 100}")
    print(f"CROSS-VALIDATION: {n_folds}-FOLD")
    print(f"{'=' * 100}")
    print(f"Total examples: {len(examples)}")
    print(f"CV seed: {cv_seed}")

    # Create fold indices
    from sklearn.model_selection import GroupKFold, KFold

    indices = list(range(len(examples)))
    groups = [ex.get("group", "") for ex in examples]
    if all(groups):
        # Translation pairs / duplicates never straddle train and validation folds.
        splits = list(GroupKFold(n_splits=n_folds).split(indices, groups=groups))
    else:
        print("[WARN] No 'group' field in the data: CV folds may leak translations.")
        splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=cv_seed).split(indices))

    all_metrics = []

    for fold_num, (train_idx, test_idx) in enumerate(splits):
        fold_metrics = train_single_fold(
            fold_num=fold_num,
            train_indices=train_idx.tolist(),
            test_indices=test_idx.tolist(),
            examples=examples,
            tokenizer=tokenizer,
            experiment_dir=experiment_dir,
            config=config
        )
        all_metrics.append(fold_metrics)

    # Calculate average metrics across folds
    print(f"\n{'=' * 100}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'=' * 100}")

    metrics_to_average = [
        'eval_precision',
        'eval_recall',
        'eval_f1',
        'eval_macro_f1_by_corpus',
        'eval_entity_token_f1_no_o',
        'eval_definition_recall',
        'eval_def_f1',
        'eval_token_accuracy',
        'eval_def_precision',
        'eval_def_recall',
    ]

    avg_metrics = {}
    std_metrics = {}

    for metric_name in metrics_to_average:
        values = [m.get(metric_name, 0) for m in all_metrics]
        avg_metrics[metric_name] = np.mean(values)
        std_metrics[metric_name] = np.std(values)

    print(f"\nAverage Metrics across {n_folds} folds:")
    print(f"  Precision:            {avg_metrics['eval_precision']:.4f} ± {std_metrics['eval_precision']:.4f}")
    print(f"  Recall:               {avg_metrics['eval_recall']:.4f} ± {std_metrics['eval_recall']:.4f}")
    print(f"  F1:                   {avg_metrics['eval_f1']:.4f} ± {std_metrics['eval_f1']:.4f}")
    print(f"  Macro F1 by corpus:   {avg_metrics['eval_macro_f1_by_corpus']:.4f} ± {std_metrics['eval_macro_f1_by_corpus']:.4f}")
    print(f"  Entity F1 (no O):     {avg_metrics['eval_entity_token_f1_no_o']:.4f} ± {std_metrics['eval_entity_token_f1_no_o']:.4f}")
    print(f"  Definition recall:    {avg_metrics['eval_definition_recall']:.4f} ± {std_metrics['eval_definition_recall']:.4f}")

    print(f"\nPer-fold results:")
    for i, metrics in enumerate(all_metrics, 1):
        print(
            f"  Fold {i}: F1={metrics.get('eval_f1', 0):.4f}, "
            f"MacroF1(corpus)={metrics.get('eval_macro_f1_by_corpus', 0):.4f}, "
            f"DefRecall={metrics.get('eval_definition_recall', 0):.4f}"
        )

    # Save cross-validation summary
    cv_summary = {
        "n_folds": n_folds,
        "cv_seed": cv_seed,
        "average_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "per_fold_metrics": all_metrics
    }

    with open(experiment_dir / "cross_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2, ensure_ascii=False)

    print(f"\nCross-validation summary saved to: {experiment_dir / 'cross_validation_summary.json'}")
    print(f"{'=' * 100}\n")

    return cv_summary


def main():
    try:
        print("\n" + "=" * 100)
        print("STARTING TRAINING WITH LAYER-WISE LEARNING RATE DECAY (LLRD)")
        print("=" * 100 + "\n")

        # Create experiment directory
        experiment_dir = create_experiment_dir(BASE_OUTPUT_DIR, TRAINING_CONFIG["experiment_name"])
        print(f"Experiment directory: {experiment_dir}")
        if TRAINING_CONFIG.get("preset_name"):
            print(f"Active preset: {TRAINING_CONFIG['preset_name']}")

        model_output_dir = experiment_dir / "model"
        model_output_dir.mkdir(exist_ok=True)

        tb_log_dir = experiment_dir / "tensorboard_logs"
        tb_log_dir.mkdir(exist_ok=True)

        experiment_config = {
            "training_config": TRAINING_CONFIG,
            "start_time": datetime.now().isoformat(),
            "experiment_dir": str(experiment_dir),
            "tensorboard_log_dir": str(tb_log_dir),
        }
        save_experiment_config(experiment_dir, experiment_config)

        # Load and process data
        print(f"\nLoading data from: {DATA_PATH}")
        raw = read_json(DATA_PATH)
        if TRAINING_CONFIG.get("ocr_noise_fraction", 0.0) > 0:
            from ocr_noise import augment_records

            n_before = len(raw)
            raw = augment_records(raw, TRAINING_CONFIG["ocr_noise_fraction"],
                                  TRAINING_CONFIG.get("ocr_noise_max_level", 0.12), TRAINING_CONFIG["seed"])
            print(f"[OCR noise] added {len(raw) - n_before} noisy copies to {n_before} training records")
        examples = build_examples(raw)

        # Load an explicit validation set if one is configured
        val_path_cfg = TRAINING_CONFIG.get("validation_data_path")
        val_examples_built = None
        if val_path_cfg:
            val_path = Path(val_path_cfg)
            if val_path.exists():
                print(f"\nLoading explicit validation data from: {val_path}")
                raw_val = read_json_any(val_path)
                val_examples_built = build_examples(raw_val)
            else:
                print(f"[WARN] validation_data_path is set but the file was not found: {val_path}")

        data_info = {
            "total_examples": len(examples),
            "with_definitions": sum(1 for ex in examples if ex["def_span"] is not None),
            "without_definitions": sum(1 for ex in examples if ex["def_span"] is None),
            "dataset_created": datetime.now().isoformat(),
        }
        with open(experiment_dir / "data_info.json", "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=2, ensure_ascii=False)

        print(f"\nDataset statistics:")
        print(f"  Total examples: {data_info['total_examples']}")
        print(f"  With definitions: {data_info['with_definitions']}")
        print(f"  Without definitions: {data_info['without_definitions']}")

        # Initialize tokenizer
        print(f"\nLoading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Create model configuration
        print(f"\nInitializing model configuration: {MODEL_NAME}")
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.num_labels = len(LABELS)
        config.id2label = ID2LABEL
        config.label2id = LABEL2ID
        config._name_or_path = MODEL_NAME

        # Add encoder configuration
        config.use_bert_encoder = TRAINING_CONFIG.get("use_bert_encoder", True)
        config.embedding_dim = TRAINING_CONFIG.get("embedding_dim", 256)
        config.embedding_dropout = TRAINING_CONFIG.get("embedding_dropout", 0.1)

        # Add BiLSTM configuration
        config.use_bilstm = TRAINING_CONFIG["use_bilstm"]
        config.use_sentence_head = TRAINING_CONFIG.get("use_sentence_head", False)
        config.sentence_head_weight = TRAINING_CONFIG.get("sentence_head_weight", 0.3)
        config.bilstm_hidden_size = TRAINING_CONFIG["bilstm_hidden_size"]
        config.bilstm_num_layers = TRAINING_CONFIG["bilstm_num_layers"]
        config.bilstm_dropout = TRAINING_CONFIG["bilstm_dropout"]

        # Add CRF configuration
        config.use_crf = TRAINING_CONFIG["use_crf"]

        # Add token class-weight configuration
        config.use_token_class_weights = TRAINING_CONFIG.get("use_token_class_weights", False)
        config.token_class_weights = TRAINING_CONFIG.get("token_class_weights", {"O": 1.0, "ENTITY": 1.0})
        config.crf_emission_loss_weight = TRAINING_CONFIG.get("crf_emission_loss_weight", 0.0)

        print(f"\nModel configuration:")
        # Build architecture string dynamically
        arch_parts = []
        if config.use_bert_encoder:
            arch_parts.append("BERT")
        else:
            arch_parts.append("TokenEmbedding")

        if TRAINING_CONFIG["use_bilstm"]:
            arch_parts.append("BiLSTM")

        if TRAINING_CONFIG["use_crf"]:
            arch_parts.append("CRF")
        else:
            arch_parts.append("Softmax")

        print(f"  Architecture: {' + '.join(arch_parts)}")

        if not config.use_bert_encoder and not TRAINING_CONFIG["use_bilstm"] and not TRAINING_CONFIG["use_crf"]:
            raise ValueError("Without an encoder, enable use_bilstm and/or use_crf (embeddings + linear layer alone is not a model).")

        if not config.use_bert_encoder:
            print(f"  Embedding dim: {TRAINING_CONFIG.get('embedding_dim', 256)}")
            print(f"  Embedding dropout: {TRAINING_CONFIG.get('embedding_dropout', 0.1)}")

        if TRAINING_CONFIG["use_bilstm"]:
            print(f"  BiLSTM hidden size: {TRAINING_CONFIG['bilstm_hidden_size']}")
            print(f"  BiLSTM layers: {TRAINING_CONFIG['bilstm_num_layers']}")
            print(f"  BiLSTM dropout: {TRAINING_CONFIG['bilstm_dropout']}")

        if TRAINING_CONFIG["use_crf"]:
            print(f"  CRF: ENABLED")
        else:
            print(f"  CRF: DISABLED (using standard softmax)")

        print(f"  Token class weights: {TRAINING_CONFIG.get('token_class_weights')}")
        print(f"  Auto token class weights: {TRAINING_CONFIG.get('auto_token_class_weights', False)}")
        print(f"  Weighted sampler: {TRAINING_CONFIG.get('use_weighted_sampler', False)} ({TRAINING_CONFIG.get('weighted_sampler_mode', 'corpus_binary')})")
        print(f"  CRF emission loss weight: {TRAINING_CONFIG.get('crf_emission_loss_weight', 0.0)}")
        print(f"  Early stopping metric: {TRAINING_CONFIG.get('metric_for_best_model')}")

        # Check if cross-validation is enabled
        use_cross_validation = TRAINING_CONFIG.get("use_cross_validation", False)

        if use_cross_validation:
            # Use cross-validation
            print(f"\nCross-validation ENABLED ({TRAINING_CONFIG['n_folds']} folds)")
            cv_summary = train_with_cross_validation(examples, tokenizer, experiment_dir, config)

            # Update experiment config with CV results
            experiment_config.update({
                "end_time": datetime.now().isoformat(),
                "training_mode": "cross_validation",
                "n_folds": TRAINING_CONFIG["n_folds"],
                "cv_results": cv_summary
            })
            save_experiment_config(experiment_dir, experiment_config)

            print(f"\n{'=' * 100}")
            print("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
            print(f"{'=' * 100}")
            print(f"\nResults saved to: {experiment_dir}")
            print(f"Average F1 Score: {cv_summary['average_metrics']['eval_f1']:.4f} ± {cv_summary['std_metrics']['eval_f1']:.4f}")
            print(f"{'=' * 100}\n")

            return

        # Standard training (no cross-validation)
        print(f"\nStandard training (no cross-validation)")
        if val_examples_built is not None and len(val_examples_built) > 0:
            dataset = prepare_dataset(examples, tokenizer, TRAINING_CONFIG["split_ratio"], val_examples=val_examples_built)
        else:
            dataset = prepare_dataset(examples, tokenizer, TRAINING_CONFIG["split_ratio"])

        print(f"\nDataset splits:")
        print(f"  Training samples: {len(dataset['train'])}")
        print(f"  Validation samples: {len(dataset['test'])}")

        if TRAINING_CONFIG.get("use_token_class_weights", False) and TRAINING_CONFIG.get("auto_token_class_weights", False):
            auto_weights = infer_token_class_weights_from_train_split(dataset["train"])
            config.token_class_weights = auto_weights
            print(f"  Auto token class weights (train split): {auto_weights}")

        # Initialize CRF model
        model = BertCRFForTokenClassification(config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        data_collator = DataCollatorForTokenClassification(tokenizer)

        # Calculate effective batch size
        effective_batch_size = TRAINING_CONFIG["batch_size"] * TRAINING_CONFIG["gradient_accumulation_steps"]
        total_steps = (len(dataset["train"]) // effective_batch_size) * TRAINING_CONFIG["num_train_epochs"]

        print(f"\nTraining configuration:")
        print(f"  Batch size per device: {TRAINING_CONFIG['batch_size']}")
        print(f"  Gradient accumulation steps: {TRAINING_CONFIG['gradient_accumulation_steps']}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Number of epochs: {TRAINING_CONFIG['num_train_epochs']}")
        print(f"  Total training steps: {total_steps}")
        print(f"  Mixed precision (FP16): {TRAINING_CONFIG['fp16']}")

        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            eval_strategy=TRAINING_CONFIG["eval_strategy"],
            save_strategy=TRAINING_CONFIG["save_strategy"],
            learning_rate=TRAINING_CONFIG["bert_base_learning_rate"],  # Base LR (will be overridden by LLRD)
            per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
            per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
            num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            logging_dir=str(experiment_dir / "logs"),
            logging_steps=TRAINING_CONFIG["logging_steps"],
            report_to=[],
            run_name=f"def_extract_llrd_{experiment_dir.name}",
            logging_first_step=True,
            logging_strategy="steps",
            seed=TRAINING_CONFIG["seed"],
            overwrite_output_dir=True,
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
            warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
            warmup_steps=TRAINING_CONFIG.get("warmup_steps"),
            load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
            metric_for_best_model=TRAINING_CONFIG["metric_for_best_model"],
            lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
            save_total_limit=TRAINING_CONFIG["save_total_limit"], save_only_model=True,
            fp16=TRAINING_CONFIG["fp16"],
            adam_beta1=TRAINING_CONFIG["adam_beta1"],
            adam_beta2=TRAINING_CONFIG["adam_beta2"],
            adam_epsilon=TRAINING_CONFIG["adam_epsilon"],
        )

        # Add custom attributes for LLRD
        training_args.use_llrd = TRAINING_CONFIG["use_llrd"]
        training_args.llrd_decay_rate = TRAINING_CONFIG["llrd_decay_rate"]
        training_args.bert_base_learning_rate = TRAINING_CONFIG["bert_base_learning_rate"]
        training_args.embedding_learning_rate = TRAINING_CONFIG.get("embedding_learning_rate", 5e-4)
        training_args.classifier_learning_rate = TRAINING_CONFIG["classifier_learning_rate"]
        training_args.crf_learning_rate = TRAINING_CONFIG["crf_learning_rate"]
        training_args.bilstm_learning_rate = TRAINING_CONFIG["bilstm_learning_rate"]
        training_args.num_cycles = TRAINING_CONFIG["num_cycles"]
        training_args.use_weighted_sampler = TRAINING_CONFIG.get("use_weighted_sampler", False)
        training_args.weighted_sampler_mode = TRAINING_CONFIG.get("weighted_sampler_mode", "corpus_binary")
        training_args.weighted_sampler_max_weight = TRAINING_CONFIG.get("weighted_sampler_max_weight", 5.0)
        training_args.weighted_sampler_seed = TRAINING_CONFIG["seed"]

        # Create callbacks
        tb_callback = TensorBoardCallback(tb_log_dir)
        validation_callback = ValidationTextsCallback(
            tokenizer=tokenizer,
            val_dataset=dataset["test"],
            num_samples=TRAINING_CONFIG["show_validation_samples"],
        )
        early_stopping_callback = EarlyStoppingCallback(
            patience=TRAINING_CONFIG["early_stopping_patience"],
            threshold=TRAINING_CONFIG["early_stopping_threshold"]
        )

        # Add error collection callback if enabled
        callbacks: List[TrainerCallback] = [tb_callback, validation_callback, early_stopping_callback]
        error_collection_callback = None
        if TRAINING_CONFIG.get("collect_errors_per_epoch", True):
            error_collection_callback = ErrorCollectionCallback(
                tokenizer=tokenizer,
                val_dataset=dataset["test"],
                output_dir=model_output_dir,
            )
            callbacks.append(error_collection_callback)
            print(f"\nError collection per epoch: ENABLED")
        else:
            print(f"\nError collection per epoch: DISABLED")

        # Create trainer
        trainer = CRFTrainerWithLLRD(
            word_start_ids=build_word_start_ids(tokenizer),
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        # Link trainer to error collection callback
        if error_collection_callback is not None:
            error_collection_callback.trainer = trainer

        # Start training
        print(f"\n{'=' * 100}")
        print("STARTING TRAINING")
        print(f"{'=' * 100}")
        print(f"\nExperiment: {experiment_dir.name}")
        print(f"TensorBoard logs: {tb_log_dir}")
        print(f"\nTo view logs run:")
        print(f"  tensorboard --logdir={tb_log_dir}")
        print(f"\nShowing {TRAINING_CONFIG['show_validation_samples']} validation samples per epoch\n")

        train_result = trainer.train()

        # Save model and tokenizer
        print("\nSaving model and tokenizer...")
        trainer.save_model(str(model_output_dir))
        tokenizer.save_pretrained(str(model_output_dir))
        config.save_pretrained(str(model_output_dir))

        # Save final metrics
        print("\nEvaluating final model...")
        final_metrics = trainer.evaluate()

        print(f"\n{'=' * 100}")
        print("FINAL EVALUATION METRICS")
        print(f"{'=' * 100}")
        for key, value in sorted(final_metrics.items()):
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.4f}")
            else:
                print(f"  {key:30s}: {value}")
        print(f"{'=' * 100}\n")

        with open(experiment_dir / "final_metrics.json", "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False )

        test_paths = [p for p in TRAINING_CONFIG.get("test_data_paths") or [] if Path(p).exists()]
        if test_paths:
            print("\nEvaluating on held-out test files (word-level spans)...")
            test_results = evaluate_test_files(trainer, tokenizer, test_paths)
            with open(experiment_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
        else:
            print("\n[WARN] No test files found; skipping held-out evaluation.")

        # Collect and save error examples
        if TRAINING_CONFIG.get("collect_errors", True):
            collect_error_examples(trainer, dataset["test"], tokenizer, model_output_dir)

        # Update experiment configuration with results
        experiment_config.update(
            {
                "end_time": datetime.now().isoformat(),
                "training_metrics": final_metrics,
                "training_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "num_train_samples": len(dataset["train"]),
                "num_eval_samples": len(dataset["test"]),
                "num_epochs_completed": trainer.state.epoch,
                "model_type": "BertCRFForTokenClassification",
                "crf_enabled": bool(TRAINING_CONFIG["use_crf"]),
                "bilstm_enabled": bool(TRAINING_CONFIG["use_bilstm"]),
                "llrd_enabled": TRAINING_CONFIG["use_llrd"],
                "llrd_decay_rate": TRAINING_CONFIG["llrd_decay_rate"],
            }
        )
        save_experiment_config(experiment_dir, experiment_config)

        print(f"\n{'=' * 100}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'=' * 100}")
        print(f"\nResults saved to: {experiment_dir}")
        print(f"Model saved to: {model_output_dir}")
        print(f"TensorBoard logs: {tb_log_dir}")
        print(f"\nBest F1 Score: {final_metrics.get('eval_f1', 0):.4f}")
        print(f"{'=' * 100}\n")

    except Exception as e:
        print(f"\n{'=' * 100}")
        print("ERROR OCCURRED DURING TRAINING")
        print(f"{'=' * 100}")
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
        print(f"\n{'=' * 100}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a term/definition sequence labeler.")
    parser.add_argument("--preset", choices=sorted(TRAINING_PRESETS), default=None,
                        help="Architecture / experiment preset (see TRAINING_PRESETS).")
    parser.add_argument("--data-path", default=None, help="Training data (JSONL), default: output/train.json")
    parser.add_argument("--validation-data-path", default=None, help="Optional explicit validation file.")
    parser.add_argument("--base-output-dir", default=None, help="Where experiments are written.")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--model-name", default=None, help="HF encoder name, default: FacebookAI/xlm-roberta-large")
    parser.add_argument("--test-files", nargs="*", default=None,
                        help="Held-out files evaluated after training (default: output/test_*.json).")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Maximum number of epochs.")
    parser.add_argument("--sampler-mode", choices=["none", "binary", "corpus", "corpus_binary"], default=None,
                        help="Weighted sampler grouping (default: corpus_binary; 'none' disables it).")
    parser.add_argument("--entity-weight", type=float, default=None,
                        help="Fixed CE weight of entity tokens vs O (disables the automatic weight; 1.0 = unweighted).")
    parser.add_argument("--corpus-tag", action="store_true", help="Prefix sentences with a source-corpus tag.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (data split, sampler, init).")
    parser.add_argument("--ocr-noise", type=float, default=None,
                        help="Share of training records that get an additional OCR-noised copy (e.g. 0.4).")
    parser.add_argument("--ocr-noise-max-level", type=float, default=None)
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--sentence-head", action="store_true",
                        help="Add the auxiliary sentence-level 'has definition' head.")
    parser.add_argument("--no-cross-validation", action="store_true", help="Single train/validation split instead of k-fold.")
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> None:
    global DATA_PATH, BASE_OUTPUT_DIR, MODEL_NAME
    if args.preset:
        TRAINING_CONFIG["preset_name"] = args.preset
        apply_training_preset(TRAINING_CONFIG, TRAINING_PRESETS)
    if args.data_path:
        TRAINING_CONFIG["data_path"] = args.data_path
    if args.validation_data_path:
        TRAINING_CONFIG["validation_data_path"] = args.validation_data_path
    if args.base_output_dir:
        TRAINING_CONFIG["base_output_dir"] = args.base_output_dir
    if args.experiment_name:
        TRAINING_CONFIG["experiment_name"] = args.experiment_name
    if args.model_name:
        TRAINING_CONFIG["model_name"] = args.model_name
    if args.test_files is not None:
        TRAINING_CONFIG["test_data_paths"] = args.test_files
    if getattr(args, "batch_size", None) is not None:
        TRAINING_CONFIG["batch_size"] = args.batch_size
    if getattr(args, "gradient_accumulation_steps", None) is not None:
        TRAINING_CONFIG["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if getattr(args, "epochs", None) is not None:
        TRAINING_CONFIG["num_train_epochs"] = args.epochs
    if getattr(args, "sampler_mode", None):
        if args.sampler_mode == "none":
            TRAINING_CONFIG["use_weighted_sampler"] = False
        else:
            TRAINING_CONFIG["use_weighted_sampler"] = True
            TRAINING_CONFIG["weighted_sampler_mode"] = args.sampler_mode
    if getattr(args, "entity_weight", None) is not None:
        TRAINING_CONFIG["auto_token_class_weights"] = False
        TRAINING_CONFIG["token_class_weights"] = {"O": 1.0, "ENTITY": float(args.entity_weight)}
    if getattr(args, "corpus_tag", False):
        TRAINING_CONFIG["corpus_tag"] = True
    if getattr(args, "ocr_noise", None) is not None:
        TRAINING_CONFIG["ocr_noise_fraction"] = args.ocr_noise
    if getattr(args, "ocr_noise_max_level", None) is not None:
        TRAINING_CONFIG["ocr_noise_max_level"] = args.ocr_noise_max_level
    if getattr(args, "seed", None) is not None:
        TRAINING_CONFIG["seed"] = args.seed
    if getattr(args, "sentence_head", False):
        TRAINING_CONFIG["use_sentence_head"] = True
    if getattr(args, "no_torch_compile", False):
        TRAINING_CONFIG["torch_compile"] = False
    if args.no_cross_validation:
        TRAINING_CONFIG["use_cross_validation"] = False
    DATA_PATH = Path(TRAINING_CONFIG["data_path"])
    BASE_OUTPUT_DIR = Path(TRAINING_CONFIG["base_output_dir"])
    MODEL_NAME = TRAINING_CONFIG["model_name"]


if __name__ == "__main__":
    apply_cli_overrides(parse_args())
    main()
