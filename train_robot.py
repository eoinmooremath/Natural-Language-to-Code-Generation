"""
Single-GPU GH200 (H100 96GB) LoRA SFT.

Supports either:
  A) --data-json (one file; split train/val in-memory), or
  B) --train-json + --val-json (two files).

Key features:
- Prefers `output_list` (one action per line). Falls back to normalized string.
- Response-only loss (prompt masked with -100; pads also -100).
- BF16 + SDPA on H100 (no bitsandbytes).
- LoRA r=32, attention + MLP target modules.
- Dynamic padding + group_by_length for throughput.
- Paired logging: every 100 steps we record BOTH train loss and eval loss in the same row.
"""

import os
import json
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

# ----------------- Config -----------------

@dataclass
class ModelCfg:
    model_name: str = "codellama/CodeLlama-7b-Python-hf"
    max_length: int = 256                      # larger cap to avoid response truncation
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.15
    target_modules: List[str] = tuple("q_proj k_proj v_proj o_proj gate_proj up_proj down_proj".split())

@dataclass
class TrainCfg:
    output_dir: str = "./outputs/robot-lora"
    seed: int = 42

    # Batch / accumulation (tuned for GH200; override with --batch-size if needed)
    per_device_train_batch_size: int = 24
    per_device_eval_batch_size: int = 24
    gradient_accumulation_steps: int = 1

    # Optim/schedule/regularization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    optim: str = "adamw_torch_fused"
    label_smoothing_factor: float = 0.05
    max_grad_norm: float = 1.0

    # Runtime
    num_train_epochs: int = 3
    max_steps: int = -1                  # -1 => no cap (we pass None to HF)
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False

    # Eval/save/log cadence — PAIRED every 100 steps
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 100
    report_to: str = "tensorboard"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 2
    early_stopping_threshold: Optional[float] = 1e-3

    # DataLoader
    dataloader_num_workers: int = 12
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    dataloader_persistent_workers: bool = True

    # Length-aware batching
    group_by_length: bool = True
    length_column_name: str = "length"

# --------- Category weighting (optional) ---------

HARD_TARGET_SHARE = {
    # emphasize chained/complex & parametric reasoning
    "complex": 0.15,
    "chained_complex": 0.15,
    "parametric": 0.05,
    "contextual": 0.1,
    "error_handling": 0.02,
    "movement": 0.1, 
    "manipulation": 0.1, 
    "navigation": 0.1, 
    "sensors": 0.1,
    "conversational": 0.1, 
    "safety": 0.03, 
    "efficiency": 0.0,
}
def _norm(d):
    s = sum(d.values())
    return {k: v/s for k, v in d.items()} if abs(s-1.0) > 1e-9 else d
HARD_TARGET_SHARE = _norm(HARD_TARGET_SHARE)

def compute_category_weights(items: List[Dict], key="category", targets: Dict[str, float]=HARD_TARGET_SHARE) -> List[float]:
    from collections import Counter
    counts = Counter([d.get(key, "unknown") for d in items])
    total = max(1, sum(counts.values()))
    cur_share = {k: c / total for k, c in counts.items()}
    ws = []
    for d in items:
        cat = d.get(key, "unknown")
        tgt = targets.get(cat, 0.01)
        cur = cur_share.get(cat, 1e-12)
        w = max(tgt / cur, 1e-6)
        ws.append(w)
    m = sum(ws) / len(ws)
    return [w / m for w in ws]  # mean ~ 1

class WeightedTrainer(Trainer):
    """Single-GPU WeightedRandomSampler; falls back to default if DDP > 1."""
    def __init__(self, *args, train_weights=None, persistent_workers=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_weights = train_weights
        self._persistent_workers = persistent_workers

    def get_train_dataloader(self):
        if self.train_dataset is None or self.args.world_size > 1 or self._train_weights is None:
            return super().get_train_dataloader()
        sampler = WeightedRandomSampler(
            weights=self._train_weights,
            num_samples=len(self._train_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self._persistent_workers,
        )

# --------------- QoL callbacks ----------------

class ThroughputCallback(TrainerCallback):
    def __init__(self, seq_len: int, per_device_bs: int, every_n: int = 20):
        self.seq_len = seq_len
        self.bs = per_device_bs
        self.every = every_n
        self.last = None
    def on_train_begin(self, args, state, control, **kwargs):
        self.last = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        now = time.time(); dt = now - self.last; self.last = now
        if state.global_step and state.global_step % self.every == 0:
            toks = self.seq_len * self.bs * (args.gradient_accumulation_steps or 1)
            print(f"[step {state.global_step}] {dt:.3f}s/it  ~  {toks/dt:.0f} tok/s")

class PairedLossLogger(TrainerCallback):
    """
    Writes a single CSV row with both train loss and eval loss at the SAME step.
    Requires eval_steps == logging_steps.
    """
    def __init__(self, out_csv: Path):
        self.out_csv = out_csv
        self.header_written = False
        self._last_train_step = None
        self._last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        if "loss" in logs:
            self._last_train_loss = float(logs["loss"])
            self._last_train_step = state.global_step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None: return
        step = state.global_step
        eval_loss = float(metrics.get("eval_loss", "nan"))
        loss = self._last_train_loss if self._last_train_step == step else float("nan")

        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        write_header = (not self.out_csv.exists()) and (not self.header_written)
        with open(self.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step","loss","eval_loss"])
            if write_header:
                w.writeheader(); self.header_written = True
            w.writerow({"step": step, "loss": loss, "eval_loss": eval_loss})

# -------------- Collator (dynamic padding + label padding) --------------

class SimpleSupervisedCollator:
    """
    Pads input_ids/attention_mask with tokenizer.pad and pads labels to -100 up to the same max length.
    """
    def __init__(self, tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.mult = pad_to_multiple_of
        self.label_pad = label_pad_token_id

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            pad_to_multiple_of=self.mult,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [self.label_pad] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(torch.tensor(lab, dtype=torch.long))

        batch["labels"] = torch.stack(padded_labels, dim=0)
        return batch

# -------------- Data processing ----------------

class Processor:
    def __init__(self, tok: AutoTokenizer, max_length: int = 256):
        self.tok = tok
        self.L = max_length
        self.marker_text = "### Response:"
        self.marker_ids = self.tok(self.marker_text, add_special_tokens=False)["input_ids"]

    def tokenize(self, data: List[Dict]) -> Dataset:
        ds = Dataset.from_list(data)
        return ds.map(
            self._encode_batch,
            batched=True,
            remove_columns=ds.column_names,
            desc="Tokenizing (response-only; dynamic pad)",
        )

    def _encode_batch(self, examples: Dict[str, List]) -> Dict[str, List[List[int]]]:
        L = self.L
        out_input_ids, out_labels, out_len = [], [], []

        for i in range(len(examples["input"])):
            inst = f"### Instruction: {examples['input'][i]}\n{self.marker_text}"

            # Prefer output_list if present; fallback to normalized string
            if "output_list" in examples and examples["output_list"][i]:
                if isinstance(examples["output_list"][i], list):
                    resp_txt = "\n".join(
                        s.strip().rstrip(".")
                        for s in examples["output_list"][i]
                        if isinstance(s, str) and s.strip()
                    )
                else:
                    resp_txt = str(examples["output_list"][i]).strip()
            else:
                raw = examples["output"][i]
                resp_txt = raw.replace("\n;", "\n").replace(";\n", "\n").replace(";", "\n").strip()

            inst_ids = self.tok(inst, add_special_tokens=False)["input_ids"]
            resp_ids = self.tok(resp_txt, add_special_tokens=False)["input_ids"]

            # keep the full response if possible; trim prompt first
            if len(resp_ids) >= L:
                resp_ids = resp_ids[: L - 1]        # keep >=1 slot for the marker
                inst_ids = self.marker_ids[:]
            else:
                max_inst = L - len(resp_ids)
                if len(inst_ids) > max_inst:
                    inst_ids = inst_ids[-max_inst:]
                    if len(inst_ids) < len(self.marker_ids):
                        inst_ids = self.marker_ids[:]

            input_ids = inst_ids + resp_ids
            labels = [-100] * len(inst_ids) + resp_ids

            if len(input_ids) > L:
                overflow = len(input_ids) - L
                inst_ids = inst_ids[overflow:]
                input_ids = inst_ids + resp_ids
                labels = [-100] * len(inst_ids) + resp_ids

            out_input_ids.append(input_ids)
            out_labels.append(labels)
            out_len.append(len(input_ids))

        return {"input_ids": out_input_ids, "labels": out_labels, "length": out_len}

# ------------- Model / Tokenizer --------------

def setup_model_and_tokenizer(cfg: ModelCfg):
    print(f"🤖 Loading base: {cfg.model_name}")
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Hopper-friendly throughput
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # training

    # LoRA adapters
    print("⚡ Attaching LoRA adapters (r=32, attn+MLP)...")
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    return model, tok

# ----------------- Split + Preflight -----------------

def split_train_val(all_rows, val_split=0.08, seed=42):
    import random
    rng = random.Random(seed)
    rows = list(all_rows)
    rng.shuffle(rows)
    n = len(rows)
    n_val = max(1, int(n * val_split))
    val = rows[:n_val]
    train = rows[n_val:]
    return train, val

def preflight_checks(train_raw: List[Dict], tok, max_length: int):
    """Quick sanity: length percentiles + tiny forward pass."""
    print("🔎 Preflight: computing length percentiles...")
    import numpy as np
    lens = []
    for d in train_raw[:5000]:
        inst = f"### Instruction: {d['input']}\n### Response:"
        if "output_list" in d and d["output_list"]:
            resp = "\n".join(s.strip().rstrip(".") for s in d["output_list"] if isinstance(s, str) and s.strip())
        else:
            raw = d["output"]
            resp = raw.replace("\n;", "\n").replace(";\n", "\n").replace(";", "\n").strip()
        Li = len(tok(inst, add_special_tokens=False)["input_ids"])
        Lr = len(tok(resp, add_special_tokens=False)["input_ids"])
        lens.append(Li + Lr)
    pct = np.percentile(lens, [50, 75, 90, 95, 99])
    print(f"   sum(inst+resp) percentiles: 50%={pct[0]:.0f} 75%={pct[1]:.0f} 90%={pct[2]:.0f} 95%={pct[3]:.0f} 99%={pct[4]:.0f}")
    if pct[4] > max_length and max_length < 256:
        print(f"   ⚠️  P99 ({pct[4]:.0f}) exceeds max_length={max_length}. Consider 256.")

    # Tiny forward
    print("🔎 Preflight: tiny batch forward...")
    proc = Processor(tok, max_length)
    tiny_ds = Dataset.from_list(train_raw[:32]).map(proc._encode_batch, batched=True, remove_columns=Dataset.from_list(train_raw[:32]).column_names)
    collator = SimpleSupervisedCollator(tokenizer=tok, pad_to_multiple_of=8)
    batch = collator([tiny_ds[i] for i in range(min(8, len(tiny_ds)))])
    assert batch["input_ids"].shape[0] == min(8, len(tiny_ds))
    assert (batch["labels"][:,0] == -100).all(), "Labels should be -100 at start (prompt masked)."
    print("   ✅ Preflight OK.")

# ------------------ Main ---------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()

    # Either single file OR train/val files
    ap.add_argument("--data-json", default=None, help="One file with all examples; we'll split train/val in-memory.")
    ap.add_argument("--val-split", type=float, default=0.08, help="Only used with --data-json")

    ap.add_argument("--train-json", default=None, help="Train file (if not using --data-json)")
    ap.add_argument("--val-json", default=None, help="Val file   (if not using --data-json)")

    ap.add_argument("--output-dir", default="./outputs/robot-lora")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=None, help="Per-device train batch size override.")
    ap.add_argument("--oversample-hard", action="store_true", help="Enable category-weighted sampling (single GPU).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mcfg = ModelCfg(max_length=args.max_length)
    tcfg = TrainCfg(output_dir=args.output_dir, seed=args.seed)
    if args.batch_size:
        tcfg.per_device_train_batch_size = args.batch_size
        tcfg.per_device_eval_batch_size = max(16, args.batch_size)

    # Repro
    set_seed(tcfg.seed)

    # Load data (single file or two files)
    if args.data_json:
        all_rows = json.loads(Path(args.data_json).read_text())
        train_raw, val_raw = split_train_val(all_rows, args.val_split, args.seed)
        print(f"Loaded {len(all_rows)} rows from {args.data_json} -> train={len(train_raw)} val={len(val_raw)}")
    else:
        if not (args.train_json and args.val_json):
            raise SystemExit("Provide either --data-json OR both --train-json and --val-json.")
        train_raw = json.loads(Path(args.train_json).read_text())
        val_raw   = json.loads(Path(args.val_json).read_text())
        print(f"Loaded train={len(train_raw)} from {args.train_json}; val={len(val_raw)} from {args.val_json}")

    # Model/tokenizer
    model, tok = setup_model_and_tokenizer(mcfg)

    # Preflight
    preflight_checks(train_raw, tok, mcfg.max_length)

    # Tokenize
    print("🔄 Tokenizing datasets...")
    proc = Processor(tok, mcfg.max_length)
    train_ds = proc.tokenize(train_raw)
    val_ds   = proc.tokenize(val_raw)

    # Use a small eval subset during training for smooth curves; full eval at end
    eval_small = val_ds.select(range(min(3000, len(val_ds))))

    # Collator
    collator = SimpleSupervisedCollator(tokenizer=tok, pad_to_multiple_of=8)

    # Training args
    args_hf = TrainingArguments(
        output_dir=tcfg.output_dir,
        seed=tcfg.seed,
        num_train_epochs=tcfg.num_train_epochs,

        per_device_train_batch_size=tcfg.per_device_train_batch_size,
        per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,

        learning_rate=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        warmup_ratio=tcfg.warmup_ratio,
        lr_scheduler_type="cosine",
        optim=tcfg.optim,
        label_smoothing_factor=tcfg.label_smoothing_factor,
        max_grad_norm=tcfg.max_grad_norm,

        bf16=tcfg.bf16,
        fp16=tcfg.fp16,
        gradient_checkpointing=tcfg.gradient_checkpointing,

        # PAIRED cadence
        logging_strategy="steps",
        logging_steps=tcfg.logging_steps,
        eval_strategy=tcfg.eval_strategy,
        eval_steps=tcfg.eval_steps,
        save_strategy=tcfg.save_strategy,
        save_steps=tcfg.save_steps,
        save_total_limit=tcfg.save_total_limit,

        report_to=tcfg.report_to,
        logging_dir=str(Path(tcfg.output_dir) / "runs"),

        dataloader_num_workers=tcfg.dataloader_num_workers,
        dataloader_pin_memory=tcfg.dataloader_pin_memory,
        dataloader_drop_last=tcfg.dataloader_drop_last,
        dataloader_persistent_workers=tcfg.dataloader_persistent_workers,

        group_by_length=tcfg.group_by_length,
        length_column_name=tcfg.length_column_name,

        load_best_model_at_end=tcfg.load_best_model_at_end,
        metric_for_best_model=tcfg.metric_for_best_model,
        greater_is_better=tcfg.greater_is_better,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=tcfg.early_stopping_patience,
            early_stopping_threshold=tcfg.early_stopping_threshold,
        ),
        ThroughputCallback(seq_len=mcfg.max_length,
                           per_device_bs=tcfg.per_device_train_batch_size,
                           every_n=20),
        PairedLossLogger(out_csv=Path(tcfg.output_dir) / "paired_losses.csv"),
    ]

    # Optional category weighting
    if args.oversample_hard:
        train_weights = compute_category_weights(train_raw)
        print("🎯 Using category-weighted sampling for training (single GPU).")
        trainer_cls = WeightedTrainer
        trainer_kwargs = dict(train_weights=train_weights,
                              persistent_workers=tcfg.dataloader_persistent_workers)
    else:
        trainer_cls = Trainer
        trainer_kwargs = {}

    trainer = trainer_cls(
        model=model,
        args=args_hf,
        train_dataset=train_ds,
        eval_dataset=eval_small,
        tokenizer=tok,
        data_collator=collator,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    # Train
    print("\n🏃‍♂️ Starting training...")
    trainer.train()
    print("\n🎉 Training complete.")

    # Final full validation
    print("\n📊 Final evaluation on FULL val set...")
    final_metrics = trainer.evaluate(eval_dataset=val_ds)
    for k, v in final_metrics.items():
        print(f"  • {k}: {v:.6f}" if isinstance(v, float) else f"  • {k}: {v}")

    # Save model/tokenizer
    print("\n💾 Saving model & tokenizer...")
    trainer.save_model()
    tok.save_pretrained(tcfg.output_dir)

    # Save trainer logs (CSV & curves)
    (Path(tcfg.output_dir) / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2)
    )
    try:
        import csv as _csv, matplotlib.pyplot as plt
        rows = trainer.state.log_history
        keys = sorted({k for d in rows for k in d})
        with open(Path(tcfg.output_dir) / "training_log.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        tr = [(d["step"], d["loss"]) for d in rows if "loss" in d]
        ev = [(d["step"], d["eval_loss"]) for d in rows if "eval_loss" in d]
        plt.figure()
        if tr: plt.plot([s for s,_ in tr], [y for _,y in tr], label="train loss")
        if ev: plt.plot([s for s,_ in ev], [y for _,y in ev], label="eval loss")
        plt.xlabel("step"); plt.ylabel("loss"); plt.title("Loss curves"); plt.legend()
        plt.tight_layout(); plt.savefig(Path(tcfg.output_dir) / "loss_curves.png", dpi=150)
        print("Saved: training_log.csv, trainer_log_history.json, loss_curves.png")
    except Exception as e:
        print(f"Saved trainer_log_history.json (CSV/plot skipped): {e}")

if __name__ == "__main__":
    main()
