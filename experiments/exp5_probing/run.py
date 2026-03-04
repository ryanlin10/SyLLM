#!/usr/bin/env python3
"""
Experiment 5: Probing Classifiers

Train small linear classifiers on frozen hidden-state activations to test
what the model has learned. Runs on base model and both adapted models.

Probes:
  A. Validity detection — probe on last-token hidden state predicts whether
     the argument (premises → conclusion) is logically valid.

  B. Premise count — probe predicts number of premises from hidden state
     (structural sanity check).

  C. Cross-model comparison — same probes on base / stage0 / stage1.
     If probe accuracy jumps after fine-tuning → model developed new
     internal representations.
     If already high on base model → LoRA unlocked existing capability.

Architecture:
  - Load model in 4-bit (QLoRA) via transformers + bitsandbytes
  - Apply LoRA adapter WITHOUT merging (supports 4-bit)
  - Extract last-token hidden states from 6 evenly-spaced layers
  - Train logistic regression on 80% examples, test on 20%
  - Report layer-wise accuracy
"""

import gc
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

BASE_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
STAGE0_ADAPTER = str(
    PROJECT_ROOT / "mistralai_Mistral_Small_3.2_24B_Instruct_2506_20260302_175851" / "final"
)
STAGE1_ADAPTER = str(
    PROJECT_ROOT / "mistralai_Mistral_Small_3.2_24B_Instruct_2506_20260302_223714" / "final"
)
STAGE0_DATA = str(
    PROJECT_ROOT / "chain_stage0_n10000_len2-20_comp1_20260302_172126.jsonl"
)

SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. "
    "Given the following premises, derive their valid conclusion."
)

N_VALID = 100
N_INVALID = 100
SEED = 45
MAX_INPUT_LEN = 768   # tokens
PROBE_LAYER_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BATCH_SIZE = 2        # Small batch to avoid OOM during hidden state extraction


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_data(n_valid=N_VALID, n_invalid=N_INVALID, seed=SEED):
    rng = random.Random(seed)

    all_examples = []
    with open(STAGE0_DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                all_examples.append(json.loads(line))

    pool = rng.sample(all_examples, min(n_valid + n_invalid + 50, len(all_examples)))

    valid_subset = pool[:n_valid]
    swap_pool = pool[n_valid:n_valid + n_invalid + 50]

    def get_conc(ex):
        content = ex.get("content", "")
        m = re.search(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", content, re.DOTALL)
        return m.group(1).strip() if m else ""

    def get_premises(ex):
        return [p["text"] for p in ex["premises"]]

    swap_conclusions = [get_conc(ex) for ex in swap_pool]
    rng.shuffle(swap_conclusions)

    def format_input(premises, conclusion, is_valid_marker=None):
        """Format as plain text prompt (no special tokenizer needed)."""
        parts = [f"[INST] {SYSTEM_PROMPT}\n\n"]
        for p in premises:
            parts.append(f"<PREMISE> {p} </PREMISE> ")
        parts.append(f"\nProposed conclusion: {conclusion} [/INST]")
        return "".join(parts)

    valid_msgs, invalid_msgs = [], []
    metadata = []

    for ex in valid_subset:
        premises = get_premises(ex)
        conc = get_conc(ex)
        if not conc:
            continue
        msg = format_input(premises, conc)
        valid_msgs.append(msg)
        vn = json.loads(ex.get("verifier_notes", "{}"))
        metadata.append({
            "label": 1,
            "n_premises": len(premises),
            "logic_order": vn.get("logic_order", "unknown"),
            "chain_length": vn.get("chain_length", 0),
        })

    for i, ex in enumerate(swap_pool[:n_invalid]):
        premises = get_premises(ex)
        conc = swap_conclusions[i % len(swap_conclusions)]
        orig_conc = get_conc(ex)
        if conc == orig_conc:
            conc = swap_conclusions[(i + 1) % len(swap_conclusions)]
        msg = format_input(premises, conc)
        invalid_msgs.append(msg)
        vn = json.loads(ex.get("verifier_notes", "{}"))
        metadata.append({
            "label": 0,
            "n_premises": len(premises),
            "logic_order": vn.get("logic_order", "unknown"),
            "chain_length": vn.get("chain_length", 0),
        })

    all_msgs = valid_msgs + invalid_msgs
    labels = np.array([m["label"] for m in metadata])
    n_premises = np.array([m["n_premises"] for m in metadata])

    print(f"Dataset: {len(valid_msgs)} valid + {len(invalid_msgs)} invalid = {len(all_msgs)} total")
    return all_msgs, labels, n_premises, metadata


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    messages: List[str],
    model_name: str,
    adapter_path: Optional[str],
    probe_layer_indices: List[int],
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_INPUT_LEN,
) -> Dict[int, np.ndarray]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\nLoading tokenizer: {model_name}")
    # Apply predictor patch for Mistral tokenizer (loads mistral_common fixes)
    try:
        import src.inference.predictor  # noqa — applies _patch_mistral_tokenizer
    except Exception:
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True, padding_side="left"
        )
    except Exception as e:
        print(f"  Standard tokenizer failed ({e}), trying mistral_common...")
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub
        # Fallback: use tiktoken-based tokenizer for encoding lengths
        raise RuntimeError(f"Could not load tokenizer: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading base model (4-bit)...")
    load_kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    try:
        from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
        try:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_name, attn_implementation="flash_attention_2", **load_kwargs
            )
            print("  Loaded as Mistral3ForConditionalGeneration (Flash Attention 2)")
        except Exception:
            model = Mistral3ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            print("  Loaded as Mistral3ForConditionalGeneration (SDPA)")
    except (ImportError, Exception) as e:
        print(f"  Mistral3 class failed ({e}), using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Apply LoRA adapter WITHOUT merging (merge not supported for 4-bit)
    if adapter_path is not None:
        print(f"Loading LoRA adapter (no merge): {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    model.eval()
    try:
        cfg = model.config
        num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(
            getattr(cfg, "text_config", None), "num_hidden_layers", 40)
    except Exception:
        num_layers = 40
    print(f"Model has {num_layers} layers; extracting from layers: {probe_layer_indices}")

    all_layer_states = {li: [] for li in probe_layer_indices}

    with torch.no_grad():
        for i in range(0, len(messages), batch_size):
            batch_msgs = messages[i:i+batch_size]

            enc = tokenizer(
                batch_msgs,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
                return_attention_mask=True,
            )
            input_ids = enc["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
            attention_mask = enc["attention_mask"].to(input_ids.device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract last real token's hidden state for each example
            for b_idx in range(len(batch_msgs)):
                last_tok = int(attention_mask[b_idx].sum().item()) - 1
                for li in probe_layer_indices:
                    hs_idx = li + 1  # +1 for embedding layer
                    if hs_idx >= len(out.hidden_states):
                        hs_idx = -1
                    hs = out.hidden_states[hs_idx][b_idx, last_tok, :].float().cpu().numpy()
                    all_layer_states[li].append(hs)

            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(messages):
                print(f"  Processed {min(i + batch_size, len(messages))}/{len(messages)} examples")

    result = {li: np.stack(all_layer_states[li]) for li in probe_layer_indices
              if all_layer_states[li]}

    # Free GPU memory
    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(X_train, y_train, X_test, y_test, task="classification"):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, r2_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if task == "classification":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED, solver="lbfgs")
        clf.fit(X_tr, y_train)
        score = accuracy_score(y_test, clf.predict(X_te))
    else:
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, y_train)
        score = r2_score(y_test, reg.predict(X_te))
    return float(score)


def run_probes(messages, labels, n_premises, hidden_states, probe_layer_indices, model_label):
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(messages))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=labels
    )

    results = {"model": model_label, "validity_probe": {}, "premise_count_probe": {}}
    print(f"\n  Probe results for {model_label}:")

    for li in probe_layer_indices:
        X = hidden_states.get(li)
        if X is None or len(X) != len(messages):
            continue

        acc = train_probe(X[train_idx], labels[train_idx], X[test_idx], labels[test_idx])
        r2 = train_probe(X[train_idx], n_premises[train_idx],
                         X[test_idx], n_premises[test_idx], task="regression")
        results["validity_probe"][li] = acc
        results["premise_count_probe"][li] = r2
        print(f"    Layer {li:3d}: validity_acc={acc:.4f}  premise_count_R2={r2:.4f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_probe_comparison(all_results, probe_key, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Base model": "#9E9E9E", "Stage 0": "#2196F3", "Stage 1": "#F44336"}
    for res in all_results:
        label = res["model"]
        data = res.get(probe_key, {})
        if not data:
            continue
        layers = sorted(data.keys())
        vals = [data[l] for l in layers]
        ax.plot(layers, vals, marker="o", label=label,
                color=colors.get(label, "#000"), linewidth=2, markersize=7)
    if probe_key == "validity_probe":
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    messages, labels, n_premises, metadata = load_data()

    # Determine probe layers from model config
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
        num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(
            getattr(cfg, "text_config", None), "num_hidden_layers", 40)
    except Exception:
        num_layers = 40
    print(f"Model has {num_layers} layers")

    probe_layer_indices = sorted(set(
        int(f * (num_layers - 1)) for f in PROBE_LAYER_FRACTIONS
    ))
    print(f"Probing layers: {probe_layer_indices}")

    all_results = []

    model_variants = [
        ("Base model", None),
        ("Stage 0", STAGE0_ADAPTER),
        ("Stage 1", STAGE1_ADAPTER),
    ]

    for model_label, adapter_path in model_variants:
        print(f"\n{'='*60}")
        print(f"Processing: {model_label}")

        hidden_states = extract_hidden_states(
            messages=messages,
            model_name=BASE_MODEL,
            adapter_path=adapter_path,
            probe_layer_indices=probe_layer_indices,
        )

        # Save hidden state shapes for reference (not full arrays, too large)
        hs_meta = {str(li): {"shape": list(hs.shape), "mean": float(hs.mean()), "std": float(hs.std())}
                   for li, hs in hidden_states.items()}
        with open(OUT_DIR / f"hs_meta_{model_label.replace(' ', '_').lower()}.json", "w") as f:
            json.dump(hs_meta, f, indent=2)

        res = run_probes(messages, labels, n_premises, hidden_states, probe_layer_indices, model_label)
        all_results.append(res)

    # Save probe results
    with open(OUT_DIR / "probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plots
    plot_probe_comparison(
        all_results, "validity_probe",
        "Validity Detection Probe: Accuracy by Layer\n(1=valid argument, 0=wrong conclusion)",
        "Accuracy",
        OUT_DIR / "validity_probe_by_layer.png"
    )
    plot_probe_comparison(
        all_results, "premise_count_probe",
        "Premise Count Probe: R² by Layer",
        "R² score",
        OUT_DIR / "premise_count_probe_by_layer.png"
    )

    # Heatmap: model × layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax_idx, (pk, yl, cm) in enumerate([
        ("validity_probe", "Validity accuracy", "Blues"),
        ("premise_count_probe", "Premise count R²", "Greens"),
    ]):
        ax = axes[ax_idx]
        grid, m_labels = [], []
        for res in all_results:
            data = res.get(pk, {})
            if not data:
                continue
            layers = sorted(data.keys())
            grid.append([data[l] for l in layers])
            m_labels.append(res["model"])
        if not grid:
            continue
        grid = np.array(grid)
        im = ax.imshow(grid, aspect="auto", cmap=cm, vmin=0, vmax=1)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, fontsize=8)
        ax.set_yticks(range(len(m_labels)))
        ax.set_yticklabels(m_labels)
        ax.set_xlabel("Layer index"); ax.set_title(yl)
        plt.colorbar(im, ax=ax)
    plt.suptitle("Probing Classifier Results: Model × Layer", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "probe_heatmap.png"), dpi=150)
    plt.close()

    # Text summary
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("=== Experiment 5: Probing Classifiers ===\n\n")
        f.write(f"Dataset: {len(messages)} examples ({int(labels.sum())} valid, {int((1-labels).sum())} invalid)\n")
        f.write(f"Probe layers: {probe_layer_indices}\n\n")

        f.write("Probe A: Validity Detection (Logistic Regression)\n")
        f.write("  Random baseline: 0.500\n\n")
        for res in all_results:
            f.write(f"  [{res['model']}]\n")
            for li in sorted(res["validity_probe"].keys()):
                f.write(f"    Layer {li:3d}: {res['validity_probe'][li]:.4f}\n")
            if res["validity_probe"]:
                best = max(res["validity_probe"], key=res["validity_probe"].get)
                f.write(f"    Best: Layer {best} = {res['validity_probe'][best]:.4f}\n\n")

        f.write("Probe B: Premise Count (Ridge Regression, R²)\n\n")
        for res in all_results:
            f.write(f"  [{res['model']}]\n")
            for li in sorted(res["premise_count_probe"].keys()):
                f.write(f"    Layer {li:3d}: {res['premise_count_probe'][li]:.4f}\n")
            if res["premise_count_probe"]:
                best = max(res["premise_count_probe"], key=res["premise_count_probe"].get)
                f.write(f"    Best: Layer {best} = {res['premise_count_probe'][best]:.4f}\n\n")

        f.write("Interpretation:\n")
        f.write("  Validity probe accuracy jump from base→adapted: model developed new representations.\n")
        f.write("  High base accuracy + low generation: LoRA unlocked existing capability.\n")
        f.write("  Premise count R²~1 in early layers: model encodes structural metadata early.\n")

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
