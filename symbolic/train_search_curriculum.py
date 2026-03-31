import multiprocessing
import os
import json
import pickle
import sysconfig
import time
from os import listdir, makedirs, popen
from os.path import isfile, isdir
from random import sample, randrange, choice, shuffle, seed, getstate, setstate, Random
from collections import deque
from sys import stdout

import numpy as np
from pybind11.__main__ import print_includes
from io import StringIO
import torch
from torch import nn, LongTensor
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader

from Sophia import SophiaG
from torch.optim import AdamW
from gpt2 import Transformer, TransformerLayer, ToeplitzMode, AblationMode, PositionEmbedding


def build_module(name):
    import sys
    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        print_includes()
        includes = sys.stdout.getvalue().strip()
        sys.stdout.close()
        sys.stdout = old_stdout
    except Exception as e:
        raise e
    finally:
        sys.stdout = old_stdout

    python_extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if sys.platform == "darwin":
        # macOS command
        command = (
            f"g++ -std=c++11 -Ofast -DNDEBUG -fno-stack-protector "
            f"-Wall -Wpedantic -undefined dynamic_lookup -shared -fPIC "
            f"{includes} -I. {name}.cpp -o {name}{python_extension_suffix}"
        )
    else:
        # Non-macOS command
        command = (
            f"g++ -Ofast -std=c++11 -DNDEBUG -fno-stack-protector "
            f"-Wall -Wpedantic -shared -fPIC "
            f"{includes} -I. {name}.cpp -o {name}{python_extension_suffix}"
        )
    print(command)
    if os.system(command) != 0:
        print(f"ERROR: Unable to compile `{name}.cpp`.")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    try:
        from os.path import getmtime
        from importlib.util import find_spec
        generator_spec = find_spec('generator')
        if generator_spec == None:
            raise ModuleNotFoundError
        if getmtime(generator_spec.origin) < getmtime('generator.cpp'):
            print("C++ module `generator` is out-of-date. Compiling from source...")
            build_module("generator")
        import generator
    except ModuleNotFoundError:
        print("C++ module `generator` not found. Compiling from source...")
        build_module("generator")
        import generator
    except ImportError:
        print("Error loading C++ module `generator`. Compiling from source...")
        build_module("generator")
        import generator
    print("C++ module `generator` loaded.")

RESERVED_INDICES = (0,)


# ================== Curriculum helpers ==================

def effective_search_L(alpha, n, max_lookahead_cap=None,
                       tokens_per_edge=3, fixed_tokens=4, reserve_edges=1):
    """Compute effective lookahead L for given alpha and input size n."""
    edges_unscaled = max(0, (n - fixed_tokens) // tokens_per_edge)
    max_edges = int(alpha * edges_unscaled)
    safe_edges = max(0, max_edges - reserve_edges)
    L_edges = safe_edges // 2
    L_tokens = max(0, (n - fixed_tokens) // (2 * tokens_per_edge))
    L_eff = min(L_edges, L_tokens)
    if max_lookahead_cap:
        L_eff = min(L_eff, int(max_lookahead_cap))
    return L_eff


def alpha_for_lookahead(L_target, n, tokens_per_edge=3, fixed_tokens=4, reserve_edges=1):
    """Compute minimum alpha needed to achieve target effective lookahead L."""
    if L_target <= 0:
        return 0.0
    edges_unscaled = max(1, (n - fixed_tokens) // tokens_per_edge)
    needed_alpha = (2 * L_target + reserve_edges) / edges_unscaled
    return min(max(needed_alpha, 0.0), 1.0)


# ================== Plotting ==================

def _smooth(values, window):
    """Simple moving average."""
    if len(values) < window or window < 1:
        return list(range(len(values))), values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='valid')
    x_offset = list(range(window - 1, len(values)))
    return x_offset, smoothed.tolist()


def _draw_stage_markers(ax, loss_history, x_key="step", colors=None):
    """Draw vertical lines at stage transitions."""
    if not loss_history:
        return
    stages = [h["stage"] for h in loss_history]
    max_stage = max(stages)
    if colors is None:
        import matplotlib.pyplot as plt
        colors = plt.cm.tab10(np.linspace(0, 1, max(max_stage, 1)))

    prev_stage = stages[0]
    stage_info = {}
    for h in loss_history:
        s = h["stage"]
        if s not in stage_info:
            stage_info[s] = h

    for i, h in enumerate(loss_history):
        if h["stage"] != prev_stage:
            x = h[x_key]
            ax.axvline(x=x, color=colors[(h["stage"] - 1) % 10], linestyle='--', alpha=0.7)
            L = h.get("effective_L", "?")
            ax.text(x, ax.get_ylim()[1] * 0.95, f'S{h["stage"]}\nL={L}',
                    fontsize=7, ha='left', va='top')
            prev_stage = h["stage"]


def _draw_resume_markers(ax, loss_history, x_key="step"):
    """Draw vertical dashed red lines at resume points."""
    for h in loss_history:
        if h.get("resume"):
            ax.axvline(x=h.get(x_key, 0), color='red', linestyle=':', alpha=0.5, linewidth=1.5)


def plot_overall_loss(loss_history, output_dir, n_stages, metadata=None):
    """Plot training loss across all stages."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not loss_history:
        return

    steps = [h["step"] for h in loss_history]
    losses = [h["loss"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, losses, alpha=0.3, linewidth=0.5, color='blue')

    window = min(100, len(losses) // 10) or 1
    x_sm, losses_sm = _smooth(losses, window)
    if len(losses_sm) > 1:
        ax.plot([steps[i] for i in x_sm], losses_sm, linewidth=2, color='blue',
                label=f'Smoothed (w={window})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Training Loss (Overall)')

    if metadata:
        parts = []
        if metadata.get("model"): parts.append(metadata["model"])
        if metadata.get("batch_size"): parts.append(f'BS={metadata["batch_size"]}')
        if metadata.get("learning_rate"): parts.append(f'LR={metadata["learning_rate"]:.1e}')
        if metadata.get("max_lookahead"): parts.append(f'maxL={metadata["max_lookahead"]}')
        fig.suptitle(" | ".join(parts), fontsize=9, color='gray', y=0.02)

    _draw_stage_markers(ax, loss_history)
    _draw_resume_markers(ax, loss_history)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "loss_overall.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved loss_overall.png")


def plot_accuracy(loss_history, output_dir, n_stages, metadata=None):
    """Plot training and test accuracy across all stages."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not loss_history:
        return

    steps = [h["step"] for h in loss_history]
    train_acc = [h.get("training_acc", 0) for h in loss_history]
    test_acc = [h.get("test_acc", 0) for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Training accuracy
    ax.plot(steps, train_acc, alpha=0.3, linewidth=0.5, color='blue')
    window = min(50, len(train_acc) // 10) or 1
    x_sm, train_sm = _smooth(train_acc, window)
    if len(train_sm) > 1:
        ax.plot([steps[i] for i in x_sm], train_sm, linewidth=2, color='blue',
                label=f'Train Acc (smooth w={window})')

    # Test accuracy
    ax.plot(steps, test_acc, alpha=0.3, linewidth=0.5, color='green')
    x_sm, test_sm = _smooth(test_acc, window)
    if len(test_sm) > 1:
        ax.plot([steps[i] for i in x_sm], test_sm, linewidth=2, color='green',
                label=f'Test Acc (smooth w={window})')

    # Threshold line
    thresh = metadata.get("accuracy_threshold", 0.99) if metadata else 0.99
    ax.axhline(y=thresh, color='red', linestyle=':', alpha=0.5, label=f'{thresh:.0%} threshold')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Training & Test Accuracy')

    if metadata:
        parts = []
        if metadata.get("model"): parts.append(metadata["model"])
        if metadata.get("batch_size"): parts.append(f'BS={metadata["batch_size"]}')
        if metadata.get("learning_rate"): parts.append(f'LR={metadata["learning_rate"]:.1e}')
        if metadata.get("max_lookahead"): parts.append(f'maxL={metadata["max_lookahead"]}')
        fig.suptitle(" | ".join(parts), fontsize=9, color='gray', y=0.02)

    _draw_stage_markers(ax, loss_history)
    _draw_resume_markers(ax, loss_history)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(output_dir, "accuracy_overall.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved accuracy_overall.png")


def plot_stage_loss(loss_history, stage, output_dir, metadata=None):
    """Plot loss for a single stage with exponential fit."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    stage_data = [h for h in loss_history if h["stage"] == stage]
    if len(stage_data) < 5:
        return

    steps = [h["step"] for h in stage_data]
    losses = [h["loss"] for h in stage_data]
    alpha = stage_data[0].get("alpha")
    effective_L = stage_data[0].get("effective_L")

    start_step = steps[0]
    rel_steps = [s - start_step for s in steps]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rel_steps, losses, alpha=0.3, linewidth=0.5, color='blue', label='Raw')

    window = min(50, len(losses) // 3) or 1
    x_sm, losses_sm = _smooth(losses, window)
    if len(losses_sm) > 1:
        ax.plot([rel_steps[i] for i in x_sm], losses_sm, linewidth=2, color='blue',
                label=f'Smoothed (w={window})')

    # Try exponential fit
    try:
        from scipy.optimize import curve_fit
        if len(losses) >= 20:
            t = np.array(rel_steps, dtype=float)
            y = np.array(losses, dtype=float)
            def exp_decay(t, a, b, c):
                return a * np.exp(-b * t) + c
            a0 = max(y[0] - y[-1], 0.01)
            c0 = max(y[-1], 0.001)
            b0 = 1.0 / max(len(t) / 3, 1)
            popt, _ = curve_fit(exp_decay, t, y, p0=[a0, b0, c0],
                                bounds=([0, 1e-10, 0], [np.inf, 1.0, np.inf]), maxfev=5000)
            a, b, c = popt
            y_pred = exp_decay(t, a, b, c)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            if r2 > 0.8:
                ax.plot(rel_steps, y_pred.tolist(), '--', linewidth=2, color='red',
                        label=f'Exp fit (R\u00b2={r2:.3f})')
                ax.axhline(y=c, color='green', linestyle=':', alpha=0.7,
                           label=f'Asymptote={c:.4f}')
    except Exception:
        pass

    ax.set_xlabel('Epochs in Stage')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')

    title = f'Stage {stage} Loss ({len(stage_data)} epochs, final={losses[-1]:.4f})'
    if effective_L is not None:
        title += f' | L={effective_L}'
    if alpha is not None:
        title += f' | \u03b1={alpha:.4f}'
    ax.set_title(title)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_stage_{stage}.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved loss_stage_{stage}.png")


def plot_loss_vs_walltime(loss_history, output_dir, n_stages, metadata=None):
    """Plot loss vs wall clock time."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not loss_history or not any(h.get("wall_time", 0) > 0 for h in loss_history):
        return

    wall_hours = [h.get("wall_time", 0) / 3600 for h in loss_history]
    losses = [h["loss"] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wall_hours, losses, alpha=0.3, linewidth=0.5, color='blue')

    window = min(100, len(losses) // 10) or 1
    x_sm, losses_sm = _smooth(losses, window)
    if len(losses_sm) > 1:
        ax.plot([wall_hours[i] for i in x_sm], losses_sm, linewidth=2, color='blue',
                label=f'Smoothed (w={window})')

    ax.set_xlabel('Wall Clock Time (hours)')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Training Loss vs Wall Time')

    # Stage markers using wall_time
    stages = [h["stage"] for h in loss_history]
    prev_stage = stages[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(max(stages), n_stages)))
    for i, h in enumerate(loss_history):
        if h["stage"] != prev_stage:
            ax.axvline(x=wall_hours[i], color=colors[(h["stage"] - 1) % 10], linestyle='--', alpha=0.7)
            L = h.get("effective_L", "?")
            ax.text(wall_hours[i], ax.get_ylim()[1] * 0.95, f'S{h["stage"]}\nL={L}',
                    fontsize=7, ha='left', va='top')
            prev_stage = h["stage"]

    _draw_resume_markers(ax, loss_history, x_key="wall_time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_vs_walltime.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved loss_vs_walltime.png")


def plot_stage_eval(stage_eval_history, output_dir, metadata=None):
    """Plot test accuracy vs effective L after each stage."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not stage_eval_history or len(stage_eval_history) < 2:
        return

    has_L = "effective_L" in stage_eval_history[0]
    if has_L:
        x_vals = [h["effective_L"] for h in stage_eval_history]
        x_label = "Effective Lookahead (L)"
    else:
        x_vals = [h["stage"] for h in stage_eval_history]
        x_label = "Stage"

    test_acc = [h["test_acc"] for h in stage_eval_history]
    train_acc = [h["training_acc"] for h in stage_eval_history]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_vals, train_acc, 'o-', color='#2196F3', label='Train Acc (at promotion)', markersize=6)
    ax1.plot(x_vals, test_acc, 's-', color='#4CAF50', label='Test Acc (alpha=1.0)', markersize=6)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    if "test_loss" in stage_eval_history[0]:
        ax2 = ax1.twinx()
        test_losses = [h["test_loss"] for h in stage_eval_history]
        ax2.plot(x_vals, test_losses, '^--', color='#F44336', label='Test Loss (alpha=1.0)', markersize=5)
        ax2.set_ylabel('Test Loss', color='#F44336')
        ax2.tick_params(axis='y', labelcolor='#F44336')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    else:
        ax1.legend(loc='lower right')

    for i, h in enumerate(stage_eval_history):
        ax1.annotate(f'S{h["stage"]}', (x_vals[i], test_acc[i]),
                     textcoords="offset points", xytext=(0, 8), fontsize=7, ha='center')

    ax1.set_title('Held-Out Eval (alpha=1.0) After Each Stage')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_eval.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved stage_eval.png ({len(stage_eval_history)} stages)")


def plot_eval_acc_vs_step(stage_eval_history, loss_history, output_dir, metadata=None):
    """Plot test accuracy vs training epoch."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not stage_eval_history or len(stage_eval_history) < 2:
        return

    steps = [h["step"] for h in stage_eval_history]
    test_acc = [h["test_acc"] * 100 for h in stage_eval_history]
    train_acc = [h["training_acc"] * 100 for h in stage_eval_history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, test_acc, 's-', color='#4CAF50', label='Test Acc (alpha=1.0)', markersize=5)
    ax.plot(steps, train_acc, 'o-', color='#2196F3', label='Train Acc (at promotion)', markersize=5, alpha=0.6)

    for i, h in enumerate(stage_eval_history):
        if "effective_L" in h:
            ax.annotate(f'L={h["effective_L"]}', (steps[i], test_acc[i]),
                        textcoords="offset points", xytext=(0, 8), fontsize=7, ha='center')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Eval Accuracy vs Epoch')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if loss_history:
        _draw_resume_markers(ax, loss_history)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_acc_vs_step.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved eval_acc_vs_step.png")


def generate_all_plots(output_dir, loss_history, n_stages, metadata=None,
                       stage_eval_history=None):
    """Generate all available plots."""
    if not loss_history:
        return
    os.makedirs(output_dir, exist_ok=True)

    plot_overall_loss(loss_history, output_dir, n_stages, metadata)
    plot_accuracy(loss_history, output_dir, n_stages, metadata)
    plot_loss_vs_walltime(loss_history, output_dir, n_stages, metadata)

    stages_seen = sorted(set(h["stage"] for h in loss_history))
    for stage in stages_seen:
        plot_stage_loss(loss_history, stage, output_dir, metadata)

    if stage_eval_history and len(stage_eval_history) >= 2:
        plot_stage_eval(stage_eval_history, output_dir, metadata)
        plot_eval_acc_vs_step(stage_eval_history, loss_history, output_dir, metadata)


def save_histories(output_dir, loss_history, stage_eval_history):
    """Save history data as JSON for later plot regeneration."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)
    with open(os.path.join(output_dir, "stage_eval_history.json"), "w") as f:
        json.dump(stage_eval_history, f)


# ================== Model helpers ==================

class Node(object):
    def __init__(self, id):
        self.id = id
        self.children = []
        self.parents = []

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return 'n(' + str(self.id) + ')'

    def __repr__(self):
        return 'n(' + str(self.id) + ')'

def binomial_confidence_int(p, n):
    return 1.96 * np.sqrt(p * (1.0 - p) / n)

def evaluate_model(model, inputs, outputs):
    device = next(model.parameters()).device
    inputs = torch.tensor(inputs)
    outputs = torch.tensor(outputs)
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    max_input_size = inputs.shape[1]

    if outputs.dim() == 2:
        loss_func = BCEWithLogitsLoss(reduction='mean')
    else:
        loss_func = CrossEntropyLoss(reduction='mean')
    logits, _ = model(inputs)
    loss = loss_func(logits[:, -1, :], outputs).item()

    predictions = torch.argmax(logits[:, -1, :], 1)
    if outputs.dim() == 2:
        acc = torch.sum(torch.gather(outputs, 1, torch.argmax(logits[:,-1,:],dim=1).unsqueeze(1))).item() / outputs.size(0)
    else:
        acc = sum(predictions == outputs).item() / len(predictions)
    return acc, loss, predictions

def unique(x):
    y = []
    for e in x:
        if e not in y:
            y.append(e)
    return y

def train(max_input_size, dataset_size, distribution, max_lookahead, seed_value, nlayers, nhead, hidden_dim,
          bidirectional, pos_emb, learnable_token_emb, toeplitz_attn, toeplitz_reg, toeplitz_pos_only,
          add_padding, ablate, pre_ln, curriculum_mode, looped, task, warm_up, batch_size, learning_rate,
          update_rate, grad_accumulation_steps, max_edges, distance_from_start, max_prefix_vertices, loss,
          base_lookahead, lookahead_step, accuracy_threshold, accuracy_window_size=100, test_eval_every=50,
          lr_schedule='constant', lr_warmup_epochs=0, lr_min_ratio=0.01, lr_cosine_epochs=20000,
          output_dir='search_results',
          optimizer_type='sophiag', use_amp=False, use_compile=False, hessian_update_interval=0):

    generator.set_seed(seed_value)
    seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    PADDING_TOKEN = (max_input_size-5) // 3 + 3
    BATCH_SIZE = batch_size // grad_accumulation_steps
    print('Number of available CPUs: {}'.format(os.cpu_count()))
    stdout.flush()

    if curriculum_mode == 'y':
        print("Using curriculum learning (linear L increase)")
        n_stages = (max_lookahead - base_lookahead) // lookahead_step + 1
        print("  Stages: {} (L={} to L={}, step={})".format(
            n_stages, base_lookahead, max_lookahead, lookahead_step))
        for s in range(n_stages):
            L = min(base_lookahead + s * lookahead_step, max_lookahead)
            a = alpha_for_lookahead(L, max_input_size)
            print("  Stage {}: L={}, alpha={:.4f}".format(s + 1, L, a))
        stdout.flush()
    else:
        n_stages = 1

    if loss == "bce":
        print("Using BCE loss")

    if curriculum_mode != 'n' and dataset_size != -1:
        print('ERROR: Curriculum learning is only supported with streaming training (i.e. dataset_size = -1).')
        stdout.flush()
        return

    if distribution in ("crafted", "crafted_no_prefix", "star") and max_lookahead == None:
        print('ERROR: Crafted or star training distribution is selected but `max_lookahead` argument is missing.')
        stdout.flush()
        return

    if distribution == "simple" and max_lookahead != None:
        print('ERROR: `max_lookahead` is not supported with the simple training distribution.')
        stdout.flush()
        return

    if task != "search":
        print('ERROR: This script is configured for search task only.')
        stdout.flush()
        return

    if max_lookahead == None:
        max_lookahead = -1

    # Calculate max_edges if not provided
    if max_edges == None:
        max_edges = (max_input_size - 5) // 3

    # first reserve some data for OOD testing
    random_state = getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.get_rng_state()

    reserved_inputs = set()
    NUM_TEST_SAMPLES = 10000

    # Generate reserved test data for search task
    print('Reserving OOD test data for search task')
    stdout.flush()
    gen_eval_start_time = time.perf_counter()

    # Generate test data with full complexity (alpha=1.0)
    eval_inputs, eval_outputs, eval_labels, _ = generator.generate_training_set(
        max_input_size, NUM_TEST_SAMPLES, max_lookahead, max_edges, reserved_inputs,
        distance_from_start, max_prefix_vertices if max_prefix_vertices != None else -1, True, 1.0)

    print('Done. Throughput: {} examples/s'.format(NUM_TEST_SAMPLES / (time.perf_counter() - gen_eval_start_time)))
    for i in range(eval_inputs.shape[0]):
        reserved_inputs.add(tuple([x for x in eval_inputs[i,:] if x != PADDING_TOKEN]))

    if batch_size < eval_inputs.shape[0]:
        eval_inputs = eval_inputs[:batch_size]
        eval_outputs = eval_outputs[:batch_size]

    train_filename = 'train{}_search_inputsize{}_maxlookahead{}_{}seed{}.pkl'.format(
        dataset_size, max_input_size, max_lookahead, 'padded_' if add_padding else '', seed_value)

    prefix = output_dir.rstrip('/') + '/'

    if not torch.cuda.is_available():
        print("WARNING: CUDA device is not available, using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # compute the checkpoint filenames
    filename = prefix + 'checkpoints_search_{}layer_inputsize{}_maxlookahead{}_seed{}_train{}'.format(
        nlayers, max_input_size, max_lookahead, seed_value, dataset_size if dataset_size != -1 else 'streaming')
    if hidden_dim != 16:
        filename += '_hiddendim{}'.format(hidden_dim)
    if bidirectional:
        filename += '_nomask'
    if pos_emb == 'none':
        filename += '_NoPE'
    elif pos_emb == 'rotary':
        filename += '_RoPE'
    if learnable_token_emb:
        filename += '_learntokemb'
    if ablate != "none":
        filename += '_ablate' + ablate
    if toeplitz_attn:
        filename += '_toeplitz'
        if toeplitz_pos_only:
            filename += 'pos'
    if toeplitz_reg != 0.0:
        filename += '_toeplitz'
        if toeplitz_pos_only:
            filename += 'pos'
        filename += str(toeplitz_reg)
    if not pre_ln:
        filename += '_postLN'
    if add_padding:
        filename += '_padded'
    if curriculum_mode == 'y':
        filename += '_curriculum_linearL'
    if looped:
        filename += '_looped'
    if distribution != 'crafted':
        filename += '_' + distribution.replace('_', '-')
    if nhead != 1:
        filename += '_nhead' + str(nhead)
    if warm_up != 0:
        filename += '_warmup' + str(warm_up)
    if batch_size != 2**8:
        filename += '_batchsize' + str(batch_size)
    if learning_rate != 1.0e-5:
        filename += '_lr' + str(learning_rate)
    if update_rate != 2 ** 18:
        filename += '_update' + str(update_rate)
    if loss == "bce":
        filename += '_bce'
    if optimizer_type != 'sophiag':
        filename += '_' + optimizer_type
    if use_amp:
        filename += '_amp'
    if use_compile:
        filename += '_compile'

    if isdir(filename):
        existing_epochs = [int(ckpt[(ckpt.rfind('epoch') + len('epoch')):-len('.pt')]) for ckpt in listdir(filename) if ckpt.startswith('epoch')]
    else:
        existing_epochs = []
        makedirs(filename)

    ntoken = (max_input_size-5) // 3 + 5
    d_hid = ntoken + hidden_dim
    dropout = 0
    if ablate == "none":
        ablation_mode = AblationMode.NO_ABLATION
    elif ablate == "attn_linear":
        ablation_mode = AblationMode.ABLATE_ATTN_LINEAR
    elif ablate == "attn_linear_projv":
        ablation_mode = AblationMode.ABLATE_ATTN_LINEAR_PROJV
    if toeplitz_attn and toeplitz_pos_only:
        toeplitz = ToeplitzMode.LOWER_RIGHT
    elif toeplitz_attn and not toeplitz_pos_only:
        toeplitz = ToeplitzMode.BLOCK
    else:
        toeplitz = ToeplitzMode.NONE
    if pos_emb == "absolute":
        pos_emb_mode = PositionEmbedding.ABSOLUTE
    elif pos_emb == "rotary":
        pos_emb_mode = PositionEmbedding.ROTARY
    else:
        pos_emb_mode = PositionEmbedding.NONE

    _restored_optimizer_state = None
    _restored_accuracy_window = None
    _restored_lr_scheduler_state = None

    if len(existing_epochs) == 0:
        print("Building search model.")
        model = Transformer(
            layers=nlayers,
            pad_idx=PADDING_TOKEN,
            words=ntoken,
            seq_len=max_input_size,
            heads=nhead,
            dims=max(ntoken, d_hid),
            rate=1,
            dropout=dropout,
            bidirectional=bidirectional,
            pos_emb=pos_emb_mode,
            learn_token_emb=learnable_token_emb,
            ablate=ablation_mode,
            toeplitz=toeplitz,
            pre_ln=pre_ln,
            looped=looped
        )
        model.to(device)
        epoch = 0
    else:
        # Resume from checkpoint
        last_epoch = max(existing_epochs)
        epoch = last_epoch + 1
        print("Loading model from '{}/epoch{}.pt'...".format(filename, last_epoch))
        stdout.flush()
        loaded_obj = torch.load(filename + '/epoch{}.pt'.format(last_epoch), map_location=device, weights_only=False)
        # Support both old (tuple) and new (dict) checkpoint formats
        _restored_optimizer_state = None
        _restored_accuracy_window = None
        _restored_lr_scheduler_state = None
        if isinstance(loaded_obj, dict):
            model = loaded_obj['model']
            setstate(loaded_obj['python_rng'])
            np.random.set_state(loaded_obj['numpy_rng'])
            torch.set_rng_state(loaded_obj['torch_rng'].cpu())
            _restored_optimizer_state = loaded_obj.get('optimizer')
            _restored_accuracy_window = loaded_obj.get('accuracy_window')
            _restored_lr_scheduler_state = loaded_obj.get('lr_scheduler')
        elif isinstance(loaded_obj, tuple) and len(loaded_obj) == 4:
            model, random_state, np_random_state, torch_random_state = loaded_obj
            setstate(random_state)
            np.random.set_state(np_random_state)
            torch.set_rng_state(torch_random_state.cpu())
        else:
            raise ValueError(f"Unknown checkpoint format: {type(loaded_obj)}")

    if loss == "bce":
        loss_func = BCEWithLogitsLoss(reduction='mean')
    else:
        loss_func = CrossEntropyLoss(ignore_index=PADDING_TOKEN, reduction='mean')

    INITIAL_LR = 1.0e-4
    TARGET_LR = learning_rate

    if optimizer_type == 'adamw':
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=0.1)
        print(f"Using AdamW optimizer (lr={learning_rate}, wd=0.1)")
    elif optimizer_type == 'sophiag_hessian':
        optimizer = SophiaG((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=0.1)
        print(f"Using SophiaG optimizer WITH hessian updates every {hessian_update_interval} steps (lr={learning_rate}, wd=0.1)")
    else:
        optimizer = SophiaG((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=0.1)
        print(f"Using SophiaG optimizer (no hessian updates) (lr={learning_rate}, wd=0.1)")

    # Restore optimizer state from checkpoint (must happen after optimizer creation)
    if _restored_optimizer_state is not None:
        optimizer.load_state_dict(_restored_optimizer_state)
        print("Restored optimizer state from checkpoint")

    # LR scheduler
    lr_scheduler = None
    lr_eta_min = learning_rate * lr_min_ratio

    def _create_cosine_scheduler(warmup=lr_warmup_epochs, t_max=lr_cosine_epochs):
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        # Reset optimizer LR to peak
        for pg in optimizer.param_groups:
            pg['lr'] = learning_rate
        if warmup > 0:
            warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
            cosine_sched = CosineAnnealingLR(optimizer, T_max=t_max - warmup, eta_min=lr_eta_min)
            return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr_eta_min)

    if lr_schedule in ('cosine', 'cosine_per_stage'):
        lr_scheduler = _create_cosine_scheduler()
        if _restored_lr_scheduler_state is not None:
            lr_scheduler.load_state_dict(_restored_lr_scheduler_state)
            print("Restored LR scheduler state from checkpoint")
        mode_str = "per-stage" if lr_schedule == 'cosine_per_stage' else "global"
        print(f"Using cosine LR schedule ({mode_str}, peak={learning_rate}, min={lr_eta_min}, T_max={lr_cosine_epochs}, warmup={lr_warmup_epochs})")
    else:
        print(f"Using constant LR ({learning_rate})")

    # Mixed precision setup
    from contextlib import nullcontext
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    if use_amp:
        print("Using mixed precision (BF16)")

    # torch.compile (skip if already compiled from checkpoint)
    if use_compile and not isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        model = torch.compile(model, mode="reduce-overhead")
        print("Using torch.compile (reduce-overhead)")
    elif use_compile:
        print("Model already compiled (from checkpoint), skipping torch.compile")

    log_interval = 1
    eval_interval = 1   # Check accuracy every epoch (cheap: uses rolling window)
    save_interval = 50

    # Initialize curriculum parameters
    if curriculum_mode == 'n':
        curriculum_alpha = 1.0
        current_stage = 0
    elif curriculum_mode == 'y':
        current_stage = 0
        target_L = base_lookahead
        curriculum_alpha = alpha_for_lookahead(target_L, max_input_size)
        print("Starting at Stage 1: L={}, alpha={:.4f}".format(target_L, curriculum_alpha))

    if hasattr(model, 'alpha'):
        curriculum_alpha = model.alpha
    else:
        model.alpha = curriculum_alpha

    if hasattr(model, 'curriculum_stage'):
        current_stage = model.curriculum_stage
        if curriculum_mode == 'y':
            target_L = min(base_lookahead + current_stage * lookahead_step, max_lookahead)
            curriculum_alpha = alpha_for_lookahead(target_L, max_input_size)
            print("Resumed at Stage {}: L={}, alpha={:.4f}".format(current_stage + 1, target_L, curriculum_alpha))
            model.alpha = curriculum_alpha
    else:
        model.curriculum_stage = current_stage

    if hasattr(model, 'max_lookahead'):
        max_lookahead = model.max_lookahead
    else:
        model.max_lookahead = max_lookahead

    if hasattr(model, 'max_edges'):
        max_edges = model.max_edges
    else:
        model.max_edges = max_edges

    # ========== History tracking for plots ==========
    loss_history = []
    stage_eval_history = []
    mark_next_as_resume = False
    plot_metadata = {
        "model": f"Transformer {nlayers}L/{nhead}H/{max(ntoken, d_hid)}D",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_lookahead": max_lookahead,
        "max_input_size": max_input_size,
        "base_lookahead": base_lookahead,
        "lookahead_step": lookahead_step,
        "accuracy_threshold": accuracy_threshold,
    }
    # Load existing histories on resume, truncating to checkpoint epoch
    history_path = os.path.join(filename, "loss_history.json")
    if isfile(history_path) and len(existing_epochs) > 0:
        with open(history_path) as f:
            loss_history = json.load(f)
        # Truncate to checkpoint epoch (discard entries beyond checkpoint)
        resume_epoch = epoch - 1  # epoch was set to last_epoch + 1
        before = len(loss_history)
        loss_history = [h for h in loss_history if h["step"] <= resume_epoch]
        if before != len(loss_history):
            print(f"Truncated loss history from {before} to {len(loss_history)} (to match checkpoint epoch {resume_epoch})")
        else:
            print(f"Loaded {len(loss_history)} loss history records")
        # Mark the next new entry as a resume point
        mark_next_as_resume = True
    eval_history_path = os.path.join(filename, "stage_eval_history.json")
    if isfile(eval_history_path) and len(existing_epochs) > 0:
        with open(eval_history_path) as f:
            stage_eval_history = json.load(f)
        # Truncate stage eval history too
        resume_epoch = epoch - 1
        stage_eval_history = [h for h in stage_eval_history if h["step"] <= resume_epoch]
        print(f"Loaded {len(stage_eval_history)} stage eval records")

    PLOT_EVERY = 50  # Generate plots every N epochs
    training_start_time = time.perf_counter()

    if dataset_size == -1:
        # Streaming training setup
        from itertools import cycle
        from threading import Lock
        STREAMING_BLOCK_SIZE = update_rate
        NUM_DATA_WORKERS = 2
        seed_generator = Random(seed_value)
        seed_generator_lock = Lock()
        seed_values = []

        def get_seed(index):
            if index < len(seed_values):
                return seed_values[index]
            seed_generator_lock.acquire()
            while index >= len(seed_values):
                seed_values.append(seed_generator.randrange(2 ** 32))
            seed_generator_lock.release()
            return seed_values[index]

        class StreamingDatasetSearch(torch.utils.data.IterableDataset):
            def __init__(self, offset, alpha, max_lookahead, max_edges, distance_from_start, max_prefix_vertices):
                super().__init__()
                self.offset = offset
                self.alpha = alpha
                self.max_lookahead = max_lookahead
                self.max_edges = max_edges
                self.distance_from_start = distance_from_start
                self.max_prefix_vertices = max_prefix_vertices
                self.multiprocessing_manager = multiprocessing.Manager()
                self.total_collisions = self.multiprocessing_manager.Value(int, 0)
                self.collisions_lock = self.multiprocessing_manager.Lock()

            def process_data(self, start):
                current = start
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id
                max_prefix_verts = (0 if distribution == 'crafted_no_prefix' else
                                   (max_input_size if self.max_prefix_vertices == -1 else self.max_prefix_vertices))
                while True:
                    worker_start_time = time.perf_counter()
                    new_seed = get_seed(current)
                    generator.set_seed(new_seed)
                    seed(new_seed)
                    torch.manual_seed(new_seed)
                    np.random.seed(new_seed)

                    generate_start_time = time.perf_counter()
                    # Generate search training data with curriculum learning support
                    batch = generator.generate_training_set(
                        max_input_size, BATCH_SIZE, self.max_lookahead, self.max_edges,
                        reserved_inputs, self.distance_from_start, max_prefix_verts,
                        True, self.alpha)

                    if batch[3] != 0:  # num_collisions
                        with self.collisions_lock:
                            self.total_collisions.value += batch[3]
                        stdout.flush()

                    worker_end_time = time.perf_counter()
                    yield batch[:-1]  # Return inputs, outputs, labels (exclude num_collisions)
                    current += NUM_DATA_WORKERS

            def __iter__(self):
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id
                return self.process_data(self.offset + worker_id)

        dataset = StreamingDatasetSearch(
            epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model.alpha,
            model.max_lookahead, model.max_edges, distance_from_start,
            max_prefix_vertices if max_prefix_vertices != None else -1)
        loader = DataLoader(dataset, batch_size=None, num_workers=NUM_DATA_WORKERS,
                           pin_memory=True, prefetch_factor=8)

    examples_seen = epoch * STREAMING_BLOCK_SIZE
    LR_DECAY_TIME = 2**24  # examples seen
    accuracy_window_buf = deque(maxlen=accuracy_window_size)
    if _restored_accuracy_window is not None:
        accuracy_window_buf.extend(_restored_accuracy_window)
        print(f"Restored accuracy window ({len(accuracy_window_buf)} entries)")

    while True:
        start_time = time.perf_counter()
        transfer_time = 0.0
        train_time = 0.0
        log_time = 0.0
        epoch_loss = 0.0
        num_batches = 0
        effective_dataset_size = (STREAMING_BLOCK_SIZE if dataset_size == -1 else dataset_size)
        reinit_data_loader = False

        for batch in loader:
            batch_start_time = time.perf_counter()

            # Learning rate scheduling
            if lr_scheduler is not None:
                # LR managed by PyTorch scheduler (cosine, cosine_per_stage)
                lr = optimizer.param_groups[0]['lr']
            elif warm_up != 0:
                if examples_seen < warm_up:
                    lr = examples_seen * INITIAL_LR / warm_up
                elif examples_seen < warm_up + LR_DECAY_TIME:
                    lr = (0.5 * np.cos(np.pi * (examples_seen - warm_up) / LR_DECAY_TIME) + 0.5) * (
                                INITIAL_LR - TARGET_LR) + TARGET_LR
                else:
                    lr = TARGET_LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = TARGET_LR

            model.train()

            inputs, outputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            outputs = outputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            examples_seen += BATCH_SIZE

            train_start_time = time.perf_counter()
            transfer_time += train_start_time - batch_start_time

            # Forward pass
            with amp_ctx:
                logits = model(inputs)

                if loss == "bce":
                    loss_val = loss_func(logits[:, -1, :], outputs)
                else:
                    loss_val = loss_func(logits[:, -1, :], labels)

            # Compute pre-update accuracy on this batch (before gradient step)
            with torch.no_grad():
                if loss == "bce":
                    batch_acc = torch.sum(torch.gather(outputs, 1,
                        torch.argmax(logits[:,-1,:],dim=1).unsqueeze(1))).item() / outputs.size(0)
                else:
                    batch_acc = torch.sum(torch.gather(outputs, 1,
                        torch.argmax(logits[:,-1,:],dim=1).unsqueeze(1))).item() / outputs.size(0)
                accuracy_window_buf.append(batch_acc)

            epoch_loss += loss_val.item()
            loss_val.backward()

            if examples_seen % (BATCH_SIZE * grad_accumulation_steps) == 0:
                if hessian_update_interval > 0 and optimizer_type == 'sophiag_hessian':
                    step_count = examples_seen // (BATCH_SIZE * grad_accumulation_steps)
                    if step_count % hessian_update_interval == 0:
                        optimizer.update_hessian()
                optimizer.step()
                optimizer.zero_grad()

            log_start_time = time.perf_counter()
            train_time += log_start_time - train_start_time
            num_batches += 1

            if num_batches == effective_dataset_size // BATCH_SIZE:
                if epoch % log_interval == 0:
                    elapsed_time = time.perf_counter() - start_time
                    avg_loss = epoch_loss / num_batches
                    print("epoch = {}, training loss = {:.6f}".format(epoch, avg_loss))
                    print("throughput = {} examples/s".format(effective_dataset_size / elapsed_time))
                    print('Total number of training examples in test set: {}'.format(dataset.total_collisions.value))
                    print('Learning rate: {}'.format(lr))
                    if curriculum_mode == 'y':
                        target_L = min(base_lookahead + current_stage * lookahead_step, max_lookahead)
                        print('Stage {}: L={}, alpha={:.4f}'.format(
                            current_stage + 1, target_L, model.alpha))
                    print("[PROFILE] Total batch time: {}s".format(elapsed_time))
                    print("[PROFILE] Time to transfer data to GPU: {}s".format(transfer_time))
                    print("[PROFILE] Time to train: {}s".format(train_time))
                    print("[PROFILE] Time to log/save/validate: {}s".format(log_time))
                    stdout.flush()
                    start_time = time.perf_counter()
                    transfer_time = 0.0
                    train_time = 0.0
                    log_time = 0.0

                if epoch % eval_interval == 0:
                    # Training accuracy (rolling window of pre-update batch accuracies)
                    training_acc = sum(accuracy_window_buf) / len(accuracy_window_buf) if accuracy_window_buf else 0.0

                    print("training accuracy: %.2f±%.2f (w=%d)" % (training_acc,
                          binomial_confidence_int(training_acc, len(accuracy_window_buf) * BATCH_SIZE),
                          len(accuracy_window_buf)))
                    del inputs, outputs, labels
                    stdout.flush()

                    # Record history
                    target_L = min(base_lookahead + current_stage * lookahead_step, max_lookahead) if curriculum_mode == 'y' else max_lookahead
                    wall_time = time.perf_counter() - training_start_time
                    entry = {
                        "step": epoch,
                        "loss": epoch_loss / num_batches,
                        "stage": current_stage + 1,
                        "alpha": model.alpha,
                        "effective_L": target_L,
                        "wall_time": wall_time,
                        "training_acc": training_acc,
                    }

                    if mark_next_as_resume:
                        entry["resume"] = True
                        mark_next_as_resume = False

                    # Periodic test eval
                    if test_eval_every > 0 and epoch % test_eval_every == 0 and eval_inputs is not None:
                        model.eval()
                        test_acc_periodic, test_loss_periodic, _ = evaluate_model(model, eval_inputs, eval_outputs)
                        entry["test_acc"] = test_acc_periodic
                        entry["test_loss"] = test_loss_periodic
                        model.train()

                    loss_history.append(entry)

                    # Curriculum learning update (linear L increase)
                    if curriculum_mode == 'y' and training_acc > accuracy_threshold:
                        target_L = min(base_lookahead + current_stage * lookahead_step, max_lookahead)

                        model.eval()

                        # Test accuracy (alpha=1.0, full difficulty)
                        test_acc, test_loss_val, _ = evaluate_model(model, eval_inputs, eval_outputs)
                        print("Epoch {}: Test Acc = {:.2f}±{:.2f}, Loss = {:.6f}".format(
                            epoch, test_acc, binomial_confidence_int(test_acc, eval_inputs.shape[0]), test_loss_val))
                        stdout.flush()

                        # Fresh-data eval at current alpha before promotion
                        rng_py_backup = getstate()
                        rng_np_backup = np.random.get_state()
                        rng_torch_backup = torch.get_rng_state()
                        generator.set_seed(current_stage * 55555 + 99)
                        seed(current_stage * 55555 + 99)
                        max_prefix_verts = (0 if distribution == 'crafted_no_prefix' else
                                           (max_input_size if max_prefix_vertices is None else max_prefix_vertices))
                        promo_eval_in, promo_eval_out, _, _ = generator.generate_training_set(
                            max_input_size, min(BATCH_SIZE, 512), max_lookahead, model.max_edges,
                            reserved_inputs, distance_from_start, max_prefix_verts,
                            True, model.alpha)
                        promo_acc, promo_loss, _ = evaluate_model(model, promo_eval_in, promo_eval_out)
                        print("Stage {} promotion eval (fresh alpha={:.4f}): Acc = {:.2f}, Loss = {:.6f}".format(
                            current_stage + 1, model.alpha, promo_acc, promo_loss))
                        del promo_eval_in, promo_eval_out
                        setstate(rng_py_backup)
                        np.random.set_state(rng_np_backup)
                        torch.set_rng_state(rng_torch_backup)
                        stdout.flush()

                        # Record stage eval at promotion
                        stage_eval_history.append({
                            "step": epoch,
                            "stage": current_stage + 1,
                            "effective_L": target_L,
                            "training_acc": training_acc,
                            "test_acc": test_acc,
                            "test_loss": test_loss_val,
                            "stage_test_acc": promo_acc,
                            "stage_test_loss": promo_loss,
                        })

                        if target_L < max_lookahead:
                            old_stage = current_stage
                            current_stage += 1
                            new_L = min(base_lookahead + current_stage * lookahead_step, max_lookahead)
                            new_alpha = alpha_for_lookahead(new_L, max_input_size)
                            model.alpha = new_alpha
                            model.curriculum_stage = current_stage
                            print("Curriculum update: Stage {} -> Stage {}, L={} -> L={}, alpha={:.4f}".format(
                                old_stage + 1, current_stage + 1, target_L, new_L, new_alpha))

                            # Clear accuracy window for new stage
                            accuracy_window_buf.clear()

                            # Reset cosine schedule for new stage
                            if lr_schedule == 'cosine_per_stage':
                                lr_scheduler = _create_cosine_scheduler()
                                print(f"  LR schedule reset for new stage (peak={learning_rate})")

                            # Save histories and generate plots at stage transition
                            save_histories(filename, loss_history, stage_eval_history)
                            generate_all_plots(filename, loss_history, n_stages, plot_metadata,
                                               stage_eval_history)

                            reinit_data_loader = True
                            break
                        else:
                            print("Curriculum complete! Final stage reached (L={})".format(target_L))
                            # Final plots
                            save_histories(filename, loss_history, stage_eval_history)
                            generate_all_plots(filename, loss_history, n_stages, plot_metadata,
                                               stage_eval_history)

                    # Periodic plot generation
                    elif epoch > 0 and epoch % PLOT_EVERY == 0:
                        save_histories(filename, loss_history, stage_eval_history)
                        generate_all_plots(filename, loss_history, n_stages, plot_metadata,
                                           stage_eval_history)

                if epoch % save_interval == 0:
                    ckpt_filename = filename + '/epoch{}.pt'.format(epoch)
                    print('Saving model to "{}".'.format(ckpt_filename))
                    ckpt_data = {
                        'model': model,
                        'optimizer': optimizer.state_dict(),
                        'python_rng': getstate(),
                        'numpy_rng': np.random.get_state(),
                        'torch_rng': torch.get_rng_state(),
                        'accuracy_window': list(accuracy_window_buf),
                    }
                    if lr_scheduler is not None:
                        ckpt_data['lr_scheduler'] = lr_scheduler.state_dict()
                    torch.save(ckpt_data, ckpt_filename)
                    # Also save histories at checkpoint time
                    save_histories(filename, loss_history, stage_eval_history)
                    print('Done saving model.')
                    stdout.flush()

                epoch += 1
                if lr_scheduler is not None:
                    lr_scheduler.step()
                num_batches = 0
                epoch_loss = 0.0
                if reinit_data_loader:
                    break

            log_end_time = time.perf_counter()
            log_time += log_end_time - log_start_time

        if reinit_data_loader:
            dataset = StreamingDatasetSearch(
                epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model.alpha,
                model.max_lookahead, model.max_edges, distance_from_start,
                max_prefix_vertices if max_prefix_vertices != None else -1)
            loader = DataLoader(dataset, batch_size=None, num_workers=NUM_DATA_WORKERS,
                              pin_memory=True, prefetch_factor=8)
            reinit_data_loader = False


if __name__ == "__main__":
    import argparse
    def parse_bool_arg(v):
        if isinstance(v, bool):
            return v
        elif v.lower() in ('yes', 'true', 'y', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'n', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-input-size", type=int)
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--max-lookahead", type=int, required=False)
    parser.add_argument("--max-edges", type=int, required=False)
    parser.add_argument("--distance-from-start", type=int, default=-1, required=False)
    parser.add_argument("--max-prefix-vertices", type=int, required=False)
    parser.add_argument("--nlayers", type=int)
    parser.add_argument("--nhead", type=int, default=1, required=False)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bidirectional", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--pos-emb", type=str, required=True, choices=["absolute", "rotary", "none"])
    parser.add_argument("--learn-tok-emb", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--toeplitz-attn", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--toeplitz-reg", type=float, required=True, default=0.0)
    parser.add_argument("--toeplitz-pos-only", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--add-padding", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--ablate", type=str, default="none", choices=["none", "attn_linear", "attn_linear_projv"])
    parser.add_argument("--preLN", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--curriculum", type=str, required=True, choices=["y", "n"])
    parser.add_argument("--looped", type=parse_bool_arg, default=False)
    parser.add_argument("--task", type=str, default="search", choices=["search"])
    parser.add_argument("--distribution", type=str, default="crafted", choices=["simple", "crafted", "crafted_no_prefix", "star"])
    parser.add_argument("--warm-up", type=int, default=0, required=False)
    parser.add_argument("--batch-size", type=int, default=2**8, required=False)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5, required=False)
    parser.add_argument("--update-rate", type=int, default=2**18, required=False)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1, required=False)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'bce'], required=False)
    # New curriculum args
    parser.add_argument("--base-lookahead", type=int, default=2,
                        help="Starting lookahead L at stage 1 (linear L curriculum)")
    parser.add_argument("--lookahead-step", type=int, default=2,
                        help="Lookahead increase per stage (linear L curriculum)")
    parser.add_argument("--accuracy-threshold", type=float, default=0.98,
                        help="Training accuracy threshold to advance curriculum stage")
    parser.add_argument("--accuracy-window", type=int, default=100,
                        help="Rolling window size (in batches) for pre-update accuracy measurement")
    parser.add_argument("--test-eval-every", type=int, default=50,
                        help="Run test eval every N epochs (0 to disable)")
    parser.add_argument("--output-dir", type=str, default="search_results",
                        help="Directory for checkpoints and results")
    parser.add_argument("--optimizer", type=str, default="sophiag",
                        choices=["sophiag", "adamw", "sophiag_hessian"],
                        help="Optimizer: sophiag (default, no hessian), adamw, sophiag_hessian")
    parser.add_argument("--use-amp", action="store_true",
                        help="Enable mixed precision training (BF16)")
    parser.add_argument("--use-compile", action="store_true",
                        help="Enable torch.compile for the model")
    parser.add_argument("--hessian-update-interval", type=int, default=10,
                        help="Steps between hessian updates (only for sophiag_hessian)")
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["constant", "cosine", "cosine_per_stage"],
                        help="LR schedule: constant, cosine (global decay), cosine_per_stage (reset on stage advance)")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0,
                        help="Number of warmup epochs for LR schedule")
    parser.add_argument("--lr-min-ratio", type=float, default=0.01,
                        help="Minimum LR as ratio of peak LR for cosine decay")
    parser.add_argument("--lr-cosine-epochs", type=int, default=20000,
                        help="Total epochs for one cosine decay cycle (T_max). For cosine_per_stage, this is the T_max per stage")
    args = parser.parse_args()

    train(
        max_input_size=args.max_input_size,
        dataset_size=args.dataset_size,
        distribution=args.distribution,
        max_lookahead=args.max_lookahead,
        seed_value=args.seed,
        nhead=args.nhead,
        nlayers=args.nlayers,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        pos_emb=args.pos_emb,
        learnable_token_emb=args.learn_tok_emb,
        toeplitz_attn=args.toeplitz_attn,
        toeplitz_reg=args.toeplitz_reg,
        toeplitz_pos_only=args.toeplitz_pos_only,
        add_padding=args.add_padding,
        ablate=args.ablate,
        pre_ln=args.preLN,
        curriculum_mode=args.curriculum,
        looped=args.looped,
        task=args.task,
        warm_up=args.warm_up,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        update_rate=args.update_rate,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_edges=args.max_edges,
        distance_from_start=args.distance_from_start,
        max_prefix_vertices=args.max_prefix_vertices,
        loss=args.loss,
        base_lookahead=args.base_lookahead,
        lookahead_step=args.lookahead_step,
        accuracy_threshold=args.accuracy_threshold,
        accuracy_window_size=args.accuracy_window,
        test_eval_every=args.test_eval_every,
        lr_schedule=args.lr_schedule,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_min_ratio=args.lr_min_ratio,
        lr_cosine_epochs=args.lr_cosine_epochs,
        output_dir=args.output_dir,
        optimizer_type=args.optimizer,
        use_amp=args.use_amp,
        use_compile=args.use_compile,
        hessian_update_interval=args.hessian_update_interval)
