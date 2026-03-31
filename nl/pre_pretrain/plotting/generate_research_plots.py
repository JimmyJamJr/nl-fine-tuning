#!/usr/bin/env python3
"""Generate all plots for research meeting.

Generates comparison plots for:
- Three methods at 5% and 25% mix (accuracy and loss)
- Mix percentage effect (from_pptrain and from_scratch)
- Training duration (2000 vs 3000 steps)
"""

from pathlib import Path

from plot_accuracy import load_csv, plot_accuracy
from plot_loss import plot_loss


# Base paths
SCRATCH = Path("/scratch/gautschi/mnickel")
OUTPUT_DIR = Path(__file__).parent.parent / "important_figs"


# Label mappings for clean display
LABELS_5PCT = {
    "pretrain_fresh": "Fresh (C4 only)",
    "pretrain_mix": "Fresh + 5% Synthetic",
    "pretrain_from_pptrain": "Pre-pretrained + 5% Synthetic",
}

LABELS_25PCT = {
    "pretrain_fresh_2000_steps": "Fresh (C4 only)",
    "pretrain_mix_25%": "Fresh + 25% Synthetic",
    "pretrain_from_pptrain_25%": "Pre-pretrained + 25% Synthetic",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load eval data
    eval_5pct = load_csv(SCRATCH / "eval_results_5%_2000" / "checkpoint_eval.csv")
    eval_25pct = load_csv(SCRATCH / "eval_results_25%_2000" / "checkpoint_eval.csv")

    # ===== THREE METHODS COMPARISON - ACCURACY =====

    # 5% mix accuracy
    plot_accuracy(
        results=eval_5pct,
        output_path=OUTPUT_DIR / "three_methods_5pct_accuracy.png",
        title="Three Methods Comparison (5% Synthetic Mix)",
        labels=LABELS_5PCT,
        log_scale=True,
    )

    # 25% mix accuracy
    plot_accuracy(
        results=eval_25pct,
        output_path=OUTPUT_DIR / "three_methods_25pct_accuracy.png",
        title="Three Methods Comparison (25% Synthetic Mix)",
        labels=LABELS_25PCT,
        log_scale=True,
    )

    # ===== THREE METHODS COMPARISON - LOSS =====

    # 5% mix loss
    plot_loss(
        run_dirs=[
            SCRATCH / "pretrain_fresh_2000",
            SCRATCH / "pretrain_mix_5%_2000",
            SCRATCH / "pretrain_from_pptrain_5%_2000",
        ],
        output_path=OUTPUT_DIR / "three_methods_5pct_loss.png",
        title="Training Loss (5% Synthetic Mix)",
        labels=["Fresh (C4 only)", "Fresh + 5% Synthetic", "Pre-pretrained + 5% Synthetic"],
    )

    # 25% mix loss
    plot_loss(
        run_dirs=[
            SCRATCH / "pretrain_fresh_2000",
            SCRATCH / "pretrain_mix_25%_2000",
            SCRATCH / "pretrain_from_pptrain_25%_2000",
        ],
        output_path=OUTPUT_DIR / "three_methods_25pct_loss.png",
        title="Training Loss (25% Synthetic Mix)",
        labels=["Fresh (C4 only)", "Fresh + 25% Synthetic", "Pre-pretrained + 25% Synthetic"],
    )

    # ===== MIX PERCENTAGE EFFECT - ACCURACY =====

    # Combine 5% and 25% data for from_pptrain comparison
    from_pptrain_combined = {
        "pretrain_from_pptrain": eval_5pct.get("pretrain_from_pptrain", []),
        "pretrain_from_pptrain_25%": eval_25pct.get("pretrain_from_pptrain_25%", []),
    }
    plot_accuracy(
        results=from_pptrain_combined,
        output_path=OUTPUT_DIR / "mix_effect_from_pptrain_accuracy.png",
        title="Mix Percentage Effect (Pre-pretrained)",
        labels={
            "pretrain_from_pptrain": "5% Synthetic",
            "pretrain_from_pptrain_25%": "25% Synthetic",
        },
        log_scale=True,
    )

    # From scratch comparison
    from_scratch_combined = {
        "pretrain_mix": eval_5pct.get("pretrain_mix", []),
        "pretrain_mix_25%": eval_25pct.get("pretrain_mix_25%", []),
    }
    plot_accuracy(
        results=from_scratch_combined,
        output_path=OUTPUT_DIR / "mix_effect_from_scratch_accuracy.png",
        title="Mix Percentage Effect (From Scratch)",
        labels={
            "pretrain_mix": "5% Synthetic",
            "pretrain_mix_25%": "25% Synthetic",
        },
        log_scale=True,
    )

    # ===== TRAINING DURATION - LOSS =====

    plot_loss(
        run_dirs=[
            SCRATCH / "pretrain_fresh_2000",
            SCRATCH / "pretrain_fresh_3000",
        ],
        output_path=OUTPUT_DIR / "fresh_2000_vs_3000_loss.png",
        title="Training Duration: 2000 vs 3000 Steps",
        labels=["Fresh 2000 steps", "Fresh 3000 steps"],
    )

    print(f"\nGenerated {len(list(OUTPUT_DIR.glob('*.png')))} plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
