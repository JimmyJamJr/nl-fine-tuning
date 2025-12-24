#!/usr/bin/env python3
"""
Curriculum Stage Cutoffs Calculator with Percentile-based Assignment
Uses percentile-based approach for robust stage assignment across different dataset samples
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime
from scipy import stats


class CurriculumCutoffsCalculator:
    def __init__(self,
                 tokenizer_name="Qwen/Qwen3-8B",
                 sample_size=100000,
                 n_stages=10,
                 alpha_weights=(0.5, 0.5),
                 cache_dir=None):
        """
        Initialize the cutoffs calculator.

        Args:
            tokenizer_name: HuggingFace tokenizer to use
            sample_size: Number of samples to analyze for determining cutoffs
            n_stages: Number of curriculum stages
            alpha_weights: (length_weight, vocab_weight) for alpha calculation
            cache_dir: Directory to cache downloaded models/tokenizers
        """
        self.tokenizer_name = tokenizer_name
        self.sample_size = sample_size
        self.n_stages = n_stages
        self.alpha_weights = alpha_weights
        self.cache_dir = cache_dir

        self.tokenizer = None
        self.global_token_freq = Counter()
        self.token_to_rank = {}
        self.cutoffs = None
        self.stage_stats = None
        
        # For percentile-based approach
        self.length_percentiles = None
        self.vocab_percentiles = None

    def load_tokenizer(self):
        """Load the tokenizer."""
        print(f"Loading tokenizer: {self.tokenizer_name}")
        if self.cache_dir:
            print(f"Using cache directory: {self.cache_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data_sample(self, dataset_configs: List[Dict]) -> List[Dict]:
        """
        Load a sample of data from specified datasets.

        Args:
            dataset_configs: List of dataset configurations
                             [{"name": "dataset_name", "split": "train", "format_fn": callable}]

        Returns:
            List of formatted samples
        """
        print(f"\nLoading sample of {self.sample_size} samples...")

        all_samples = []
        samples_per_dataset = self.sample_size // len(dataset_configs)

        for config in dataset_configs:
            dataset_name = config["name"]
            split = config.get("split", "train")
            format_fn = config.get("format_fn", lambda x: x)

            try:
                print(f"Loading {dataset_name}...")
                
                # Set cache directory for datasets if specified
                kwargs = {}
                if self.cache_dir:
                    kwargs['cache_dir'] = os.path.join(self.cache_dir, 'datasets')

                # Try loading with streaming first for large datasets
                try:
                    dataset = load_dataset(dataset_name, split=split, streaming=True, **kwargs)
                    samples = []
                    for i, item in enumerate(tqdm(dataset, desc=f"Sampling {dataset_name}")):
                        if i >= samples_per_dataset:
                            break
                        samples.append(format_fn(item))
                except:
                    # Fallback to regular loading
                    dataset = load_dataset(dataset_name, split=split, **kwargs)
                    indices = np.random.choice(
                        len(dataset),
                        min(samples_per_dataset, len(dataset)),
                        replace=False
                    )
                    samples = [format_fn(dataset[int(i)]) for i in indices]

                all_samples.extend(samples)
                print(f"  Loaded {len(samples)} samples from {dataset_name}")

            except Exception as e:
                print(f"  Error loading {dataset_name}: {e}")

        print(f"Total samples loaded: {len(all_samples)}")
        return all_samples

    def build_token_statistics(self, samples: List[Dict]):
        """Build token frequency statistics from the sample."""
        print("\nBuilding token frequency statistics...")

        for sample in tqdm(samples, desc="Analyzing tokens"):
            full_text = f"{sample.get('instruction', '')}\n{sample.get('output', '')}"
            tokens = self.tokenizer.encode(full_text, truncation=False)
            self.global_token_freq.update(tokens)

        # Create token to rank mapping
        sorted_tokens = sorted(
            self.global_token_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self.token_to_rank = {
            token: rank + 1
            for rank, (token, _) in enumerate(sorted_tokens)
        }

        print(f"Unique tokens: {len(self.token_to_rank)}")
        print(f"Total token occurrences: {sum(self.global_token_freq.values())}")

    def calculate_sample_metrics(self, sample: Dict) -> Dict:
        """Calculate complexity metrics for a single sample."""
        full_text = f"{sample.get('instruction', '')}\n{sample.get('output', '')}"
        tokens = self.tokenizer.encode(full_text, truncation=False)

        if not tokens:
            return None

        # Get token ranks
        token_ranks = [
            self.token_to_rank.get(token, len(self.token_to_rank) + 1)
            for token in tokens
        ]

        # Calculate metrics
        metrics = {
            "length": len(tokens),
            "max_token_rank": max(token_ranks),
            "mean_token_rank": np.mean(token_ranks),
            "median_token_rank": np.median(token_ranks),
            "vocab_diversity": len(set(tokens)) / len(tokens),
            "instruction": sample.get("instruction", "")[:200],
            "output": sample.get("output", "")[:200],
            "source": sample.get("source", "unknown")
        }

        return metrics

    def calculate_cutoffs(self, samples: List[Dict]) -> Dict:
        """
        Calculate stage cutoffs using percentile-based approach.
        This ensures robust distribution across stages regardless of outliers.

        Returns:
            Dictionary containing cutoffs and statistics
        """
        print("\nCalculating complexity metrics for samples...")

        # Calculate metrics for all samples
        metrics_list = []
        for sample in tqdm(samples, desc="Processing samples"):
            metrics = self.calculate_sample_metrics(sample)
            if metrics:
                metrics_list.append(metrics)

        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)

        # Calculate percentiles for robust normalization
        # Create 1000 percentile points for fine-grained mapping
        percentile_points = np.linspace(0, 100, 1001)
        
        # Calculate percentile values for length and vocab complexity
        self.length_percentiles = np.percentile(df['length'].values, percentile_points)
        self.vocab_percentiles = np.percentile(df['max_token_rank'].values, percentile_points)
        
        # Function to get percentile rank (0-1) for a value
        def get_percentile_rank(value, percentile_values):
            # Use searchsorted to find where value would be inserted
            idx = np.searchsorted(percentile_values, value)
            # Convert to 0-1 range
            return min(idx / 1000.0, 1.0)
        
        # Calculate percentile ranks for each sample
        df['length_percentile'] = df['length'].apply(
            lambda x: get_percentile_rank(x, self.length_percentiles)
        )
        df['vocab_percentile'] = df['max_token_rank'].apply(
            lambda x: get_percentile_rank(x, self.vocab_percentiles)
        )
        
        # Calculate alpha using percentile ranks (already in 0-1 range)
        df['alpha'] = (
            self.alpha_weights[0] * df['length_percentile'] +
            self.alpha_weights[1] * df['vocab_percentile']
        )

        # Sort by alpha
        df = df.sort_values('alpha').reset_index(drop=True)

        # Create stages by dividing sorted samples into equal percentile chunks
        # This ensures each stage gets approximately equal number of samples
        samples_per_stage = len(df) // self.n_stages

        # Calculate alpha cutoffs at stage boundaries
        alpha_cutoffs = [0.0]  # Start with 0

        for i in range(1, self.n_stages):
            # Find the alpha value at the boundary between stages
            boundary_idx = i * samples_per_stage
            if boundary_idx < len(df):
                # Use the average of values around the boundary for smoother transition
                if boundary_idx > 0:
                    cutoff = (df.iloc[boundary_idx - 1]['alpha'] + df.iloc[boundary_idx]['alpha']) / 2
                else:
                    cutoff = df.iloc[boundary_idx]['alpha']
                alpha_cutoffs.append(cutoff)
            else:
                alpha_cutoffs.append(1.0)

        alpha_cutoffs.append(1.0)  # End with 1.0

        # Calculate statistics for each stage
        stage_stats = []
        for i in range(self.n_stages):
            start_idx = i * samples_per_stage
            end_idx = (i + 1) * samples_per_stage if i < self.n_stages - 1 else len(df)

            stage_data = df.iloc[start_idx:end_idx]

            stats = {
                'stage': i + 1,
                'alpha_range': (stage_data['alpha'].min(), stage_data['alpha'].max()),
                'length_range': (
                    stage_data['length'].min(),
                    stage_data['length'].max()
                ),
                'vocab_range': (
                    stage_data['max_token_rank'].min(),
                    stage_data['max_token_rank'].max()
                ),
                'length_percentile_range': (
                    stage_data['length_percentile'].min(),
                    stage_data['length_percentile'].max()
                ),
                'vocab_percentile_range': (
                    stage_data['vocab_percentile'].min(),
                    stage_data['vocab_percentile'].max()
                ),
                'n_samples': len(stage_data),
                'avg_length': stage_data['length'].mean(),
                'avg_vocab_rank': stage_data['max_token_rank'].mean(),
                'avg_alpha': stage_data['alpha'].mean()
            }

            stage_stats.append(stats)

        # Store results
        self.cutoffs = {
            'alpha_cutoffs': alpha_cutoffs,
            'n_stages': self.n_stages,
            'alpha_weights': self.alpha_weights,
            'tokenizer': self.tokenizer_name,
            'sample_size': len(df),
            'timestamp': datetime.now().isoformat(),
            'cache_dir': self.cache_dir,
            'method': 'percentile',  # Mark that we're using percentile method
            'percentile_points': percentile_points.tolist(),
            'length_percentiles': self.length_percentiles.tolist(),
            'vocab_percentiles': self.vocab_percentiles.tolist(),
            'statistics': {
                'alpha_min': float(df['alpha'].min()),
                'alpha_max': float(df['alpha'].max()),
                'alpha_mean': float(df['alpha'].mean()),
                'alpha_std': float(df['alpha'].std()),
                'length_min': int(df['length'].min()),
                'length_max': int(df['length'].max()),
                'length_mean': float(df['length'].mean()),
                'length_p25': float(df['length'].quantile(0.25)),
                'length_p50': float(df['length'].quantile(0.50)),
                'length_p75': float(df['length'].quantile(0.75)),
                'vocab_rank_min': int(df['max_token_rank'].min()),
                'vocab_rank_max': int(df['max_token_rank'].max()),
                'vocab_rank_mean': float(df['max_token_rank'].mean()),
                'vocab_rank_p25': float(df['max_token_rank'].quantile(0.25)),
                'vocab_rank_p50': float(df['max_token_rank'].quantile(0.50)),
                'vocab_rank_p75': float(df['max_token_rank'].quantile(0.75))
            }
        }

        self.stage_stats = stage_stats

        print("\nStage Cutoffs Calculated (Percentile-based method):")
        print("-" * 60)
        for i, stats in enumerate(stage_stats):
            print(f"Stage {stats['stage']}: "
                  f"α ∈ [{stats['alpha_range'][0]:.4f}, {stats['alpha_range'][1]:.4f}], "
                  f"n={stats['n_samples']} ({stats['n_samples'] / len(df) * 100:.1f}%)")
            print(f"       Length %ile: [{stats['length_percentile_range'][0]:.3f}, "
                  f"{stats['length_percentile_range'][1]:.3f}], "
                  f"Vocab %ile: [{stats['vocab_percentile_range'][0]:.3f}, "
                  f"{stats['vocab_percentile_range'][1]:.3f}]")

        return self.cutoffs, df

    def save_cutoffs(self, output_dir: str = "./curriculum_cutoffs"):
        """Save cutoffs and related data to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save cutoffs as JSON
        cutoffs_file = os.path.join(output_dir, "cutoffs.json")
        with open(cutoffs_file, 'w') as f:
            json.dump(self.cutoffs, f, indent=2)
        print(f"\nCutoffs saved to {cutoffs_file}")

        # Save stage statistics
        stats_file = os.path.join(output_dir, "stage_statistics.json")
        
        # Convert stage stats to JSON-serializable format
        stage_stats_to_save = []
        for stats in self.stage_stats:
            clean_stats = {}
            for key, value in stats.items():
                if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                    clean_stats[key] = int(value)
                elif isinstance(value, (np.float64, np.float32, np.float16)):
                    clean_stats[key] = float(value)
                elif isinstance(value, np.ndarray):
                    clean_stats[key] = value.tolist()
                elif isinstance(value, tuple):
                    clean_stats[key] = tuple(
                        float(v) if isinstance(v, (np.float64, np.float32, np.float16))
                        else int(v) if isinstance(v, (np.int64, np.int32, np.int16, np.int8))
                        else v
                        for v in value
                    )
                else:
                    clean_stats[key] = value
            stage_stats_to_save.append(clean_stats)

        with open(stats_file, 'w') as f:
            json.dump(stage_stats_to_save, f, indent=2)
        print(f"Stage statistics saved to {stats_file}")

        # Save token statistics as pickle
        token_stats_file = os.path.join(output_dir, "token_statistics.pkl")
        with open(token_stats_file, 'wb') as f:
            pickle.dump({
                'global_token_freq': self.global_token_freq,
                'token_to_rank': self.token_to_rank,
                'tokenizer': self.tokenizer_name,
                'cache_dir': self.cache_dir
            }, f)
        print(f"Token statistics saved to {token_stats_file}")

    def plot_cutoffs_visualization(self, df: pd.DataFrame, output_dir: str = "./curriculum_cutoffs"):
        """Create visualization of the cutoffs and data distribution."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        plt.subplots_adjust(hspace=0.35, wspace=0.3)

        # Assign stages to data
        df['stage'] = 1
        for i in range(len(self.cutoffs['alpha_cutoffs']) - 1):
            lower = self.cutoffs['alpha_cutoffs'][i]
            upper = self.cutoffs['alpha_cutoffs'][i + 1]
            if i == 0:
                mask = (df['alpha'] >= lower) & (df['alpha'] <= upper)
            else:
                mask = (df['alpha'] > lower) & (df['alpha'] <= upper)
            df.loc[mask, 'stage'] = i + 1

        # 1. Curriculum Learning Stages (Top Left)
        ax = axes[0, 0]
        colors = plt.cm.coolwarm(np.linspace(0, 1, self.n_stages))
        for stage in range(1, self.n_stages + 1):
            stage_data = df[df['stage'] == stage]
            if len(stage_data) > 0:
                ax.scatter(
                    stage_data['length'],
                    stage_data['max_token_rank'],
                    label=f'Stage {stage}',
                    color=colors[stage - 1],
                    alpha=0.6,
                    s=10
                )
        ax.set_xlabel('Sequence Length (tokens)', fontsize=11)
        ax.set_ylabel('Max Token Rank', fontsize=11)
        ax.set_title('Curriculum Learning Stages', fontsize=12, pad=10)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # 2. Alpha Distribution with Stage Cutoffs (Top Middle)
        ax = axes[0, 1]
        ax.hist(df['alpha'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        for i, cutoff in enumerate(self.cutoffs['alpha_cutoffs'][1:-1]):
            ax.axvline(x=cutoff, color='red', linestyle='--', alpha=0.7,
                       label=f'Stage {i + 2}' if i == 0 else None)
        ax.set_xlabel('Alpha (Complexity Score)', fontsize=11)
        ax.set_ylabel('Sample Count', fontsize=11)
        ax.set_title('Alpha Distribution (Percentile-based)', fontsize=12, pad=10)
        if len(self.cutoffs['alpha_cutoffs']) > 2:
            ax.legend(['Stage boundaries'], loc='upper right')

        # 3. Percentile Distributions (Top Right)
        ax = axes[0, 2]
        ax.hist(df['length_percentile'], bins=50, alpha=0.5, color='green', label='Length')
        ax.hist(df['vocab_percentile'], bins=50, alpha=0.5, color='orange', label='Vocab')
        ax.set_xlabel('Percentile Rank', fontsize=11)
        ax.set_ylabel('Sample Count', fontsize=11)
        ax.set_title('Percentile Rank Distributions', fontsize=12, pad=10)
        ax.legend()

        # 4. 2D Histogram of Complexity Distribution (Bottom Left)
        ax = axes[1, 0]
        h = ax.hist2d(
            df['length'],
            df['max_token_rank'],
            bins=[30, 30],
            cmap='YlOrRd',
            cmin=1
        )
        ax.set_xlabel('Sequence Length (tokens)', fontsize=11)
        ax.set_ylabel('Max Token Rank', fontsize=11)
        ax.set_title('2D Histogram of Complexity Distribution', fontsize=12, pad=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        cbar = plt.colorbar(h[3], ax=ax, label='Sample Count')
        cbar.ax.tick_params(labelsize=9)

        # 5. Percentile Mapping Curves (Bottom Middle)
        ax = axes[1, 1]
        percentiles = np.linspace(0, 100, 101)
        length_vals = np.percentile(df['length'].values, percentiles)
        vocab_vals = np.percentile(df['max_token_rank'].values, percentiles)
        
        ax2 = ax.twinx()
        ax.plot(percentiles, length_vals, 'g-', label='Length', linewidth=2)
        ax2.plot(percentiles, vocab_vals, 'r-', label='Vocab Rank', linewidth=2)
        
        ax.set_xlabel('Percentile', fontsize=11)
        ax.set_ylabel('Sequence Length', color='g', fontsize=11)
        ax2.set_ylabel('Max Token Rank', color='r', fontsize=11)
        ax.set_title('Percentile Mapping Curves', fontsize=12, pad=10)
        ax.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # 6. Stage Sample Distribution (Bottom Right)
        ax = axes[1, 2]
        stage_counts = df['stage'].value_counts().sort_index()
        bars = ax.bar(stage_counts.index, stage_counts.values, 
                      color=colors[:len(stage_counts)])
        ax.set_xlabel('Stage', fontsize=11)
        ax.set_ylabel('Sample Count', fontsize=11)
        ax.set_title('Samples per Stage', fontsize=12, pad=10)
        ax.set_xticks(range(1, self.n_stages + 1))
        
        # Add percentage labels on bars
        for bar, count in zip(bars, stage_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count/len(df)*100:.1f}%',
                   ha='center', va='bottom')

        plt.suptitle(f'Percentile-based Curriculum Cutoffs (n={len(df):,} samples)', 
                    fontsize=15, y=0.98)

        # Save plot
        plot_file = os.path.join(output_dir, "cutoffs_visualization.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight', pad_inches=0.3)
        print(f"Visualization saved to {plot_file}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Calculate percentile-based curriculum cutoffs")
    parser.add_argument("--sample_size", type=int, default=200000,
                        help="Number of samples to analyze (recommend at least 200k)")
    parser.add_argument("--n_stages", type=int, default=10,
                        help="Number of curriculum stages")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Tokenizer to use")
    parser.add_argument("--output_dir", type=str, default="./curriculum_cutoffs",
                        help="Directory to save cutoffs")
    parser.add_argument("--cache_dir", type=str, default="/scratch/gautschi/huan2073/",
                        help="Directory to cache downloaded models/tokenizers")
    parser.add_argument("--length_weight", type=float, default=0.5,
                        help="Weight for length in alpha calculation")
    parser.add_argument("--vocab_weight", type=float, default=0.5,
                        help="Weight for vocabulary complexity in alpha calculation")

    args = parser.parse_args()

    # Create cache directory if specified
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"Using cache directory: {args.cache_dir}")

    # Initialize calculator
    calculator = CurriculumCutoffsCalculator(
        tokenizer_name=args.tokenizer,
        sample_size=args.sample_size,
        n_stages=args.n_stages,
        alpha_weights=(args.length_weight, args.vocab_weight),
        cache_dir=args.cache_dir
    )

    # Load tokenizer
    calculator.load_tokenizer()

    # Define dataset configurations
    dataset_configs = [
        {
            "name": "nvidia/OpenMathInstruct-2",
            "split": "train",
            "format_fn": lambda x: {
                "instruction": x.get("problem", ""),
                "output": x.get("generated_solution", ""),
                "source": "openmath"
            }
        }
    ]

    # Load data sample
    samples = calculator.load_data_sample(dataset_configs)

    if not samples:
        print("No data loaded. Exiting.")
        return

    # Build token statistics
    calculator.build_token_statistics(samples)

    # Calculate cutoffs
    cutoffs, df = calculator.calculate_cutoffs(samples)

    # Save cutoffs
    calculator.save_cutoffs(args.output_dir)

    # Create visualization
    calculator.plot_cutoffs_visualization(df, args.output_dir)

    print(f"\nPercentile-based cutoffs calculation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nKey advantages of percentile method:")
    print(f"  - Robust to outliers in the full dataset")
    print(f"  - Guarantees consistent distribution across stages")
    print(f"  - Works well even with smaller sample sizes")


if __name__ == "__main__":
    main()