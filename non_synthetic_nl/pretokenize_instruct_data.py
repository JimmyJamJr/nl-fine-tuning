#!/usr/bin/env python3
# pretokenize_curriculum.py

import os
import sys
import json
import pickle
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
import time

def pretokenize_curriculum_data():
    tokenizer_name = "Qwen/Qwen3-4B-Instruct-2507"
    cache_dir = "/scratch/gautschi/huan2073/model_cache"
    output_dir = "/scratch/gautschi/huan2073/pretokenized_curriculum"
    cutoffs_dir = "./curriculum_cutoffs"
    max_samples = 2000000
    max_length = 2048
    batch_size = 100
    
    print(f"Starting pretokenization script...", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Output will be saved to: {output_dir}", flush=True)
    print(f"Target: {max_samples:,} samples at {max_length} tokens", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    
    start_time = time.time()
    
    # Check if directories exist
    if not os.path.exists(cutoffs_dir):
        print(f"ERROR: Cutoffs directory not found: {cutoffs_dir}", flush=True)
        return
    
    print(f"Loading tokenizer from {tokenizer_name}...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            cache_dir=cache_dir, 
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer loaded successfully", flush=True)
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}", flush=True)
        return
    
    # Load cutoffs
    print(f"Loading cutoffs from {cutoffs_dir}...", flush=True)
    try:
        with open(os.path.join(cutoffs_dir, "cutoffs.json"), 'r') as f:
            cutoffs = json.load(f)
        
        # Check if this is percentile-based method
        if cutoffs.get('method') != 'percentile':
            print(f"WARNING: Cutoffs were not calculated using percentile method!", flush=True)
            print(f"         Rerun curriculum_cutoffs.py with the percentile-based version", flush=True)
        
        print(f"Loaded cutoffs with {cutoffs['n_stages']} stages", flush=True)
        print(f"Method: {cutoffs.get('method', 'unknown')}", flush=True)
        
        with open(os.path.join(cutoffs_dir, "token_statistics.pkl"), 'rb') as f:
            token_stats = pickle.load(f)
            token_to_rank = token_stats['token_to_rank']
        print(f"Loaded token statistics with {len(token_to_rank)} unique tokens", flush=True)
    except Exception as e:
        print(f"ERROR loading cutoffs: {e}", flush=True)
        return
    
    def get_percentile_rank(value, percentile_values):
        """Get percentile rank (0-1) for a value using pre-computed percentiles."""
        # Use searchsorted to find where value would be inserted
        idx = np.searchsorted(percentile_values, value)
        # Convert to 0-1 range (1000 percentile points)
        return min(idx / 1000.0, 1.0)
    
    def assign_stage(text):
        """Assign stage using percentile-based cutoffs."""
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
        
        if not tokens or len(tokens) > max_length * 2:
            return -1
        
        length = len(tokens)
        token_ranks = [token_to_rank.get(t, len(token_to_rank) + 1) for t in tokens]
        max_token_rank = max(token_ranks) if token_ranks else 1
        
        # Use percentile-based normalization if available
        if 'length_percentiles' in cutoffs and 'vocab_percentiles' in cutoffs:
            # Convert lists back to numpy arrays for efficiency
            length_percentiles = np.array(cutoffs['length_percentiles'])
            vocab_percentiles = np.array(cutoffs['vocab_percentiles'])
            
            # Get percentile ranks
            length_percentile = get_percentile_rank(length, length_percentiles)
            vocab_percentile = get_percentile_rank(max_token_rank, vocab_percentiles)
        else:
            # Fallback to old min-max normalization (not recommended)
            print(f"WARNING: Using old normalization method. Percentiles not found!", flush=True)
            stats = cutoffs['statistics']
            length_percentile = (length - stats['length_min']) / (stats['length_max'] - stats['length_min'] + 1e-10)
            vocab_percentile = (max_token_rank - stats['vocab_rank_min']) / (stats['vocab_rank_max'] - stats['vocab_rank_min'] + 1e-10)
            length_percentile = max(0, min(1, length_percentile))
            vocab_percentile = max(0, min(1, vocab_percentile))
        
        # Calculate alpha using percentile ranks
        weights = cutoffs['alpha_weights']
        alpha = weights[0] * length_percentile + weights[1] * vocab_percentile
        
        # Assign stage using alpha cutoffs
        stage = 1
        for i, cutoff in enumerate(cutoffs['alpha_cutoffs'][1:], 1):
            if alpha <= cutoff:
                stage = i
                break
        else:
            stage = cutoffs['n_stages']
        
        return stage
    
    # Create output directory
    print(f"Creating output directory: {output_dir}", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize storage
    stage_data = {i: [] for i in range(1, cutoffs['n_stages'] + 1)}
    stage_counts = Counter()
    
    print(f"Loading dataset from HuggingFace (this may take a minute)...", flush=True)
    try:
        dataset = load_dataset(
            "nvidia/OpenMathInstruct-2",
            split="train",
            streaming=True,
            cache_dir=os.path.join(cache_dir, 'datasets')
        )
        print(f"Dataset loaded, starting processing...", flush=True)
    except Exception as e:
        print(f"ERROR loading dataset: {e}", flush=True)
        return
    
    # Process dataset
    batch_texts = []
    batch_stages = []
    processed = 0
    skipped = 0
    last_report = time.time()
    
    print(f"Beginning tokenization loop...", flush=True)
    print(f"Using percentile-based stage assignment for robust distribution", flush=True)
    print(f"Using int32 for input_ids to support large vocabulary", flush=True)
    
    for item_idx, item in enumerate(dataset):
        if processed >= max_samples:
            break
        
        # Show progress immediately for first few items
        if item_idx < 5:
            print(f"  Processing item {item_idx}...", flush=True)
        
        text = f"Question: {item['problem']}\n\nAnswer: {item['generated_solution']}"
        
        stage = assign_stage(text)
        if stage < 1:
            skipped += 1
            continue
        
        batch_texts.append(text)
        batch_stages.append(stage)
        
        # Process batch
        if len(batch_texts) >= batch_size:
            if processed < 100:
                print(f"  Tokenizing batch of {len(batch_texts)} samples...", flush=True)
            
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='np'
            )
            
            for i, stage_idx in enumerate(batch_stages):
                data = {
                    'input_ids': encodings['input_ids'][i].astype(np.int32),  # FIXED: Changed from int16 to int32
                    'attention_mask': encodings['attention_mask'][i].astype(np.int8),
                }
                stage_data[stage_idx].append(data)
                stage_counts[stage_idx] += 1
            
            processed += len(batch_texts)
            batch_texts = []
            batch_stages = []
            
            # Always report first batch
            if processed <= batch_size:
                print(f"First batch complete! Processed: {processed}", flush=True)
                print(f"  Initial stage distribution: {dict(sorted(stage_counts.items()))}", flush=True)
            
            # Regular progress updates
            if time.time() - last_report > 10:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (max_samples - processed) / rate if rate > 0 else 0
                print(f"Progress: {processed:,}/{max_samples:,} ({processed/max_samples*100:.1f}%) | "
                      f"Rate: {rate:.1f} samples/s | ETA: {eta/60:.1f} min | "
                      f"Skipped: {skipped}", flush=True)
                print(f"  Stage distribution: {dict(sorted(stage_counts.items()))}", flush=True)
                
                # Check if distribution looks reasonable
                if processed > 10000:
                    stages_with_data = sum(1 for count in stage_counts.values() if count > 0)
                    if stages_with_data < cutoffs['n_stages'] * 0.5:
                        print(f"  WARNING: Only {stages_with_data}/{cutoffs['n_stages']} stages have data!", flush=True)
                        print(f"           This is unusual for percentile-based assignment.", flush=True)
                
                last_report = time.time()
    
    # Process remaining batch
    if batch_texts:
        print(f"Processing final batch of {len(batch_texts)} samples...", flush=True)
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='np'
        )
        
        for i, stage_idx in enumerate(batch_stages):
            data = {
                'input_ids': encodings['input_ids'][i].astype(np.int32),  # Already correct here
                'attention_mask': encodings['attention_mask'][i].astype(np.int8),
            }
            stage_data[stage_idx].append(data)
            stage_counts[stage_idx] += 1
        
        processed += len(batch_texts)
    
    print(f"\nTokenization complete. Saving to disk...", flush=True)
    
    # Save files
    for stage, samples in stage_data.items():
        if samples:
            filename = os.path.join(output_dir, f"stage_{stage:02d}.pkl")
            print(f"  Saving stage {stage}: {len(samples)} samples...", flush=True)
            with open(filename, 'wb') as f:
                pickle.dump(samples, f)
    
    # Save metadata
    metadata = {
        'total_samples': processed,
        'skipped_samples': skipped,
        'stage_distribution': dict(stage_counts),
        'max_length': max_length,
        'tokenizer': tokenizer_name,
        'cutoffs_method': cutoffs.get('method', 'unknown'),
        'cutoffs_dir': cutoffs_dir,
        'data_types': {
            'input_ids': 'int32',
            'attention_mask': 'int8'
        }
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print final distribution analysis
    print(f"\n{'='*50}", flush=True)
    print(f"STAGE DISTRIBUTION ANALYSIS", flush=True)
    print(f"{'='*50}", flush=True)
    
    stages_with_data = sum(1 for count in stage_counts.values() if count > 0)
    print(f"Stages with data: {stages_with_data}/{cutoffs['n_stages']}", flush=True)
    
    if stages_with_data == cutoffs['n_stages']:
        print(f"✅ SUCCESS: All stages have samples!", flush=True)
    elif stages_with_data >= cutoffs['n_stages'] * 0.7:
        print(f"✓ GOOD: Most stages ({stages_with_data}) have samples", flush=True)
    else:
        print(f"⚠️  WARNING: Only {stages_with_data} stages have samples", flush=True)
    
    print(f"\nPer-stage distribution:", flush=True)
    for stage in range(1, cutoffs['n_stages'] + 1):
        count = stage_counts.get(stage, 0)
        percentage = count / processed * 100 if processed > 0 else 0
        bar = '█' * int(percentage / 2) + '░' * (50 - int(percentage / 2))
        print(f"  Stage {stage:2d}: {count:7,} ({percentage:5.1f}%) |{bar}|", flush=True)
    
    total_time = time.time() - start_time
    print(f"\n{'='*50}", flush=True)
    print(f"COMPLETE!", flush=True)
    print(f"Processed: {processed:,} samples", flush=True)
    print(f"Skipped: {skipped} samples", flush=True)
    print(f"Time: {total_time/60:.1f} minutes", flush=True)
    print(f"Rate: {processed/total_time:.1f} samples/second", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Data types: input_ids=int32, attention_mask=int8", flush=True)
    print(f"{'='*50}", flush=True)

if __name__ == "__main__":
    print("Script started", flush=True)
    pretokenize_curriculum_data()
    print("Script ended", flush=True)