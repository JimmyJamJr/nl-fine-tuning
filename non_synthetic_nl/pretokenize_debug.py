#!/usr/bin/env python3
"""
Debug script to check pre-tokenized data files
"""

import os
import json
import pickle
import sys

def check_pretokenized_data(pretokenized_dir="/scratch/gautschi/huan2073/pretokenized_curriculum"):
    """Check the status of pre-tokenized data files."""
    
    print("=" * 60)
    print("PRE-TOKENIZED DATA DIAGNOSTIC")
    print("=" * 60)
    print(f"Checking directory: {pretokenized_dir}")
    print()
    
    # Check if directory exists
    if not os.path.exists(pretokenized_dir):
        print(f"❌ ERROR: Directory does not exist: {pretokenized_dir}")
        return False
    
    print(f"✅ Directory exists")
    print()
    
    # List all files
    print("Files in directory:")
    print("-" * 40)
    files = sorted(os.listdir(pretokenized_dir))
    if not files:
        print("❌ ERROR: Directory is empty!")
        return False
    
    for f in files:
        filepath = os.path.join(pretokenized_dir, f)
        size = os.path.getsize(filepath)
        print(f"  {f}: {size/1024/1024:.2f} MB")
    print()
    
    # Check metadata
    metadata_file = os.path.join(pretokenized_dir, "metadata.json")
    if os.path.exists(metadata_file):
        print("✅ Metadata file found")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"  Total samples: {metadata.get('total_samples', 'N/A'):,}")
        print(f"  Skipped samples: {metadata.get('skipped_samples', 'N/A')}")
        print(f"  Max length: {metadata.get('max_length', 'N/A')}")
        print(f"  Tokenizer: {metadata.get('tokenizer', 'N/A')}")
        
        stage_dist = metadata.get('stage_distribution', {})
        if stage_dist:
            print("\n  Stage distribution:")
            for stage, count in sorted(stage_dist.items(), key=lambda x: int(x[0])):
                print(f"    Stage {stage}: {count:,} samples")
        print()
    else:
        print("❌ WARNING: No metadata.json file found")
        print()
    
    # Check stage files
    print("Stage files status:")
    print("-" * 40)
    stage_files_found = []
    
    for stage in range(1, 11):  # Check stages 1-10
        stage_file = os.path.join(pretokenized_dir, f"stage_{stage:02d}.pkl")
        temp_file = os.path.join(pretokenized_dir, f"stage_{stage:02d}_temp.pkl")
        
        if os.path.exists(stage_file):
            size = os.path.getsize(stage_file)
            print(f"  Stage {stage:2d}: ✅ Found ({size/1024/1024:.2f} MB)", end="")
            
            # Try to load and check content
            try:
                with open(stage_file, 'rb') as f:
                    data = pickle.load(f)
                print(f" - {len(data):,} samples")
                
                if len(data) > 0:
                    # Check first sample structure
                    sample = data[0]
                    if 'input_ids' in sample and 'attention_mask' in sample:
                        print(f"           Sample shape: input_ids={sample['input_ids'].shape}, "
                              f"attention_mask={sample['attention_mask'].shape}")
                    stage_files_found.append(stage)
                else:
                    print(f"           ⚠️  WARNING: File exists but contains no samples!")
                    
            except Exception as e:
                print(f" - ❌ ERROR loading: {e}")
                
        elif os.path.exists(temp_file):
            size = os.path.getsize(temp_file)
            print(f"  Stage {stage:2d}: ⚠️  Only temp file found ({size/1024/1024:.2f} MB)")
            print(f"           Pre-tokenization may have been interrupted")
        else:
            print(f"  Stage {stage:2d}: ❌ Not found")
    
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY:")
    print("-" * 40)
    
    if not stage_files_found:
        print("❌ CRITICAL: No valid stage files found!")
        print("\nPossible causes:")
        print("1. Pre-tokenization script didn't complete")
        print("2. Files were saved to a different directory")
        print("3. Pre-tokenization failed due to rate limits or errors")
        print("\nRecommended actions:")
        print("1. Check the pre-tokenization job logs for errors")
        print("2. Re-run the pre-tokenization script")
        print("3. Ensure HF_TOKEN is set to avoid rate limits")
        return False
    
    print(f"✅ Found {len(stage_files_found)} valid stage files: {stage_files_found}")
    
    if 1 not in stage_files_found:
        print("\n⚠️  WARNING: Stage 1 is missing!")
        print("This will cause training to fail when starting from stage 1")
        print("Consider starting from the lowest available stage:", min(stage_files_found))
    
    # Check for gaps
    gaps = []
    for i in range(min(stage_files_found), max(stage_files_found)):
        if i not in stage_files_found:
            gaps.append(i)
    
    if gaps:
        print(f"\n⚠️  WARNING: Missing stages in sequence: {gaps}")
        print("This may affect curriculum progression")
    
    return len(stage_files_found) > 0

def suggest_fixes(pretokenized_dir):
    """Suggest fixes based on the diagnostic."""
    print("\n" + "=" * 60)
    print("RECOMMENDED FIXES:")
    print("=" * 60)
    
    # Check if any temp files exist
    temp_files = [f for f in os.listdir(pretokenized_dir) if f.endswith('_temp.pkl')]
    if temp_files:
        print("\n1. Found incomplete pre-tokenization (temp files exist)")
        print("   Run this to convert temp files to final files:")
        print()
        print("   python -c \"")
        print("   import os, pickle")
        print(f"   dir_path = '{pretokenized_dir}'")
        print("   for f in os.listdir(dir_path):")
        print("       if f.endswith('_temp.pkl'):")
        print("           temp_path = os.path.join(dir_path, f)")
        print("           final_path = temp_path.replace('_temp.pkl', '.pkl')")
        print("           os.rename(temp_path, final_path)")
        print("           print(f'Renamed {f} to {f.replace(\"_temp\", \"\")}')")
        print("   \"")
    
    print("\n2. To start training with available stages:")
    print("   Find the lowest available stage number and use:")
    print("   --start_stage <lowest_stage_number>")
    
    print("\n3. To re-run pre-tokenization:")
    print("   sbatch pretokenize.sbatch")
    print("   Make sure to set HF_TOKEN in the script")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pretokenized_dir = sys.argv[1]
    else:
        pretokenized_dir = "/scratch/gautschi/huan2073/pretokenized_curriculum"
    
    success = check_pretokenized_data(pretokenized_dir)
    
    if not success:
        suggest_fixes(pretokenized_dir)
        sys.exit(1)
    else:
        print("\n✅ Pre-tokenized data looks good!")
        sys.exit(0)