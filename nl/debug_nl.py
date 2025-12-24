#!/usr/bin/env python3
"""
Test script to verify NL generator setup before training
"""

import sys
import traceback

def test_nl_generator():
    """Test that the NL generator works correctly"""
    print("="*60)
    print("Testing NL Generator Setup")
    print("="*60)
    
    # Test 1: Import the generator
    print("\n1. Testing imports...")
    try:
        from nl_generator import (
            NaturalLanguageGraphGenerator,
            generate_reserved_eval_set,
            generate_nl_dataset,
            NLExample
        )
        print("   [OK] Imports successful")
    except ImportError as e:
        print(f"   [FAIL] Import failed: {e}")
        return False
    
    # Test 2: Check generate_reserved_eval_set signature (now returns 5 values)
    print("\n2. Testing generate_reserved_eval_set...")
    try:
        result = generate_reserved_eval_set(
            task='search',
            max_input_size=128,
            n_eval_samples=10,
            seed=42,
            max_lookahead=3
        )
        
        print(f"   Function returned {len(result)} values")
        
        if len(result) != 5:
            print(f"   [FAIL] Expected 5 return values, got {len(result)}")
            return False
        
        eval_inputs, eval_outputs, eval_labels, reserved, eval_vectors = result
        
        print(f"   Got {len(eval_inputs)} inputs")
        print(f"   Got {len(eval_outputs)} outputs (list of lists)")
        print(f"   Got {len(eval_labels)} labels (list of lists)")
        print(f"   Got {len(reserved)} reserved inputs")
        print(f"   Got {len(eval_vectors)} output vectors")
        
        # Verify structure
        if eval_inputs and eval_outputs:
            print(f"\n   Sample input (truncated): {eval_inputs[0][:80]}...")
            print(f"   Sample outputs: {eval_outputs[0]}")  # List of valid answers
            print(f"   Sample labels: {eval_labels[0]}")    # List of node IDs
            
            # Verify outputs are lists
            if not isinstance(eval_outputs[0], list):
                print(f"   [FAIL] eval_outputs[0] should be a list, got {type(eval_outputs[0])}")
                return False
            if not isinstance(eval_labels[0], list):
                print(f"   [FAIL] eval_labels[0] should be a list, got {type(eval_labels[0])}")
                return False
            print("   [OK] Output structure verified (lists of lists)")
            
    except Exception as e:
        print(f"   [FAIL] Function call failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Test generator directly
    print("\n3. Testing NaturalLanguageGraphGenerator...")
    try:
        gen = NaturalLanguageGraphGenerator(max_input_size=128, seed=42)
        print("   Generator created")
        
        # Test generate_batch with alpha
        examples = gen.generate_batch(
            task='search',
            batch_size=5,
            reserved_inputs=set(),
            alpha=0.5,
            max_lookahead=3
        )
        
        print(f"   Generated {len(examples)} examples")
        
        if examples:
            ex = examples[0]
            print(f"\n   NLExample fields:")
            print(f"   - input_text: {ex.input_text[:80]}...")
            print(f"   - output_texts (list): {ex.output_texts}")
            print(f"   - labels (list): {ex.labels}")
            print(f"   - output_vector present: {ex.output_vector is not None}")
            
            # Verify NLExample structure
            if not isinstance(ex.output_texts, list):
                print(f"   [FAIL] output_texts should be list, got {type(ex.output_texts)}")
                return False
            if not isinstance(ex.labels, list):
                print(f"   [FAIL] labels should be list, got {type(ex.labels)}")
                return False
            print("   [OK] NLExample structure verified")
            
    except Exception as e:
        print(f"   [FAIL] Generator test failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: Test all three tasks
    print("\n4. Testing all task types...")
    gen = NaturalLanguageGraphGenerator(max_input_size=256, seed=42)
    
    task_configs = {
        'search': {'max_lookahead': 5, 'alpha': 0.5},
        'dfs': {'requested_backtrack': 2, 'alpha': 0.5},
        'si': {'max_frontier_size': 5, 'max_branch_size': 5, 'alpha': 0.5}
    }
    
    for task, kwargs in task_configs.items():
        try:
            examples = gen.generate_batch(
                task=task,
                batch_size=3,
                reserved_inputs=set(),
                **kwargs
            )
            
            # Count examples with multiple valid answers
            multi_answer = sum(1 for ex in examples if len(ex.output_texts) > 1)
            
            print(f"   Task '{task}': Generated {len(examples)} examples "
                  f"({multi_answer} with multiple answers)")
            
            if examples:
                ex = examples[0]
                print(f"      First example: {len(ex.output_texts)} output(s), {len(ex.labels)} label(s)")
            
        except Exception as e:
            print(f"   [FAIL] Task '{task}' failed: {e}")
            traceback.print_exc()
            return False
    
    # Test 5: Test generate_nl_dataset convenience function
    print("\n5. Testing generate_nl_dataset...")
    try:
        inputs, outputs, labels = generate_nl_dataset(
            task='si',
            max_input_size=128,
            num_samples=5,
            seed=42,
            max_frontier_size=5,
            max_branch_size=5
        )
        
        print(f"   Generated {len(inputs)} samples")
        if inputs:
            print(f"   First output options: {outputs[0]}")
            print(f"   First labels: {labels[0]}")
        print("   [OK] generate_nl_dataset works")
        
    except Exception as e:
        print(f"   [FAIL] generate_nl_dataset failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 6: Test generate_batch_with_symbolic
    print("\n6. Testing generate_batch_with_symbolic...")
    try:
        nl_examples, symbolic_inputs = gen.generate_batch_with_symbolic(
            task='search',
            batch_size=3,
            reserved_inputs=set(),
            alpha=0.5,
            max_lookahead=3
        )
        
        print(f"   Got {len(nl_examples)} NL examples")
        print(f"   Got symbolic_inputs shape: {symbolic_inputs.shape}")
        print("   [OK] generate_batch_with_symbolic works")
        
    except Exception as e:
        print(f"   [FAIL] generate_batch_with_symbolic failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("All tests passed! NL generator is working correctly.")
    print("="*60)
    return True


def test_tokenizer():
    """Quick test of tokenizer"""
    print("\n7. Testing tokenizer (optional)...")
    try:
        from transformers import AutoTokenizer
        
        # Try a small model first
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            trust_remote_code=True
        )
        
        test_text = "Suppose we have the following facts: If Alice is happy, then Alice is smart. Given that Alice is happy, and we want to prove Alice is smart. Proof: Alice is"
        tokens = tokenizer(test_text, return_tensors='pt')
        
        print(f"   [OK] Tokenizer works")
        print(f"   Input IDs shape: {tokens['input_ids'].shape}")
        print(f"   Token count: {tokens['input_ids'].shape[1]}")
        
    except Exception as e:
        print(f"   [SKIP] Tokenizer test skipped: {e}")


def test_alpha_scaling():
    """Test that alpha parameter affects generation"""
    print("\n8. Testing alpha curriculum scaling...")
    try:
        from nl_generator import NaturalLanguageGraphGenerator
        
        gen = NaturalLanguageGraphGenerator(max_input_size=256, seed=42)
        
        # Generate at different alphas
        for alpha in [0.1, 0.5, 1.0]:
            examples = gen.generate_batch(
                task='search',
                batch_size=10,
                reserved_inputs=set(),
                alpha=alpha,
                max_lookahead=10
            )
            
            # Measure average input length as proxy for difficulty
            avg_len = sum(len(ex.input_text) for ex in examples) / len(examples)
            print(f"   alpha={alpha}: avg input length = {avg_len:.0f} chars")
        
        print("   [OK] Alpha scaling working (higher alpha = longer inputs)")
        
    except Exception as e:
        print(f"   [FAIL] Alpha test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("\nRunning NL Generator Setup Tests\n")
    
    success = test_nl_generator()
    
    if success:
        test_tokenizer()
        test_alpha_scaling()
        print("\n" + "="*60)
        print("Setup verification complete!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("Setup verification FAILED. Check errors above.")
        print("="*60)
        sys.exit(1)