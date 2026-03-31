# Core Issue
- modeling whole sequence (which is already much shorter than the c4)
    - C4 is 2048 prediction tokens
    - Synthetic is ~600

Fix 1: We really only care about the predictions on the next part of the proof on the reasoning data
    - Only do the loss on the next part of the prood

Example
- 100 examples in batch
- C4: 95 * 2048 = 194,560 tokens
- Synthetic: 5 * 3 = 15 (only really like 3 prediction tokens)
- Actual Percent: 0.0077% actually contributing to the loss
