#!/bin/bash

# ========================================
# SUBMISSION WRAPPER SCRIPT
# ========================================
# Usage: ./nl_slrum_master.sh [config_file]
# Default config file: job_config.conf

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get config file from argument or use default
CONFIG_FILE="${1:-job_config.conf}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file '$CONFIG_FILE' not found!${NC}"
    echo "Usage: $0 [config_file]"
    echo "Default config file: job_config.conf"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NL GRAPH TASK JOB SUBMISSION${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Loading configuration from: ${GREEN}$CONFIG_FILE${NC}"

# Source the configuration file
source "$CONFIG_FILE"

# ---------- Defaults for new params (so older configs still work) ----------
# PATH_MODE="${PATH_MODE:-single_path}"          # single_path | all_paths
SANITY_A="${SANITY_A:-false}"
SANITY_B="${SANITY_B:-false}"
SANITY_C="${SANITY_C:-false}"
SANITY_D="${SANITY_D:-false}"
# Optional seed; leave empty to let code pick default
SEED="${SEED:-}"

# ========================================
# GENERATE JOB NAME
# ========================================
# PM_SUFFIX="sp"
# if [ "$PATH_MODE" == "all_paths" ]; then PM_SUFFIX="ap"; fi

JOB_NAME="nl_${TASK}_${ACCURACY_THRESHOLD}acc_${BASE_ALPHA}alpha_LoRA${LORA_RANK}_dropout${LORA_DROPOUT}_${MAX_INPUT_SIZE}size_${FIRST_TOKEN}first_token_${PM_SUFFIX}"

if [ "$NUM_SHOTS" -gt 0 ] 2>/dev/null; then
    JOB_NAME="${JOB_NAME}_${NUM_SHOTS}shot"
fi

# Add task-specific suffixes
if [ "$TASK" == "search" ]; then
    JOB_NAME="${JOB_NAME}_la${MAX_LOOKAHEAD}"
elif [ "$TASK" == "dfs" ]; then
    JOB_NAME="${JOB_NAME}_bt${REQUESTED_BACKTRACK}"
elif [ "$TASK" == "si" ]; then
    JOB_NAME="${JOB_NAME}_fr${MAX_FRONTIER_SIZE}_br${MAX_BRANCH_SIZE}"
else
    echo -e "${RED}Error: Invalid task '$TASK'. Must be one of: si, dfs, search${NC}"
    exit 1
fi

# Add sanity markers to job name
[ "$SANITY_A" == "true" ] && JOB_NAME="${JOB_NAME}_A"
[ "$SANITY_B" == "true" ] && JOB_NAME="${JOB_NAME}_B"
[ "$SANITY_C" == "true" ] && JOB_NAME="${JOB_NAME}_C"
[ "$SANITY_D" == "true" ] && JOB_NAME="${JOB_NAME}_D"
[ -n "$SEED" ] && JOB_NAME="${JOB_NAME}_seed${SEED}"

# ========================================
# CREATE SLURM SCRIPT
# ========================================
SLURM_SCRIPT=$(mktemp /tmp/slurm_job_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << 'EOF'
#!/bin/bash

#SBATCH --mem=___SLURM_MEM___
#SBATCH --gres=gpu:___SLURM_GPUS___
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --time=___SLURM_TIME___
#SBATCH --partition=___SLURM_PARTITION___
#SBATCH --cpus-per-task=___SLURM_CPUS___
#SBATCH -A ___SLURM_ACCOUNT___
#SBATCH -q ___SLURM_QOS___

module load conda
module load ___CUDA_MODULE___
conda activate ___CONDA_ENV___

echo "=========================================="
echo "NATURAL LANGUAGE GRAPH TASK TRAINING"
echo "INFINITE STREAMING WITH CURRICULUM LEARNING"
echo "JOB START $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $HOSTNAME"
echo "Configuration: ___CONFIG_FILE___"
echo "=========================================="

# Show GPU info
nvidia-smi

# Set directories
CACHE_DIR="___CACHE_DIR___"
OUTPUT_DIR="___OUTPUT_BASE_DIR___/nl_graph_output_${SLURM_JOB_ID}"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p ./slurm

# Copy configuration file to output directory
cp "___CONFIG_FILE___" "${OUTPUT_DIR}/job_config.conf"

echo "=========================================="
echo "Configuration Summary:"
echo "  Task: ___TASK___"
echo "  Accuracy Threshold: ___ACCURACY_THRESHOLD___"
echo "  Base Alpha: ___BASE_ALPHA___"
echo "  LoRA Rank: ___LORA_RANK___"
echo "  LoRA Dropout: ___LORA_DROPOUT___"
echo "  Max Input Size: ___MAX_INPUT_SIZE___"
echo "  First Token Soft Weight: ___FIRST_TOKEN___"
echo "  Num Shots: ___NUM_SHOTS___"
echo "  Seed: ___SEED___"
echo "  Cache dir: $CACHE_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo ""
echo "Sanity Checks:"
echo "  A - Random labels: ___SANITY_A___"
echo "  B - Shuffle labels: ___SANITY_B___"
echo "  C - Mask goal: ___SANITY_C___"
echo "  D - Unseen vocab: ___SANITY_D___"
echo "=========================================="

# Install Flash Attention if not already installed
echo "Checking Flash Attention installation..."
python -c "import flash_attn" 2>/dev/null || {
    echo "Installing Flash Attention..."
    pip install flash-attn --no-build-isolation
}

# Compile C++ generator if needed
echo "Checking C++ generator module..."
python -c "import generator" 2>/dev/null || {
    echo "Compiling C++ generator module..."
    python nl_generator.py
}

# Set environment variables
export HF_TOKEN="___HF_TOKEN___"

# GPU optimizations for H100
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Build command for training
ARGS=(
    # Task selection
    --task "___TASK___"
    
    # Model configuration
    --model_name "___MODEL_NAME___"
    --cache_dir "$CACHE_DIR"
    
    # Output
    --output_dir "$OUTPUT_DIR"
    
    # Training parameters
    --batch_size "___BATCH_SIZE___"
    --gradient_accumulation_steps "___GRADIENT_ACCUMULATION_STEPS___"
    --learning_rate "___LEARNING_RATE___"
    --warmup_steps "___WARMUP_STEPS___"
    --min_batch_size "___MIN_BATCH_SIZE___"
    --first_token_soft_weight "___FIRST_TOKEN___"
    
    # Few-shot configuration
    --num_shots "___NUM_SHOTS___"
    
    # Curriculum parameters
    --n_stages "___N_STAGES___"
    --base_alpha "___BASE_ALPHA___"
    --accuracy_threshold "___ACCURACY_THRESHOLD___"
    --min_steps_per_stage "___MIN_STEPS_PER_STAGE___"
    --check_every "___CHECK_EVERY___"
    
    # Task complexity parameters
    --max_input_size "___MAX_INPUT_SIZE___"
    --max_lookahead "___MAX_LOOKAHEAD___"
    --requested_backtrack "___REQUESTED_BACKTRACK___"
    --max_frontier_size "___MAX_FRONTIER_SIZE___"
    --max_branch_size "___MAX_BRANCH_SIZE___"
    
    # Evaluation
    --eval_samples "___EVAL_SAMPLES___"
    --eval_batch_size "___EVAL_BATCH_SIZE___"
    --print_eval_examples "___PRINT_EVAL_EXAMPLES___"
    
    # Diagnostics
    --diagnostic_stages "___DIAGNOSTIC_STAGES___"
    --align_debug "___ALIGN_DEBUG___"
)

# Conditional arguments
if [ "___USE_LORA___" == "true" ]; then
    ARGS+=(--use_lora --lora_rank "___LORA_RANK___" --lora_dropout "___LORA_DROPOUT___")
fi

if [ "___USE_FLASH_ATTENTION___" == "true" ]; then
    ARGS+=(--use_flash_attention)
fi

if [ "___EVAL_AFTER_STAGE___" == "true" ]; then
    ARGS+=(--eval_after_stage)
fi

if [ "___NO_CURRICULUM___" == "true" ]; then
    ARGS+=(--no_curriculum)
fi

# Seed (only if provided)
if [ -n "___SEED___" ]; then
    ARGS+=(--seed "___SEED___")
fi

# Sanity checks (only if enabled)
if [ "___SANITY_A___" == "true" ]; then
    ARGS+=(--sanity_A)
fi
if [ "___SANITY_B___" == "true" ]; then
    ARGS+=(--sanity_B)
fi
if [ "___SANITY_C___" == "true" ]; then
    ARGS+=(--sanity_C)
fi
if [ "___SANITY_D___" == "true" ]; then
    ARGS+=(--sanity_D)
fi

# Monitor GPU usage in background
(
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
                  --format=csv >> "${OUTPUT_DIR}/gpu_usage.log"
        sleep 30
    done
) &
GPU_MONITOR_PID=$!

# Run the training
echo "Starting training with command:"
echo "python qwen_tuning_nl.py ${ARGS[@]}"
echo ""

python qwen_tuning_nl.py "${ARGS[@]}"
TRAINING_EXIT_CODE=$?

# Kill GPU monitor
kill $GPU_MONITOR_PID 2>/dev/null

# Check exit status
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "JOB COMPLETED SUCCESSFULLY $(date)"
    echo "Job Name: $SLURM_JOB_NAME"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=========================================="
else
    echo "=========================================="
    echo "JOB FAILED $(date) with exit code $TRAINING_EXIT_CODE"
    echo "Job Name: $SLURM_JOB_NAME"
    echo "=========================================="
    exit $TRAINING_EXIT_CODE
fi

# Print final statistics
echo "Final GPU status:"
nvidia-smi

echo "=========================================="
echo "Training Statistics for: $SLURM_JOB_NAME"

# GPU utilization statistics
if [ -f "${OUTPUT_DIR}/gpu_usage.log" ]; then
    echo "Average GPU utilization:"
    awk -F',' 'NR>1 {sum+=$2; count++} END {print sum/count "%"}' "${OUTPUT_DIR}/gpu_usage.log"
fi

# Stage history
TASK_DIR="${OUTPUT_DIR}/___TASK___"
if [ -f "${TASK_DIR}/stage_history.json" ]; then
    echo ""
    echo "Curriculum Stage Progression:"
    python -c "
import json
with open('${TASK_DIR}/stage_history.json', 'r') as f:
    history = json.load(f)
    total_steps = 0
    for stage_info in history:
        total_steps += stage_info['steps']
        print(f\"  Stage {stage_info['stage']}: {stage_info['steps']:,} steps, Final Acc={stage_info['final_accuracy']*100:.2f}%, Loss={stage_info['final_loss']:.4f}\")
    print(f\"\nTotal stages completed: {len(history)}/___N_STAGES___\")
    print(f\"Total training steps: {total_steps:,}\")
"
fi

# Final evaluation results
if [ -f "${TASK_DIR}/final_results.json" ]; then
    echo ""
    echo "Final Evaluation Results:"
    python -c "
import json
with open('${TASK_DIR}/final_results.json', 'r') as f:
    results = json.load(f)
    print(f'  Task: {results[\"task\"]}')
    print(f'  Accuracy: {results[\"accuracy\"]*100:.2f}%')
    print(f'  Correct: {results[\"correct\"]}/{results[\"total\"]}')
"
fi

# Baseline comparison
if [ -f "${TASK_DIR}/baseline_results.json" ] && [ -f "${TASK_DIR}/final_results.json" ]; then
    echo ""
    echo "Improvement over baseline:"
    python -c "
import json
with open('${TASK_DIR}/baseline_results.json', 'r') as f:
    baseline = json.load(f)
with open('${TASK_DIR}/final_results.json', 'r') as f:
    final = json.load(f)
improvement = (final['accuracy'] - baseline['accuracy']) * 100
print(f'  Baseline: {baseline[\"accuracy\"]*100:.2f}%')
print(f'  Final: {final[\"accuracy\"]*100:.2f}%')
print(f'  Improvement: +{improvement:.2f}%')
"
fi

echo ""
echo "Output files created:"
ls -lh "${TASK_DIR}/" 2>/dev/null || ls -lh "$OUTPUT_DIR/"

echo "=========================================="
echo "END OF JOB REPORT"
echo "=========================================="
EOF

# Replace placeholders with actual values
safe_replace() {
    local file="$1"
    local placeholder="$2"
    local value="$3"
    sed -i "s|___${placeholder}___|${value}|g" "$file"
}

# Replace all placeholders (order matters - longer names first!)
safe_replace "$SLURM_SCRIPT" "GRADIENT_ACCUMULATION_STEPS" "${GRADIENT_ACCUMULATION_STEPS}"
safe_replace "$SLURM_SCRIPT" "MIN_STEPS_PER_STAGE" "${MIN_STEPS_PER_STAGE}"
safe_replace "$SLURM_SCRIPT" "ACCURACY_THRESHOLD" "${ACCURACY_THRESHOLD}"
safe_replace "$SLURM_SCRIPT" "REQUESTED_BACKTRACK" "${REQUESTED_BACKTRACK}"
safe_replace "$SLURM_SCRIPT" "USE_FLASH_ATTENTION" "${USE_FLASH_ATTENTION}"
safe_replace "$SLURM_SCRIPT" "EVAL_AFTER_STAGE" "${EVAL_AFTER_STAGE}"
safe_replace "$SLURM_SCRIPT" "PRINT_EVAL_EXAMPLES" "${PRINT_EVAL_EXAMPLES}"
safe_replace "$SLURM_SCRIPT" "DIAGNOSTIC_STAGES" "${DIAGNOSTIC_STAGES}"
safe_replace "$SLURM_SCRIPT" "MAX_FRONTIER_SIZE" "${MAX_FRONTIER_SIZE}"
safe_replace "$SLURM_SCRIPT" "MAX_BRANCH_SIZE" "${MAX_BRANCH_SIZE}"
safe_replace "$SLURM_SCRIPT" "OUTPUT_BASE_DIR" "${OUTPUT_BASE_DIR}"
safe_replace "$SLURM_SCRIPT" "MAX_INPUT_SIZE" "${MAX_INPUT_SIZE}"
safe_replace "$SLURM_SCRIPT" "EVAL_BATCH_SIZE" "${EVAL_BATCH_SIZE}"
safe_replace "$SLURM_SCRIPT" "MIN_BATCH_SIZE" "${MIN_BATCH_SIZE}"
safe_replace "$SLURM_SCRIPT" "MAX_LOOKAHEAD" "${MAX_LOOKAHEAD}"
safe_replace "$SLURM_SCRIPT" "LEARNING_RATE" "${LEARNING_RATE}"
safe_replace "$SLURM_SCRIPT" "WARMUP_STEPS" "${WARMUP_STEPS}"
safe_replace "$SLURM_SCRIPT" "EVAL_SAMPLES" "${EVAL_SAMPLES}"
safe_replace "$SLURM_SCRIPT" "CHECK_EVERY" "${CHECK_EVERY}"
safe_replace "$SLURM_SCRIPT" "SLURM_PARTITION" "${SLURM_PARTITION}"
safe_replace "$SLURM_SCRIPT" "SLURM_ACCOUNT" "${SLURM_ACCOUNT}"
safe_replace "$SLURM_SCRIPT" "NO_CURRICULUM" "${NO_CURRICULUM:-false}"
safe_replace "$SLURM_SCRIPT" "SLURM_TIME" "${SLURM_TIME}"
safe_replace "$SLURM_SCRIPT" "SLURM_CPUS" "${SLURM_CPUS}"
safe_replace "$SLURM_SCRIPT" "SLURM_GPUS" "${SLURM_GPUS}"
safe_replace "$SLURM_SCRIPT" "SLURM_MEM" "${SLURM_MEM}"
safe_replace "$SLURM_SCRIPT" "SLURM_QOS" "${SLURM_QOS}"
safe_replace "$SLURM_SCRIPT" "CONFIG_FILE" "$(realpath ${CONFIG_FILE})"
safe_replace "$SLURM_SCRIPT" "MODEL_NAME" "${MODEL_NAME}"
safe_replace "$SLURM_SCRIPT" "BASE_ALPHA" "${BASE_ALPHA}"
safe_replace "$SLURM_SCRIPT" "BATCH_SIZE" "${BATCH_SIZE}"
safe_replace "$SLURM_SCRIPT" "CACHE_DIR" "${CACHE_DIR}"
safe_replace "$SLURM_SCRIPT" "CONDA_ENV" "${CONDA_ENV}"
safe_replace "$SLURM_SCRIPT" "CUDA_MODULE" "${CUDA_MODULE}"
safe_replace "$SLURM_SCRIPT" "HF_TOKEN" "${HF_TOKEN}"
safe_replace "$SLURM_SCRIPT" "LORA_RANK" "${LORA_RANK}"
safe_replace "$SLURM_SCRIPT" "LORA_DROPOUT" "${LORA_DROPOUT}"
safe_replace "$SLURM_SCRIPT" "NUM_SHOTS" "${NUM_SHOTS}"
safe_replace "$SLURM_SCRIPT" "N_STAGES" "${N_STAGES}"
safe_replace "$SLURM_SCRIPT" "FIRST_TOKEN" "${FIRST_TOKEN}"
safe_replace "$SLURM_SCRIPT" "USE_LORA" "${USE_LORA}"
safe_replace "$SLURM_SCRIPT" "ALIGN_DEBUG" "${ALIGN_DEBUG}"
safe_replace "$SLURM_SCRIPT" "TASK" "${TASK}"

# New placeholders
safe_replace "$SLURM_SCRIPT" "SANITY_A" "${SANITY_A}"
safe_replace "$SLURM_SCRIPT" "SANITY_B" "${SANITY_B}"
safe_replace "$SLURM_SCRIPT" "SANITY_C" "${SANITY_C}"
safe_replace "$SLURM_SCRIPT" "SANITY_D" "${SANITY_D}"
safe_replace "$SLURM_SCRIPT" "SEED" "${SEED}"

# Handle optional SLURM mail settings
if [ -n "$SLURM_MAIL_USER" ]; then
    sed -i "/#SBATCH -q/a #SBATCH --mail-user=${SLURM_MAIL_USER}" "$SLURM_SCRIPT"
    if [ -n "$SLURM_MAIL_TYPE" ]; then
        sed -i "/#SBATCH --mail-user/a #SBATCH --mail-type=${SLURM_MAIL_TYPE}" "$SLURM_SCRIPT"
    fi
fi

# ========================================
# DISPLAY CONFIGURATION
# ========================================
echo ""
echo -e "${YELLOW}Job Configuration:${NC}"
echo "  Job Name: ${GREEN}$JOB_NAME${NC}"
echo "  Task: $TASK"
echo "  Model: $MODEL_NAME"
echo "  LoRA: ${USE_LORA} (rank ${LORA_RANK}, dropout ${LORA_DROPOUT})"
echo "  Accuracy Threshold: $ACCURACY_THRESHOLD"
echo "  Base Alpha: $BASE_ALPHA"
echo "  Max Input Size: $MAX_INPUT_SIZE"
echo "  First Token Soft Weight: $FIRST_TOKEN"
echo "  Num Shots: $NUM_SHOTS"
[ -n "$SEED" ] && echo "  Seed: $SEED"

if [ "$TASK" == "search" ]; then
    echo "  Max Lookahead: $MAX_LOOKAHEAD"
elif [ "$TASK" == "dfs" ]; then
    echo "  Requested Backtrack: $REQUESTED_BACKTRACK"
elif [ "$TASK" == "si" ]; then
    echo "  Max Frontier Size: $MAX_FRONTIER_SIZE"
    echo "  Max Branch Size: $MAX_BRANCH_SIZE"
fi

echo ""
echo -e "${YELLOW}Sanity Checks:${NC}"
echo "  A - Random labels: $SANITY_A"
echo "  B - Shuffle labels: $SANITY_B"
echo "  C - Mask goal: $SANITY_C"
echo "  D - Unseen vocab: $SANITY_D"

echo ""
echo -e "${YELLOW}SLURM Resources:${NC}"
echo "  Partition: $SLURM_PARTITION"
echo "  Memory: $SLURM_MEM"
echo "  GPUs: $SLURM_GPUS"
echo "  CPUs: $SLURM_CPUS"
echo "  Time: $SLURM_TIME"
echo "  QOS: $SLURM_QOS"

# ========================================
# CONFIRMATION
# ========================================
echo ""
read -p "Do you want to submit this job? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Job submission cancelled.${NC}"
    rm "$SLURM_SCRIPT"
    exit 0
fi

# ========================================
# SUBMIT JOB
# ========================================
echo ""
echo -e "${BLUE}Submitting job...${NC}"

# Submit with explicit partition and job name
SUBMIT_OUTPUT=$(sbatch --partition="${SLURM_PARTITION}" --job-name="${JOB_NAME}" "$SLURM_SCRIPT" 2>&1)
SUBMIT_CODE=$?

# Clean up temp script
rm "$SLURM_SCRIPT"

# Check submission status
if [ $SUBMIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}JOB SUBMITTED SUCCESSFULLY!${NC}"
    echo -e "${GREEN}$SUBMIT_OUTPUT${NC}"
    echo -e "${GREEN}Job Name: $JOB_NAME${NC}"
    
    # Extract job ID
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oE '[0-9]+')
    if [ -n "$JOB_ID" ]; then
        echo -e "${GREEN}Job ID: $JOB_ID${NC}"
        echo ""
        echo "Monitor your job with:"
        echo "  squeue -u $USER"
        echo "  sacct -j $JOB_ID"
        echo "  tail -f ./slurm/${JOB_ID}_${JOB_NAME}.out"
        
        # Save submission info
        echo "$JOB_ID|$JOB_NAME|$(date)|$CONFIG_FILE" >> job_submissions.log
    fi
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}JOB SUBMISSION FAILED!${NC}"
    echo -e "${RED}Error: $SUBMIT_OUTPUT${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

# ========================================
# POST-SUBMISSION OPTIONS
# ========================================
echo ""
echo "Additional options:"
echo "1. Create another configuration for a different experiment"
echo "2. Monitor job queue (squeue)"
echo "3. Exit"
echo ""
read -p "Select option (1-3): " -n 1 -r
echo ""

case $REPLY in
    1)
        echo "Creating new configuration file..."
        NEW_CONFIG="job_config_$(date +%Y%m%d_%H%M%S).conf"
        cp "$CONFIG_FILE" "$NEW_CONFIG"
        echo "Created: $NEW_CONFIG"
        echo "Edit this file and run: $0 $NEW_CONFIG"
        ;;
    2)
        echo "Current job queue:"
        squeue -u $USER
        ;;
    3)
        echo "Exiting..."
        ;;
    *)
        echo "Invalid option"
        ;;
esac