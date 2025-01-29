# WITH TMP ACCESS, REMOVE MOST OF THESE
export TMPDIR=/data1/joey/tmp
export TMP=/data1/joey/tmp
export TEMP=/data1/joey/tmp
export TEMPDIR=/data1/joey/tmp
export GCC_TMPDIR=/data1/joey/tmp
export NVCC_TMPDIR=/data1/joey/tmp
export TORCH_EXTENSIONS_DIR=/data1/joey/torch_extensions
export HOME=/data1/joey
export DS_BUILD_TEMP_DIR=/data1/joey/tmp
export CCACHE_TEMPDIR=/data1/joey/tmp
export HF_HOME=/data1/joey/hf_cache

source .env


uv run deepspeed --include localhost:0,1,3,4,5,6 train_ppo_custom.py \
  --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --save_path ./checkpoint/test-ppo-3 \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 4 \
  --train_batch_size 48 \
  --micro_rollout_batch_size 8 \
  --rollout_batch_size 48 \
  --n_samples_per_prompt 4 \
  --max_epochs 4 \
  --prompt_max_len 1024 \
  --generate_max_len 10000 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data data/train.json \
  --apply_chat_template \
  --max_samples 1000 \
  --normalize_reward \
  --flash_attn \
  --gradient_checkpointing \
  --remote_rm_url http://localhost:5000/get_reward \
  --use_wandb $WANDB_API_KEY \
  --adam_offload \
  --advantage_estimator grpo

#RETURN THIS WITH TMP ACCESS: --adam_offload \
# --reward_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \