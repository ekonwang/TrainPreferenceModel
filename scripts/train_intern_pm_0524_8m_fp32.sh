#!/bin/sh
source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding

CODE=run.py
CKPT_NAME=intern_pm_0524_8m_fp32
DATASET_CONFIG=configs/dataset_configs/intern_pm_0524_8m.yaml
EXTRACT_PSEDUO=0  # 抽取 gpt3.5 伪标签

accelerate launch --config_file scripts/train_pref_internlm_fp32.yaml \
                                           $CODE train \
                                           --pool_type=eos \
                                           --init_backbone=/fs-computility/llm/shared/wangyikun/ckpts/internlm2-chat-1_8b-sft \
                                           --backbone_type=InternLM \
                                           --peft_lora \
                                           --embedding_norm \
                                           --task_adaptation \
                                           --sampler=random \
                                           --warmup_rate=0.1 \
                                           --num_epochs 1 \
                                           --checkpoint_batch_size=10 \
                                           --gradcache_chunk_size=10 \
                                           --temperature=1. \
                                           --learning_rate=1e-5 \
                                           --save_ckpt_steps=100 \
                                           --batch_size_per_gpu=128 \
                                           --dataset_config=$DATASET_CONFIG \
                                           --embedder_name=$CKPT_NAME \
                                           --extract_pseduolabel_0326 ${EXTRACT_PSEDUO}  \
                                           --ad_preference_0326 1 \
                                           --ckpt_saving_dir=/fs-computility/llm/shared/wangyikun/ckpts_1 2>&1 | tee logs/${CKPT_NAME}.log
