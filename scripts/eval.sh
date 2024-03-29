# !/bin/sh

source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding

# For Mistral Model
# MODEL_NAME=mistral_embedder43_sfr_eos1_prompt_hns1_matryoshka1_temp1_lr15_bs600_ml512_500
# CKPT_DIR=mistral_indataset43_20240323234208
# CKPT_NAME=mistral_indataset43_500.pt

# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=Mistral \
#                                         --pool_type=eos \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &

# MODEL_NAME=mistral_embedder28_sfr_eos1_prompt_ins_matryoshka1_temp1_lr15_bs600_ml512_800
# CKPT_DIR=mistral_sfr28_20240315051830
# CKPT_NAME=mistral_sfr28_800.pt
# CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=Mistral \
#                                         --pool_type=eos \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &

# For BGE Model
# MODEL_NAME=bge_embedder48_indataset_cls1_prompt_hns1_matryoshka1_temp15_lr15_bs1600_ml512_1000
# CKPT_DIR=bge_indataset48_adaptive_paired_prompt_20240327185931
# CKPT_NAME=bge_indataset48_adaptive_paired_prompt_1000.pt

# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=BGE \
#                                         --init_backbone=BAAI/bge-base-en-v1.5 \
#                                         --pool_type=cls \
#                                         --mytryoshka_size=768 \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &

# MODEL_NAME=bge_embedder48_indataset_cls1_prompt_hns1_matryoshka1_temp1_lr15_bs1600_ml512_1062_clipgradient_noprompt
# CKPT_DIR=bge_indataset48_adaptive_paired_prompt_20240325052808
# CKPT_NAME=bge_indataset48_adaptive_paired_prompt_1062.pt

# CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=BGE \
#                                         --init_backbone=BAAI/bge-base-en-v1.5 \
#                                         --pool_type=cls \
#                                         --mytryoshka_size=768 \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &


# # For InternLM Model

# MODEL_NAME=internlm_embedder48_sfr_eos1_prompt_hns1_matryoshka1_temp1_lr15_bs1024_ml512_200
# CKPT_DIR=internlm_indataset48_adaptive_paired_prompt_20240326104215
# CKPT_NAME=internlm_indataset48_adaptive_paired_prompt_200.pt

# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=InternLM \
#                                         --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
#                                         --pool_type=eos \
#                                         --mytryoshka_size=2048 \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &

# MODEL_NAME=internlm_embedder48_sfr_pw1_prompt_hns1_matryoshka1_temp1_lr15_bs1024_ml512_200
# CKPT_DIR=internlm_indataset48_adaptive_paired_prompt_20240327103819
# CKPT_NAME=internlm_indataset48_adaptive_paired_prompt_200.pt
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=InternLM \
#                                         --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
#                                         --pool_type=position_weight \
#                                         --mytryoshka_size=2048 \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &

MODEL_NAME=internlm_layer_1_real

CUDA_VISIBLE_DEVICES=0 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
                                        --backbone_type=InternLM \
                                        --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
                                        --pool_type=position_weight \
                                        --which_layer=-1 \
                                        --mytryoshka_size=2048 \
                                        --task_prompt \
                                        --embedder_name=$MODEL_NAME &

MODEL_NAME=internlm_layer_5

CUDA_VISIBLE_DEVICES=1 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
                                        --backbone_type=InternLM \
                                        --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
                                        --pool_type=position_weight \
                                        --which_layer=-5 \
                                        --mytryoshka_size=2048 \
                                        --task_prompt \
                                        --embedder_name=$MODEL_NAME &

wait