# !/bin/sh

source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding

# For Mistral Model
MODEL_NAME=intern_pm_0508_v2_hard00
CKPT_PATH=/fs-computility/llm/shared/wangyikun/ckpts_1/intern_pm_0508_v2_hard00_20240509072152/intern_pm_0508_v2_hard00_703.pt
TARGET_HF_PATH=/fs-computility/llm/shared/wangyikun/ckpts/intern_pm_0508_v2_0

CODEBASE=/fs-computility/llm/shared/wangyikun/code/InternPrefModel
CUDA_VISIBLE_DEVICES=1 python ${CODEBASE}/run.py reparametrize \
                                        --backbone_type=InternLM \
                                        --init_backbone=/fs-computility/llm/shared/wangyikun/ckpts/internlm2-chat-1_8b-sft \
                                        --pool_type=eos \
                                        --dataset_config=${CODEBASE}/configs/dataset_configs/pref_datasets.yaml \
                                        --target_hf_save_pth=${TARGET_HF_PATH} \
                                        --result_dir ./results \
                                        --which_layer=-1 \
                                        --peft_lora \
                                        --task_prompt \
                                        --embedder_ckpt_path=${CKPT_PATH} \
                                        --embedder_name=$MODEL_NAME

wait
