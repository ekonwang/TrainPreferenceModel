# !/bin/sh

source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding

# For Mistral Model
MODEL_NAME=pref_internlm_v1.0
RESULT_DIR=/fs-computility/llm/shared/wangyikun/dump/ad_preference_0508/ad_preference_0508_v2_hard_00k

CKPT_DIR=/fs-computility/llm/shared/wangyikun/ckpts_1/${MODEL_NAME}
step=600
CKPT_NAME=pref_internlm_ckpt_00.pt
CODEBASE=/fs-computility/llm/shared/wangyikun/code/InternPrefModel
# DATASET_CONFIG=${CODEBASE}/configs/dataset_configs/pref_datasets.yaml
DATASET_CONFIG=${CODEBASE}/configs/dataset_configs/ad_preference_0410_${AD_PREF_TAG}.yaml


CUDA_VISIBLE_DEVICES=1 python ${CODEBASE}/run.py evaluate \
                                        --backbone_type=InternLM \
                                        --init_backbone=/fs-computility/llm/shared/wangyikun/ckpts/internlm2-chat-1_8b-sft \
                                        --pool_type=eos \
                                        --dataset_config=${DATASET_CONFIG} \
                                        --result_dir /fs-computility/llm/shared/wangyikun/dump/ad_preference_0326_segmented/AD_Preference/results \
                                        --which_layer=-1 \
                                        --batch_size_per_gpu 20 \
                                        --peft_lora \
                                        --task_prompt \
                                        --embedder_ckpt_path=$CKPT_DIR/$CKPT_NAME \
                                        --dev_mode 0 \
                                        --extract_pseduolabel_0326 0 --ad_preference_0326 1 \
                                        --embedder_name=$MODEL_NAME \
                                        2>&1 | tee ${CODEBASE}/logs/eval_${MODEL_NAME}_${AD_PREF_TAG}.log
wait
