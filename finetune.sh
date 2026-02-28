#!/bin/bash
# 1. 环境变量设置，防止多卡通讯冲突
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# 2. 硬件资源配置
GPUS_PER_NODE=4                  # 每台机器的显卡数
NNODES=${NNODES:-1}              # 总机器数
NODE_RANK=${NODE_RANK:-0}        # 当前机器的编号
MASTER_ADDR=${MASTER_ADDR:-localhost} # 主机 IP
MASTER_PORT=${MASTER_PORT:-6001} # 通讯端口

# 3. 核心模型与数据路径
# 指向你执行 merge_and_save_new_model 后生成的目录
MODEL="save/v1"                 
TS_DATA="data/v1/ts.csv"
NEWS_DATA="data/v1/news.csv"
INDEX_DATA="data/v1/index.csv"
SAVE="save/qwen-ts-v1"        # 训练结果保存路径

# 4. 训练策略配置
DS_CONFIG_PATH="/s3mount/acf876c3-62e9-4ab1-9aab-e2eea5953462/kouxin/notebook/work/mmts/src/ds_config_zero2.json" # 建议先从 ZeRO-2 开始，显存足够的话更稳定
USE_LORA=False
Q_LORA=False                     # 如果显存极度紧张可开启 Q-LoRA
TUNE_ADAPTER_ONLY=True

# 5. 参数解析逻辑 (保持与原脚本一致，方便命令行覆盖)
while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model ) shift; MODEL=$1 ;;
        -d | --data ) shift; DATA=$1 ;;
        --deepspeed ) shift; DS_CONFIG_PATH=$1 ;;
        --use_lora  ) shift; USE_LORA=$1 ;;
        --q_lora    ) shift; Q_LORA=$1 ;;
        * ) echo "Unknown argument ${1}"; exit 1 ;;
    esac
    shift
done

# 6. 分布式参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 7. 启动训练
torchrun $DISTRIBUTED_ARGS src/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $INDEX_DATA \
    --ts_data_path $TS_DATA \
    --news_data_path $NEWS_DATA \
    --bf16 True \
    --output_dir $SAVE \
    --dataloader_num_workers 4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --max_news 3 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --tune_ts_adapter_only ${TUNE_ADAPTER_ONLY} \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --deepspeed ${DS_CONFIG_PATH}