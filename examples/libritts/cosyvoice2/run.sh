#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=5
stop_stage=5

data_url=www.openslr.org/resources/60
# data_dir=/gpfs01/nfs_share/data20250106/yuqiangz/master_models/aaaaaaaaa_train_data/mandarin_call_out_add/0225_spk_female3_male1/0226_exp_male1_female3_female_sample329
pretrained_model_dir=/gpfs01/nfs_share/data20250106/yuqiangz/online_models/CosyVoice2-0.5B-0904-yuqiang-female-yilina-vesta
# spks="0113_外呼录音_guohu_sample196_annotated_filtered_day0126 成熟稳重音色_vocal 邻家亲和音色_vocal 甜美音色_vocal"
# data_exp="0226_exp_male1_female3_female_sample329"
# model_exp="0226_exp_male1_female3_female_sample329_ft"
# model_exp="0226_exp_male1_female3_female_sample329_ft_lr1e5"

data_dir="/gpfs01/nfs_share/data20250106/yuqiangz/master_models/aaaaaaaaa_train_data/mandarin_call_out_add/0225_spk_female3_male1/0303_exp_male1_female3-w-denoise-wo-roformer"
data_exp="0303_exp_male1_female3-w-denoise-wo-roformer"
model_exp="0303_exp_male1_female3-w-denoise-wo-roformer_ft_lr5e7"
spks="0113_外呼录音_guohu_sample196_annotated_filtered_day0126 成熟稳重音色 邻家亲和音色 甜美音色"

data_dir="/gpfs01/nfs_share/data20250106/yuqiangz/master_models/aaaaaaaaa_train_data/mandarin_call_out_add/0225_spk_female3_male1/0303_exp_male1_female3-w-denoise-wo-roformer"
data_exp="0303_exp_male1_female3-w-denoise-wo-roformer"
model_exp="0303_exp_male1_female3-w-denoise-wo-roformer_ft_lr5e7"
spks="0113_外呼录音_guohu_sample196_annotated_filtered_day0126 成熟稳重音色 邻家亲和音色 甜美音色"


# data_dir=/gpfs01/nfs_share/data20250106/yuqiangz/master_models/aaaaaaaaa_train_data/mandarin_call_out_add/0225_spk_female3_male1/0227_exp_male1_female3-wo-denoise
# spks="0113_外呼录音_guohu_sample196_annotated_filtered_day0126 成熟稳重音色 邻家亲和音色 甜美音色"
# data_exp="0227_exp_male1_female3-wo-denoise"
# model_exp="0227_exp_male1_female3-wo-denoise_ft"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Data Download"
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    local/download_and_untar.sh ${data_dir} ${data_url} ${part}
  done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in $spks; do
    mkdir -p data/${data_exp}/$x
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/${data_exp}/$x
  done
fi

# NOTE embedding/token extraction is not necessary now as we support online feature extraction, but training speed will be influenced
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in ${spks}; do
    /gpfs01/nfs_share/data20250106/yuqiangz/master_models/CosyVoice/tools/extract_embedding.py --dir data/${data_exp}/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in ${spks}; do
    /gpfs01/nfs_share/data20250106/yuqiangz/master_models/CosyVoice/tools/extract_speech_token.py --dir data/${data_exp}/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in ${spks}; do
    mkdir -p data/${data_exp}/$x/parquet
    /gpfs01/nfs_share/data20250106/yuqiangz/master_models/CosyVoice/tools/make_parquet_list.py --num_utts_per_parquet 10 \
      --num_processes 10 \
      --src_dir data/${data_exp}/$x \
      --des_dir data/${data_exp}/$x/parquet
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "split data_list"
  python /gpfs01/nfs_share/data20250106/yuqiangz/master_models/CosyVoice/tools/split_data_list.py data/${data_exp} 2
fi

# train llm
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      ../../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2.yaml \
      --train_data data/${data_exp}/train.data.list \
      --cv_data data/${data_exp}/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$model_exp \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$model_exp \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi