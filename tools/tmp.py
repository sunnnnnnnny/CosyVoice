input_dir = "/gpfs01/nfs_share/data20250106/yuqiangz/master_models/aaaaaaaaa_train_data/mexico_exp/mexico_f5_filtered_data"
from glob import glob
from tqdm import tqdm
import os
import json
import shutil

txt_path_list = glob(input_dir + "/*.txt")
txt_path_list.sort()
chunk_id=3
txt_path_list = txt_path_list[(chunk_id-1)*300:chunk_id*300]
res_lines = []
wav_dir = f"{input_dir}_chunk{chunk_id}"
if not os.path.exists(wav_dir):
    os.mkdir(wav_dir)
for txt_path in tqdm(txt_path_list):
    wav_path = txt_path[:-4] + ".wav"
    if not os.path.exists(wav_path):
        wav_path = txt_path[:-4] + ".mp3"
    shutil.copyfile(wav_path, os.path.join(wav_dir, wav_path.split("/")[-1]))
    with open(txt_path, "r") as f:
        lines = f.readlines()
        transcript = lines[0].strip()
        filename = wav_path.split("/")[-1]
        res_lines.append(f"{filename}|{transcript}\n")

with open(f"{wav_dir}.txt", "w") as f:
    for res_line in res_lines:
        f.write(res_line)

