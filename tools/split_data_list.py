import sys
from glob import glob
main_dir = sys.argv[1]
taif_n = int(sys.argv[2])
data_list_path = glob(main_dir + '/*/parquet/data.list')
assert len(data_list_path) > 0
train_tar_list = []
eval_tar_list = []

for data_list_path in data_list_path:
    with open(data_list_path, 'r') as f:
        data_list = f.readlines()
    data_list = [x.strip() for x in data_list]
    train_sub = data_list[:-taif_n]
    eval_sub = data_list[-taif_n:]
    train_tar_list.extend(train_sub)
    eval_tar_list.extend(eval_sub)
with open(f"{main_dir}/train.data.list", "w") as f:
    f.write("\n".join(train_tar_list))
with open(f"{main_dir}/dev.data.list", "w") as f:
    f.write("\n".join(eval_tar_list))

