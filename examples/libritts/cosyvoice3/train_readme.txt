cosyvoice3进展-20260226

https://doc.weixin.qq.com/sheet/e3_ARkAVQY6AGQCN1gfWHNR1SPCSAAES?scode=AHsAtAeVAAgLk1A1yDARkAVQY6AGQ&tab=BB08J2 

一、环境安装：
1、安装python库，参照https://github.com/FunAudioLLM/CosyVoice 
2、进入/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/S3Tokenizer 运行pip install -e .
3、安装python库，参照/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/3D-Speaker/requirements.txt
4、pip install x_transformers jiwer; apt install -y vim
5、pip install datasets==3.0.1 （因为 S3Tokenizer 和 3D-Speaker 对 datasets 依赖的冲突，后者会安装 3.6.1 的datasets，导致后续跑脚本报错）
6、需要 copy 3D-Speaker 目录到自己训练目录下；
7、需要 copy pretrained 目录到自己的训练目录下；
8、需要 copy tools/ 下脚本到自己的训练目录下；
9、拷贝 cp -R path/to/zdj/cosyvoice  .
10、拷贝 pretrained_models  到目录下;

二、数据（先用西班牙小批量数据熟悉流程，再用多语言数据）：
1、西班牙数据vesta:/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice/examples/libritts/cosyvoice3/train_multi/spanish_5w.tsv
数据预处理脚本：	 +1063：1069行

/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/train_multi_language/3_test_multi.txt
2、多语言数据
数据预处理脚本：/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/gene_data.py +1099：1104行
数据量统计：https://doc.weixin.qq.com/sheet/e3_AQwAJgaMALsCNZTSoKlGLSriJTAgQ?scode=AHsAtAeVAAgaDdZ18FAQwAJgaMALs&tab=BB08J2 

nohup cp -r /gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/zh_en_malay_spanish_arabic_singlish_data_v4 ./ >log.out 2>&1 %
三、code（加速特征提取、train）：
vesta:/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/run.sh 参照脚本的第1到第5步

tensorboard 监控开启，可以用类似 /gpfs01/nfs_share/data20250106/liangguang/workspace/CosyVoice/examples/libritts/cosyvoice3/spanish_5w/tensorboard/cosyvoice3/llm/torch_ddp/ 的目录

四、推理、计算cer
vesta:/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/infer.sh
五、模型：
vesta:/gpfs01/nfs_share/data20250106/zhangdejun/tts/code/CosyVoice-20260105/examples/libritts/cosyvoice3/pretrained_models   
六、triton加速：
参照https://github.com/70557dzqc/CosyVoice/blob/main_panxb/runtime/triton_trtllm/demo_panxb.py 
git：
https://github.com/zdj97/cosyvoice-train 

七、快捷方式：
export MYHOME=/gpfs01/nfs_share/data20250106/liangguang
alias cosy3='cd $MYHOME/workspace/CosyVoice/examples/libritts/cosyvoice3'

 
八、单模型多语言相关进展及代办：
单模型多语言【内部时间节点0312，最后节点0330】
目标：
1. 主观、客观评测与线上一致或提高
2. 效果稳定性达成生产要求
3. no-stream/stream： 推理加速，包括效果，并发测试，稳定性


目标1具体待办项：
a. 多模态评测方式，出cer和llm-score；（如果score不高需要分析是不是真的是生成音频不好，如果真的是音频效果不好，需要判断下是哪方面的问题，不能无条件加数据）
b. 算法内部评测； （当前所有f5模型的音色 跑f5和cosyvoice3的效果 cosyvoice3需要和f5持平或超过）
c. 外部评测。  

目标2具体待办项：
d. zero shot效果稳定性，(使用f5的参考音频和文本走相同的流程评测)
e. 推理稳定性效果评测（各个音色跑批30分钟音频，判断是否有噪音，噪音包括笑声，对话，咳嗽等一切不正常噪音）

目标3待办项：
f. 加速完后走a,b,c,d,e这些流程。
g. 并发压测，统计最佳并发数。

预处理待办项:
h. 声音事件检测：笑声、对话等.
i. 训练时用filename当作spk
j. 分二阶段训练


具体待办项：
1. f5数据汇总
a. 当前f5模型使用的最新数据汇总到同一表格；@宇强  https://doc.weixin.qq.com/sheet/e3_AZ4AZAaaAN8CNGPmsoLfARbqqri96?scode=AHsAtAeVAAgtx1ondEAZ4AZAaaAN8&tab=BB08J2

2. 持续找数据@所有人

3. 数据预处理
a. 上述f5数据进行声音事件检测，使用beats模型或者omni模型，过滤脏数据；@郭虎，@宇强
    beats结果： https://doc.weixin.qq.com/sheet/e3_AZ4AZAaaAN8CNGPmsoLfARbqqri96?scode=AHsAtAeVAAgtx1ondEAZ4AZAaaAN8&tab=BB08J2  根据需要去过滤，可参考C、D列信息
b. 同语言跑多个asr模型，取置信度高的音频；@所有人

4. 模型训练
a. 多机多卡训练cosy3；@德俊，@梁光
b. 二阶段训练；@所有人
注意事项：
第一阶段训练注意数据配比，得出好效果；
以最后一层（YouTube账号/下面的某个视频/vad后的音频），即filename为spk，但是会有重复，需要添加唯一标识，是否需要重新提取embedding或token；
是否要用instruct？

5. 模型评测
a. cer和llm-score得分；
b. 用minimax提供的参考音频和文本跑下我们的模型,跟minimax和11-labs进行比较（即下面那个截图）。https://zhuanlan.zhihu.com/p/1906414542554117392
c. 跑批各语言f5所有音色，以f5为标准，评估cosy3效果。
d. replace需要重新跑批，f5的逻辑可能不使用。

6. 其他
a. 当前v4模型跑澳洲英语，新加坡英语，美式英语，demo给产品评测，看是否有当地感觉；相当于zero shot；
b. 使用cosy3一起训练澳洲英语，新加坡英语，美式英语，demo给产品评测，看是否有当地感觉；
c. 如果没感觉，再分别训澳洲英语，新加坡英语，美式英语3个模型，看是否有当地感觉；3个模型都跑澳洲文本，新加坡文本，美式文本，生成9个音频，
d. 如果还没感觉，使用instruct；



