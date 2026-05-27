import os
os.system("pip install func_timeout")
os.system("pip install pydub")
os.system("pip install ffmpeg-python")
import requests
import time
import numpy as np
import torch
import base64
import json
import traceback
import uvicorn
import func_timeout
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../..".format(ROOT_DIR))
sys.path.append("{}/../../third_party/Matcha-TTS".format(ROOT_DIR))
from func_timeout import func_set_timeout
# from tools.log_panxb import log, Loggers

# from tools.save_cloud import get_s3_client
import os
from fastapi import FastAPI, UploadFile, Form, File, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Response, File, UploadFile, Form, Request
import re
from pydub import AudioSegment
import io
from io import BytesIO
import soundfile as sf
from torchaudio import transforms


ark_mode = os.getenv("ark")
app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# u = get_s3_client()

from io import BytesIO
from pydub import AudioSegment
import ffmpeg
import io


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def convert_wav_bytes_to_opus_bytes(
    wav_bytes, sample_rate=48000, bitrate="256k", volume=0
):
    """
    将字节类型的WAV数据转换为字节类型的MP3数据
    :param wav_bytes: WAV音频的字节数据
    :return: MP3音频的字节数据
    """
    # log.info(f"convert_wav_bytes_to_opus_bytes")
    # 从WAV字节数据创建AudioSegment对象
    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    audio = audio + volume
    # 重新采样至指定的采样率（如果需要）
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    # 创建一个BytesIO对象用于保存MP3数据
    mp3_buffer = io.BytesIO()
    # 导出到BytesIO对象
    # audio.export(mp3_buffer, format="opus", bitrate=bitrate)
    audio.export(
        mp3_buffer,
        format="opus",
        bitrate=bitrate,
        parameters=[
            "-strict",
            "-2",
            "-page_duration",
            "20",  # 分页时长
            # "-frame_duration", "20",         # Opus 帧时长
            # "-application", "audio",         # 优化模式（voip/audio）
            # "-vbr", "on",                    # 启用可变比特率
            # "-compression_level", "10"       # 最高压缩质量
        ],
    )
    # 获取MP3的字节数据并重置缓冲区的位置以便重新读取
    mp3_bytes = mp3_buffer.getvalue()
    mp3_buffer.seek(0)
    return mp3_bytes


def convert_wav_bytes_to_mp3_bytes(
    wav_bytes, sample_rate=16000, bitrate="32k", volume=0
):
    """
    将字节类型的WAV数据转换为字节类型的MP3数据
    :param wav_bytes: WAV音频的字节数据
    :return: MP3音频的字节数据
    """
    # 从WAV字节数据创建AudioSegment对象
    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    audio = audio + volume
    # 重新采样至指定的采样率（如果需要）
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    # 创建一个BytesIO对象用于保存MP3数据
    mp3_buffer = io.BytesIO()
    # 导出到BytesIO对象
    audio.export(mp3_buffer, format="mp3", bitrate=bitrate)
    # 获取MP3的字节数据并重置缓冲区的位置以便重新读取
    mp3_bytes = mp3_buffer.getvalue()
    mp3_buffer.seek(0)
    return mp3_bytes


def speed_change(input_audio: np.ndarray, speed: float, sr: int):

    # 将 NumPy 数组转换为原始 PCM 流
    raw_audio = input_audio.astype(np.int16).tobytes()
    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)
    # 变速处理
    output_stream = input_stream.filter('atempo', speed)
    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )
    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)
    return processed_audio




def convert_wav_bytes_to_audioSegment(wav_bytes):
    # 从WAV字节数据创建AudioSegment对象
    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    return audio


def get_db(audio_tensor, sr):
    wav_data = audio_tensor.squeeze().cpu().numpy()
    wav_data = pack_audio(BytesIO(), wav_data, sr, "wav").getvalue()
    sound = convert_wav_bytes_to_audioSegment(wav_data)
    db = sound.dBFS
    return db


def audio_postprocess(
    audio: torch.Tensor,
    sr: int = 32000,
    batch_index_list: list = None,
    speed_factor: float = 1.0,
    split_bucket: bool = True,
    fragment_interval: float = 0.16,
    spk: str = "quanzhou_sd",
    prompt_db: float = -31.3,
):
    zero_wav = torch.zeros(
        int(sr * fragment_interval), dtype=torch.float32, device="cpu"
    )

    current_db = get_db(audio, sr)
    vol = transforms.Vol(gain=prompt_db - current_db, gain_type="db")
    audio = vol(audio.squeeze().unsqueeze(0)).squeeze()
    max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
    if max_audio > 1:
        audio /= max_audio
    audio: torch.Tensor = torch.cat([audio, zero_wav], dim=0)

    audio = audio.cpu().numpy()
    audio = (audio * 32768).astype(np.int16)

    try:
        if speed_factor != 1.0:
            audio = speed_change(audio, speed=speed_factor, sr=int(sr))
    except Exception as e:
        print(f"Failed to change speed of audio: \n{e}")

    return sr, audio


def tts_replace(text, ark_mode="prod", data_type="malay_tts"):
    header = {"Token": "ae15bfd45069400c86d8f5571c521728"}

    if ark_mode == "prod":
        # host = "http://aiplatform-gn-uat.inneryiche.com/arges/taskflow/infer"  # 预发环境
        host = "http://10.0.55.148:50008/nlp/tts_replace_malay"  # 生产环境
    else:
        host = "http://10.0.55.148:50008/nlp/tts_replace_malay"  # 预发环境

    input_data = {
        "appId": "fbdd0248-ad1b-4041-a777-95960c0264ml",
        "userId": "IN0001",
        "data": {
            "info": {
                "org_txt": text,
                "type": data_type,
            }
        },
    }
    result = text
    try:
        response = requests.post(host, json=input_data, headers=header).json()
        if response["code"] == 0:
            result = response["data"]["info_res"]["result"]
    except Exception as e:
        print("error")

    return result


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if (
            len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8"))
            <= max_chars
        ):
            current_chunk += (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def prepare_request(
    target_text,
    speaker: str = None,
):

    data = {
        "inputs": [
            {
                "name": "target_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [target_text],
            },
            {
                "name": "spk_name",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [speaker],
            },
        ]
    }

    return data


@app.post("/call_mx_tts")
async def inference_tts(request: Request):

    try:
        t0 = time.time()
        req = await request.body()
        input_data = json.loads(req)["data"]["Input"]
        req = json.loads(req.decode("utf-8"))
        # userRequestId = req.get("userRequestId", "test")
        # log.info(f"req:{req}")
        # orig_text = input_data.get("orig_text","")
        text = input_data.get("text", "")
        spk = input_data.get("speaker_name", "elven_1")
        speed_factor = input_data.get("speed_factor", 1.0)
        volume = input_data.get("volume", 0)
        audio_type = input_data.get("audio_type", "mp3")
        if str(audio_type) not in ["mp3", "opus"]:
            audio_type = "mp3"

        if speed_factor < 0.5 or speed_factor > 2.0:
            speed_factor = 1.0
        text = text.replace("！", ".").replace("!", ".")

        # log.info(f"Generating speech for {spk}")

        # audio_bytes,total_duration = text_to_speech(text=text,spk=spk,speed_factor=speed_factor,volume=volume,audio_type=audio_type)
        request_data = prepare_request(text, spk)
        res = requests.post(
            url="http://localhost:18000/v2/models/cosyvoice3/infer",
            json=request_data,
        ).json()
        # 解码音频数据
        audio = res["outputs"][0]["data"]
        wav_data = torch.from_numpy(np.array(audio, dtype=np.float32)).unsqueeze(0)
        speech_len = wav_data.shape[1] / 24000
        _, wav_data = audio_postprocess(
            audio=wav_data.squeeze(),
            sr=24000,
            speed_factor=speed_factor,
            spk=spk,
            prompt_db=-31,
        )
        wav_data = pack_audio(BytesIO(), wav_data, 24000, "wav").getvalue()
        if audio_type == "opus":
            wav = convert_wav_bytes_to_opus_bytes(wav_data, volume=volume)
        else:
            wav = convert_wav_bytes_to_mp3_bytes(wav_data, sample_rate=24000 , volume=volume)
        wav_base64 = base64.b64encode(wav).decode("utf-8")

        cost_time = str(time.time() - t0)
        # log.info("cost time : " + cost_time)
        rtf = float(cost_time) / speech_len
        # log.info(f"rtf: {rtf}")
        req["data"]["Input"]["cost_time"] = cost_time
        req["data"]["Input"]["rtf"] = rtf
        req["data"]["Input"]["text"] = text
        # log.info(f"final req: {req}")
        response = {"code": 0, "msg": "OK", "data": {"Audio": wav_base64}}
        return response

    except Exception as e:
        torch.cuda.empty_cache()
        # log.exception(traceback.format_exc())
        return {
            # "traceId": "7788ed19-8e6d-423d-b4ed-7a23123d3c2f",
            # "code": 1500,
            # "message": e,
            "code": 1,
            "msg": str(e),
            "data": {
                "Audio": b"",
            },
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=50006)
