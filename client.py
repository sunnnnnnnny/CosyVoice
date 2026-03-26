import requests
import json
import base64
import re
import os
from tqdm import tqdm


from datetime import datetime
import random
import string
import random
random.seed(616)


def generate_request_id_v2():
    now = datetime.now()
    timestamp_part = now.strftime("%Y%m%d%H%M%S%f")[:-1]
    random_part = "".join(random.choices(string.ascii_letters + string.digits, k=13))
    return f"{timestamp_part}{random_part}"
appid = "2026841957434658816"
env = "prod"
mode = f"_nc_day0326_{appid}_debug1"
out_dir = f"results_{env}" + mode
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# env = "test"
def call_tts(orig_text: str, text: str, speaker: str, file_name: str, speed:int, volume:int):
    # if env == "test":
    #     token_key = "v8znd4bj64awm90pk410l679ya5rj8ev"
    #     url = "https://vesta.yxqiche.com/vesta_test/v2/engine_online/flow_schedule/custom_inference"  # 测试环境
    #     aws_url = "https://vesta-oversea.yxqiche.com/vesta_test/v2/engine_online/flow_schedule/custom_inference"  # 测试环境
    # else:
    #     token_key = "w0p795gnp9htvfp0l7ntbaty5jzk2yr4"
    #     url = "https://vesta2.yxqiche.com/vesta/v2/engine_online/flow_schedule/custom_inference"  # 正式环境
    #     aws_url = "https://vesta-oversea.yxqiche.com/vesta/v2/engine_online/flow_schedule/custom_inference"  # 正式环境
    token_key = "v8znd4bj64awm90pk410l679ya5rj8ev"
    url = "http://0.0.0.0:50006/call_mx_tts"
    requests_id = generate_request_id_v2()
    print(f"requests_id: {requests_id}")
    headers = {
        "TenantId": "YIXIN",
        "TokenKey": token_key,
        "AccountId": "X-Project",
        "AccountToken": "93ab003fc4cda6247ff282add14de1ed",
    }
    input_data = {
        "appId": appid,
        "userId": "11",
        "version": "mexico_tts_v1",
        "requestId": requests_id,
        "data": {
            "Input": {
                "orig_text": orig_text,
                "text": text,
                "text_lang": "mx",
                "volume": volume,
                "speed_factor": speed,
                "speaker_name": speaker,
                "audio_type": "mp3",
            }
        },
    }
    response = requests.post(url, json=input_data, headers=headers).json()
    # import ipdb
    # ipdb.set_trace()
    print(response.keys())
    if response["code"] != 0:
        print("error")
        return
    # else:
    print("call success")
    # print(response)

    output = response["data"]["Audio"]
    # print(f'response["data"]: {response["data"]}')
    # print(type(output['audio']))
    # print('response',response)
    print("---------------------")
    audio_data = base64.b64decode(output)  # base64.b64decode(audio))
    with open(file_name, "wb") as f:
        f.write(audio_data)
    return output  # ,output['subtitles']


spk_list = ['Aitana', 'Alejandro', 'Benito', 'CDMX', 'Carlos', 'Cazzu', 'Danna', 'Florencia', 'Georgina', 'Inicianlasclips', 'Joaqui', 'Kass', 'LosYoutubersclips', 'Majo', 'Maria', 'Nicoleclips', 'Paloma', 'PreciodelOloniaclips', 'Rauw', 'Renata', 'Speitzer', 'Tini', 'Top10clips', 'Valentina', 'ali', 'alyn', 'anaymiel', 'andybadilloo', 'arath', 'autos_jlbl', 'autosexoticosmexico', 'christiansotelo', 'clip-1773047275508', 'corteswilver', 'dingler_adrian', 'ellieoficial', 'hector', 'heyaaamir', 'ittai', 'mannysinfronteras', 'mariaacastaned', 'nataliapaulin', 'sacelady', 'salvadorfiscal', 'sanangelusadoscertificad', 'soytuastro', 'vale', 'vivemexico', 'vladk', 'ximear444', 'zasha_ylacholada']
text = "¿Buscas un coche que lo tenga todo. El nuevo MG HS no es cualquier vehículo, es tu pase a tecnología de punta comodidad total y cero preocupaciones. Desde su sistema inteligente que te hace la vida fácil hasta su garantía que te da tranquilidad. Este es el auto que no sabías que necesitabas. Y lo mejor cada kilómetro será una aventura porque tú estrenas. Tú mandas. ¿Te imaginas manejándolo ya. Descúbrelo hoy y cambia la forma en que ves el camino."
text_list = text.split(".")
text_list = [text]
spk = "Aitana"
speed = 1.0
volume = 0
for idx,text in enumerate(text_list):
    call_tts(
        orig_text=text,
        text=text,
        file_name=os.path.join(out_dir, f"res_0326_text{idx+1}_spk{spk}_speed{speed}_volume{volume}.mp3"),
        speaker=spk,
        speed=speed,
        volume=volume)
