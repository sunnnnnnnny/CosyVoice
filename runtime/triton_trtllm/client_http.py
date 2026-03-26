# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import requests
import soundfile as sf
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:18000",
        help="Address of the server",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="¡Llamando a todos los papás perrunos y gatunos! Si ustedes son como yo, que adonde van, llevan a su mascota, esta camioneta les va a solucionar la vida. Los asientos traseros se abaten completamente planos, dejando un espacio gigante atrás. Le pones su cobijita, su transportadora y hace que tu perrito viaje como rey. Además, el material de la cajuela es súper resistente a rasguños y fácil de aspirar para quitar los pelitos. También trae salidas de aire acondicionado en la parte trasera para que Firulais no vaya pasando calor. Mencionen aquí en los comentarios cómo se llama su mascota, ¡los leo a todos!",
        help="",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="cosyvoice3",
        choices=[
            "f5_tts",
            "cosyvoice3",
            "spark_tts",
            "cosyvoice2"],
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--output-audio",
        type=str,
        default="output.wav",
        help="Path to save the output audio",
    )
    return parser.parse_args()


def prepare_request(
    target_text,
):    
    spk_name = "Aitana"

    data = {
        "inputs": [
            {
                "name": "spk_name",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [spk_name]
            },
            {
                "name": "target_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [target_text]
            }
        ]
    }

    return data


if __name__ == "__main__":
    args = get_args()
    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"
    url = f"{server_url}/v2/models/{args.model_name}/infer"
    
    data = prepare_request(args.target_text)

    rsp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=data,
        verify=False,
        params={"request_id": '0'}
    )
    result = rsp.json()
    audio = result["outputs"][0]["data"]
    audio = np.array(audio, dtype=np.float32)
    if args.model_name == "spark_tts":
        sample_rate = 16000
    else:
        sample_rate = 24000
    sf.write(args.output_audio, audio, sample_rate, "PCM_16")
