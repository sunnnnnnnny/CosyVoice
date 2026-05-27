import json
import re
import time
import asyncio
import os
import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils
import soundfile as sf
import httpx
import torchaudio
from functools import partial
from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram


torch.set_num_threads(1)

# CosyVoice3 mel params: fmax=None (Nyquist), center=False
mel_spectrogram = partial(matcha_mel_spectrogram,
    n_fft=1920, num_mels=80, sampling_rate=24000,
    hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)


def parse_speech_token_string(response_text):
    """Parse speech tokens from string like '<|s_123|><|s_456|>' into list of int IDs."""
    speech_tokens = response_text.strip().split('><')
    if len(speech_tokens) > 1:
        speech_tokens = ['<' + t if not t.startswith('<') else t for t in speech_tokens]
        speech_tokens = [t + '>' if not t.endswith('>') else t for t in speech_tokens]
    speech_ids = []
    for token_str in speech_tokens:
        match = re.match(r'<\|s_(\d+)\|>', token_str)
        if match:
            speech_ids.append(int(match.group(1)))
    return speech_ids


class TritonPythonModel:
    """CosyVoice3 BLS orchestrator for Triton Inference Server.

    Orchestrates: audio_tokenizer, speaker_embedding, remote LLM (httpx),
    token2wav (flow-only), and vocoder (CausalHiFTGenerator).
    Supports both streaming (decoupled) and offline (non-decoupled) modes.
    """

    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])
        parameters = self.model_config['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}

        self.device = torch.device("cuda")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        # Streaming config
        self.token_frame_rate = 25
        self.flow_pre_lookahead_len = 3
        self.token_hop_len = 15
        self.token_mel_ratio = 2
        self.dynamic_chunk_strategy = model_params.get("dynamic_chunk_strategy", "exponential")
        self.logger.log_info(f"CosyVoice3 BLS initialized, decoupled={self.decoupled}, "
                             f"chunk_strategy={self.dynamic_chunk_strategy}")

        # HTTP client for remote LLM (trtllm-serve default port: 8000)
        self.http_client = httpx.AsyncClient()
        self.api_base = model_params.get("llm_api_base", "http://localhost:8000/v1/chat/completions")

        # Speaker cache to avoid redundant audio_tokenizer/speaker_embedding calls
        self.speaker_cache = {}
        self.filename2spk = {
            "ALI_denoise3.wav_0000000000_0000119360.wav": "ali",
            "ALYN_denoise81.wav_0000356800_0000467520.wav": "alyn",
            "HECTOR_denoise37.wav_0002576000_0002716480.wav": "hector",
            "anaymiel_denoise7.wav_0001471360_0001577600.wav": "anaymiel",
            "andybadilloo_denoise166.wav_0003022720_0003136320.wav": "andybadilloo",
            "arath_rosstt_denoise7.wav_0001145280_0001294720.wav": "arath",
            "autos.jlbl_denoise366.wav_0000351680_0000441280.wav": "autos_jlbl",
            "autosexoticosmexico_denoise86.wav_0001826240_0001968000.wav": "autosexoticosmexico",
            "christiansotelo42_denoise_4_299.wav": "christiansotelo",
            "corteswilver_denoise_3_141.wav": "corteswilver",
            "dingler_adrian_denoise3.wav_0000000000_0000132800.wav": "dingler_adrian",
            "ellieoficial_denoise129.wav_0000804160_0000990720.wav": "ellieoficial",
            "heyaaamir_denoise121.wav_0001576960_0001723520.wav": "heyaaamir",
            "ittai_mexico_denoise_1_064.wav": "ittai",
            "mannysinfronteras_denoise82.wav_0004661120_0004808000.wav": "mannysinfronteras",
            "mariaacastaneda_denoise125.wav_0002526400_0002692160.wav": "mariaacastaned",
            "nataliapaulin.s_denoise36.wav_0003391040_0003523520.wav": "nataliapaulin",
            "sacelady.mexico_denoise_4_278.wav": "sacelady",
            "salvadorfiscal_denoise_3_1025.wav": "salvadorfiscal",
            "sanangelusadoscertificad_denoise30.wav_0000399360_0000514560.wav": "sanangelusadoscertificad",
            "soytuastro_denoise2.wav_0000536320_0000690560.wav": "soytuastro",
            "VALE_denoise2.wav_0000022080_0000097600.wav": "vale",
            "vivemexico_4_denoise_1_014.wav": "vivemexico",
            "vladk.ruso_denoise72.wav_0001386880_0001523520.wav": "vladk",
            "ximear444_denoise109.wav_0001784960_0001920640.wav": "ximear444",
            "zasha_ylacholada_denoise172.wav_0001047040_0001192960.wav": "zasha_ylacholada",
            "Aitana.mp3_vocals.wav": "Aitana",
            "CDMX.mp3_vocals.wav":"CDMX",
            "Danna.mp3_vocals.wav":"Danna",
            "Cazzu.mp3_vocals.wav":"Cazzu",
            "Georgina.mp3_vocals.wav":"Georgina",
            "Rauw.mp3_vocals.wav":"Rauw",
            "Valentina.mp3_vocals.wav":"Valentina",
            "Benito.mp3_vocals.wav":"Benito",
            "Majo.mp3_vocals.wav":"Majo",
            "Carlos.mp3_vocals.wav":"Carlos",
            "Tini.mp3_vocals.wav":"Tini",
            "LosYoutubersclips.mp3_vocals.wav":"LosYoutubersclips",
            "Kass.mp3_vocals.wav":"Kass",
            "Speitzer.mp3_vocals.wav":"Speitzer",
            "Florencia.mp3_vocals.wav":"Florencia",
            "Paloma.mp3_vocals.wav":"Paloma",
            "clip-1773047275508.mp3_vocals.wav":"clip-1773047275508",
            "Inicianlasclips.mp3_vocals.wav":"Inicianlasclips",
            "Nicoleclips.mp3_vocals.wav":"Nicoleclips",
            "Renata.mp3_vocals.wav":"Renata",
            "Top10clips.mp3_vocals.wav":"Top10clips",
            "Joaqui.mp3_vocals.wav":"Joaqui",
            "Alejandro.mp3_vocals.wav":"Alejandro",
            "Maria.mp3_vocals.wav":"Maria",
            "PreciodelOloniaclips.mp3_vocals.wav":"PreciodelOloniaclips",
            "14-prompt-male_vocals_16k.wav": "14-prompt-male_vocals_16k",
            "11-prompt-female_vocals_16k.wav": "11-prompt-female_vocals_16k",
            "27-prompt-female_vocals_16k-1.wav": "27-prompt-female_vocals_16k-1",
            "24-prompt-male_vocals_16k-1.wav": "24-prompt-male_vocals_16k-1",
            "10-prompt-female_vocals_16k.wav": "10-prompt-female_vocals_16k",
            "19-prompt-female_vocals_16k.wav": "19-prompt-female_vocals_16k",
            "39-prompt-female.wav_vocals_16k.wav": "39-prompt-female.wav_vocals_16k",
            "46-prompt-female.wav_vocals_16k.wav": "46-prompt-female.wav_vocals_16k",
            "9-prompt-female_vocals_16k.wav": "9-prompt-female_vocals_16k",
            "35-prompt-male_vocals_16k.wav": "35-prompt-male_vocals_16k",
            "1-prompt_vocals_16k.wav": "1-prompt_vocals_16k",
            "40-prompt-female.wav_vocals_16k.wav": "40-prompt-female.wav_vocals_16k",
            "16-prompt-female_vocals_16k.wav": "16-prompt-female_vocals_16k",
            "female_200_16k.wav": "female_200_16k",
            "5-prompt_vocals_16k-1.wav": "5-prompt_vocals_16k-1",
            "23-prompt-male_vocals_16k.wav": "23-prompt-male_vocals_16k",
            "25-prompt-male_vocals_16k.wav": "25-prompt-male_vocals_16k",
            "32-prompt-male_vocals_16k.wav": "32-prompt-male_vocals_16k",
            "Josh_Szeps_spk-4-male-11_16k.wav": "Josh_Szeps_spk-4-male-11_16k",
            "7-prompt-female_vocals_16k.wav": "7-prompt-female_vocals_16k",
            "42-prompt-female.wav_vocals_16k-1.wav": "42-prompt-female.wav_vocals_16k-1",
            "8-prompt-male_vocals_16k.wav": "8-prompt-male_vocals_16k",
            "New_Episodes-94-female-86_16k.wav": "New_Episodes-94-female-86_16k",
            "3-prompt_vocals_16k.wav": "3-prompt_vocals_16k",
            "5-prompt_vocals_16k.wav": "5-prompt_vocals_16k",
            "12-prompt-male.wav_vocals_16k.wav": "12-prompt-male.wav_vocals_16k",
            "The_Vulnerabilitea_House_spk-48-female-25_16k.wav": "The_Vulnerabilitea_House_spk-48-female-25_16k",
            "37-prompt-female.wav_vocals_16k.wav": "37-prompt-female.wav_vocals_16k",
            "15-prompt-female_vocals_16k.wav": "15-prompt-female_vocals_16k",
            "25-prompt-male_vocals_16k-1.wav": "25-prompt-male_vocals_16k-1",
            "29-prompt-male_vocals_16k.wav": "29-prompt-male_vocals_16k",
            "26-prompt-male_vocals_16k-1.wav": "26-prompt-male_vocals_16k-1",
            "4-prompt_vocals_16k.wav": "4-prompt_vocals_16k",
            "28-prompt-male_vocals_16k-1.wav": "28-prompt-male_vocals_16k-1",
            "2-prompt_vocals_16k.wav": "2-prompt_vocals_16k",
            "6-prompt_vocals_16k.wav": "6-prompt_vocals_16k",
            "43-prompt-male.wav_vocals_16k.wav": "43-prompt-male.wav_vocals_16k",
            "31-prompt-male_vocals_16k.wav": "31-prompt-male_vocals_16k",
            "36-prompt-female_vocals_16k.wav": "36-prompt-female_vocals_16k",
            "27-prompt-female_vocals_16k.wav": "27-prompt-female_vocals_16k",
            "38-prompt-female.wav_vocals_16k.wav": "38-prompt-female.wav_vocals_16k",
            "18-prompt-male_vocals_16k-1.wav": "18-prompt-male_vocals_16k-1",
            "28-prompt-male_vocals_16k.wav": "28-prompt-male_vocals_16k",
            "Josh_Szeps_spk-8-male-94_16k.wav": "Josh_Szeps_spk-8-male-94_16k",
            "Josh_Szeps_spk-2-male-94_16k.wav": "Josh_Szeps_spk-2-male-94_16k",
            "30-prompt-male_vocals_16k.wav": "30-prompt-male_vocals_16k",
            "34-prompt-male_vocals_16k.wav": "34-prompt-male_vocals_16k",
            "13-prompt-male_vocals_16k.wav": "13-prompt-male_vocals_16k",
            "22-prompt-male_vocals_16k.wav": "22-prompt-male_vocals_16k",
            "11-prompt-female_vocals_16k-1.wav": "11-prompt-female_vocals_16k-1",
            "45-prompt-female_vocals_16k.wav": "45-prompt-female_vocals_16k",
            "male_215_16k.wav": "male_215_16k",
            "New_Episodes-17-male-65_16k.wav": "New_Episodes-17-male-65_16k",
            "39-prompt-female.wav_vocals_16k-1.wav": "39-prompt-female.wav_vocals_16k-1",
            "The_Vulnerabilitea_House_spk-26-female-69_16k.wav": "The_Vulnerabilitea_House_spk-26-female-69_16k",
            "20-prompt-female_vocals_16k.wav": "20-prompt-female_vocals_16k",
            "42-prompt-female.wav_vocals_16k.wav": "42-prompt-female.wav_vocals_16k",
            "41-prompt-female.wav_vocals_16k.wav": "41-prompt-female.wav_vocals_16k",
            "21-prompt-male_vocals_16k.wav": "21-prompt-male_vocals_16k",
            "17-prompt-female_vocals_16k.wav": "17-prompt-female_vocals_16k",
            "33-prompt-female_vocals_16k-1.wav": "33-prompt-female_vocals_16k-1",
            "The_Vulnerabilitea_House_spk-60-female-99_16k.wav": "The_Vulnerabilitea_House_spk-60-female-99_16k",
            "44-prompt-female.wav_vocals_16k.wav": "44-prompt-female.wav_vocals_16k",}
        self.spk2filename = {}
        for k, v in self.filename2spk.items():
            self.spk2filename[v] = k
        self.spk2text = {
            "ali":"Miren este Nissan Z9 que estamos viendo en el diesel.",
            "alyn":"En Emstar, te mostramos que no tiene por qué ser así.",
            "Aitana":"Muchísimas gracias. De verdad que es la primera vez que hago este videoconvo, de verdad, así que.",
            "Alejandro":"Soy Alejandro Spitzer. Estoy en Milán con Dolce & Gabbana y como soy foodie.",
            "Benito":"Hoja por hoja estaban repletos de zapatillas, de moños y vestidos.",
            "CDMX":"muchas gracias te platico un poco de de qué va a tratar la experiencia la parte de la comida de Arco",
            "Carlos":"El cargador trae dos conectores, el de México o el de la parte, digamos más.",
            "Cazzu":"Bueno, tengo una cartera enorme. Yo amaba las carteras chiquititas. Yo me quejaba un montón de mi mamá que llevaba como una cartera enorme.",
            "Danna":"yo siempre quise ser una estrella yo veía las las",
            "Florencia":"de poder arrancar y bueno de tener una segunda temporada de bogón",
            "Georgina":"varias cosas de mi bolso os pueden sorprender, pero la verdad lo que más sorprende",
            "hector":"Por cierto, el espacio en las plazas traseras es ligeramente mejor que en el Swift.",
            "Inicianlasclips":"Y esta semana comenzó con sus conferencias mañaneras, que afirmó serán tres veces a la semana.",
            "Joaqui":"algo que hago todas las mañanas, sinceramente tengo que ser honesta, no sé si funciona o no, yo lo vi en TikTok, estaba trendy y yo siento personalmente que me hace ver",
            "Kass":"refleja finura y yo de que soy desordenada este y que también tengo como mucho apego a cositas",
            "LosYoutubersclips":"Estos son los top 10 youtubers con más suscriptores de todo México. Comenta si sabes quién es el top uno.",
            "Majo":"pero yo siempre tuve mucho amor. Si no hubiera sido así, quizá la historia sería muy diferente.",
            "Maria":"Okay, ahora sí ya estamos en el coche y ya estamos en camino al evento.",
            "Nicoleclips":"una vez cogí un un gloss de mi madre le pinté las patas de la mesa con gloss porque quedaban",
            "Paloma":"ya me lo puse entonces voy a probar también ponerme esto que como un primer",
            "PreciodelOloniaclips":"El o línea, el auto eléctrico mexicano a bajo costo.",
            "Rauw":"no es la flecha es el indio es el piquete es el sazón el flow que tú le metas a las cosas y te puedes poner lo más sencillo que tengas pero si lo proyectas con seguridad",
            "Renata":"Crear un look o encontrar un look para can va más allá de solamente encontrar un vestido.",
            "Speitzer":"Soy Alejandro Spitzer. Estoy en Milán con Dolce & Gabbana y como soy foodie.",
            "Tini":"bueno llegamos a México hace ya unos días estuvimos haciendo promo para",
            "Top10clips":"Fue una experiencia muy bonita, muy interesante y amé las reacciones de Yalitza.",
            "vale":"estamos hablando del nuevo.",
            "Valentina":"dije qué perfume usaba porque siento que es como algo super personal vamos a darle la muestra hola soy Valentina Cenere estoy acá con Vogue y les voy a enseñar qué hay en mi bolsa",
            "anaymiel":"creo que pueden mejorar sígueme para no perderte lugares increíbles",
            "andybadilloo":"y vamos a poner estas dos sombras en crema de Carolina Herrera.",
            "arath":"Yo nunca dejó de hacer pruebas y medir los datos, resultados.",
            "autos_jlbl":"Con cada nueva línea que sale de esta camioneta.",
            "autosexoticosmexico":"el nombre de este carro que es F5 es derivado a la escala fujita.",
            "christiansotelo":"para poder platicar acerca de su proceso de compra.",
            "clip-1773047275508":"soy como un poquito obsesionada con los productos para la piel, empezando mi rutina del skincare primero.",
            "corteswilver":"y voy pasando yo dije a la aquí me van a dejar como coladera",
            "dingler_adrian":"De todo Costa Rica, ¿cuál es el marchamo más barato que puedes pagar en es?",
            "ellieoficial":"lo que para mí es un impacto demográfico y cultural.",
            "heyaaamir":"La única manera de poderlo hacer nacional va a ser por decreto.",
            "ittai":"entonces no me podría parar y enseñarles todo lo demás ya lo van a ver en el video",
            "mannysinfronteras":"Pero si aprendemos a depender de lo nuestro.",
            "mariaacastaned":"Más de lo que te imaginas. Es que es super fresco, delgadito, cómodo.",
            "nataliapaulin":"Y darme o no la razón, los invito a que me digan en los comentarios ustedes qué opinan.",
            "sacelady":"la vamos a activar para ustedes la vamos a activar para ustedes hermosas",
            "salvadorfiscal":"A ver Fernando, si llevo cinco meses recibo aguinaldo, sí.",
            "sanangelusadoscertificad":"En el interior contamos con cluster de instrumentos digital.",
            "soytuastro":"es un grupo donde se le incluye siempre las personas con muchos planetas en la casa once buena",
            "vivemexico":"Rosita, buenas tardes, Rosita, buenos días, buen día Dani, Alejandra.",
            "vladk":"Que sea negocio, chicos, hay autos como estos que les voy a decir una realidad.",
            "ximear444":"pero cuando eres europeo o gringo todo el mundo te lo te lo festeja",
            "zasha_ylacholada":"que recientemente sacó una rola con Santa Fe Clan y días después fue que pasó",
            "14-prompt-male_vocals_16k": "And unlike a lot of specs where you can just go, give me a Japanese wheel at 17 by 8 plus 35, not going to happen in this case.",
            "11-prompt-female_vocals_16k": "There is a lot of competition right now in Australia, particularly in this mid-sized E B category.",
            "27-prompt-female_vocals_16k-1": "but there is a big mess to clean up this morning.",
            "24-prompt-male_vocals_16k-1": "Doesn't matter what cut really, just ideally go budget conscious because the cheaper the cut of meat.",
            "10-prompt-female_vocals_16k": "It's more than a car, rather it represents the future of the company.",
            "19-prompt-female_vocals_16k": "Hey guys, come test out loads of EVs and see the latest clean energy technologies at Everything Electric LIVE at the Sydney Showgrounds the seventh, eighth and ninth of March.",
            "39-prompt-female.wav_vocals_16k": "Looking at the thongs, his mates came in first, just wandering around the shop.",
            "46-prompt-female.wav_vocals_16k": "It's so weird because she remembers the details for dances that we like did two years ago.",
            "9-prompt-female_vocals_16k": "fuel efficient than the 3 litre models.",
            "35-prompt-male_vocals_16k": "when people have to think about how am I behaving when there is no police around? What am I doing to be a better driver? What am I doing to try and enhance my own capability?",
            "1-prompt_vocals_16k": "Okay. So this is the easiest test of them all. This is basically a flex test.",
            "40-prompt-female.wav_vocals_16k": "I'm definitely there to kind of calm any tensions. It's really good to have someone that is unbiased.",
            "16-prompt-female_vocals_16k": "Side profile, straight beltline, crisp creases, a classy chrome D-pillar and giant 21-inch calligraphy alloys.",
            "female_200_16k": "the levels of benzene just aren't going to cause any sort of offness about your health, this recall was not because people actually had issues.",
            "5-prompt_vocals_16k-1": "and it's yet another way the Tesla Model Y shifts everything forward in the way that we're thinking of cars.",
            "23-prompt-male_vocals_16k": "we were driving on this exact same road, except it was blanketed in snow.",
            "25-prompt-male_vocals_16k": "Did that for about three years. Everyone's like, 'Dude, you got to do an apprenticeship.' So I was like, 'Screw it.'",
            "32-prompt-male_vocals_16k": "It starts with a bang as well. You've got a four-word, here's the book here.",
            "Josh_Szeps_spk-4-male-11_16k": "The only one commerce platform to start, run, and grow your business. I know that building a business takes work.",
            "7-prompt-female_vocals_16k": "So the Palisade took out our seven-seat SUV under 90k category at the 2026 Drive.",
            "42-prompt-female.wav_vocals_16k-1": "It's not even that bad, but I'll give you a few tips.",
            "8-prompt-male_vocals_16k": "let me tell you, this thing is absolutely fantastic. And I've got the best track in Australia to test it around, Mount Panorama.",
            "New_Episodes-94-female-86_16k": "This speaks totally to my sort of thoughts of control as well. It's like, cool, if I keep on going this path.",
            "3-prompt_vocals_16k": "We've got new reviews of all the cars you need to know about landing every week. But don't just take our word for it, hit that subscribe button and watch this.",
            "5-prompt_vocals_16k": "this is a consideration for modern use for modern families.",
            "12-prompt-male.wav_vocals_16k": "okay, but not exceptional,  i have got.",
            "The_Vulnerabilitea_House_spk-48-female-25_16k": "Doing certain things but willing to and willing to repeatedly. That's where you sort of just learn to actually.",
            "37-prompt-female.wav_vocals_16k": "And our community has really been, you know, campaigning for about nine or ten months to get him out here to the biggest and best Ipswich.",
            "15-prompt-female_vocals_16k": "first-time buyer to the car enthusiast, our short and sharp reviews answer all your questions.",
            "25-prompt-male_vocals_16k-1": "Get it out of the way. I just hate my life for four years.",
            "29-prompt-male_vocals_16k": "Pick up the phone and you would ask to be connected to a particular place or numbers.",
            "26-prompt-male_vocals_16k-1": "The craziest thing would happen for us. Like we're tradies and we're going to Sydney to have meetings with the most important people.",
            "4-prompt_vocals_16k": "you let our knowledge, experience and detail testing inform you.",
            "28-prompt-male_vocals_16k-1": "But mate, all I had was me jocks on. I was chasing him up the street, and I'm just like.",
            "2-prompt_vocals_16k": "So there you go, the B M W M 8 Competition. That is a beast of a car and a handful on the track.",
            "6-prompt_vocals_16k": "On the inside, this Ranger plug-in hybrid is the same in most respects. A lot of the things that we know and love about the regular Ranger, it's all here.",
            "43-prompt-male.wav_vocals_16k": "This is why Australia is the best country to live in in 2025. Firstly, we have a different sun here. Our sun's twice as hot, which allows us to cook our food outside.",
            "31-prompt-male_vocals_16k": "And to write about the sport and the intensity of being in the paddock, you know, not just from the drivers who are under maximum pressure in the cars.",
            "36-prompt-female_vocals_16k": "drivers in particular had the most to gain and improved their scores rapidly.",
            "27-prompt-female_vocals_16k": "Daniel, quite the Australian hero here this morning. As the owners of the fish chip shop.",
            "38-prompt-female.wav_vocals_16k": "The hunk of all hunks walked literally here in this Mullum news agency yesterday.",
            "18-prompt-male_vocals_16k-1": "So minimal carbon monoxide, minimal carbon dioxide, and there are a slew of other benefits as well.",
            "28-prompt-male_vocals_16k": "Flash Emmon set them off in the direction of him.",
            "Josh_Szeps_spk-8-male-94_16k": "They keep voting against their own interests by voting for people like Trump and voting for Republicans.",
            "Josh_Szeps_spk-2-male-94_16k": "She was lined up, this activist, that is, as a guest on my show the following day in a kind of wrap of the news and everything that's going on.",
            "30-prompt-male_vocals_16k": "Also it's pet friendly and it's budget friendly so it's really actually quite a good way to go.",
            "34-prompt-male_vocals_16k": "Fear not, because now there is an impartial way to figure out who really is Australia's safest driver.",
            "13-prompt-male_vocals_16k": "And he said that he doesn't like him.",
            "22-prompt-male_vocals_16k": "This is a Pulsar Turbo. And this one is a fifty-four fifty-five. The reason I've gone for this for.",
            "11-prompt-female_vocals_16k-1": "In fact, it's one of the most fiercely contested segments going.",
            "45-prompt-female_vocals_16k": "Yesterday we did a campaign for Celeron and it was very glowy and we really liked that. So we might do a wet hair look.",
            "male_215_16k": "In Australia, they've actually had like a skunk work P R operation for quite some time.",
            "New_Episodes-17-male-65_16k": "Four hundred people. This is a Marvel movie. This is a hundred fifteen million dollar Marvel movie, right? I don't know what I'm doing.",
            "39-prompt-female.wav_vocals_16k-1": "He came in. He's like, 'Can I try a pair of these thongs on?' I'm like, 'Sure.' So I opened them for him.",
            "The_Vulnerabilitea_House_spk-26-female-69_16k": "I feel like I'm at a two, but a therapist sees people.",
            "20-prompt-female_vocals_16k": "This carnival hybrid is perfect for the economically conscious family.",
            "42-prompt-female.wav_vocals_16k": "As you guys know, I'm Australian and I've had so many people come up to me and be like Bridie, I want to come visit Australia but I'm so scared of your dangerous animals.",
            "41-prompt-female.wav_vocals_16k": "Since it's our first single, we've practiced really hard and we won't disappoint you.",
            "21-prompt-male_vocals_16k": "you know, we're testing it as four-wheel drivers, you know, people wanted to do a lap, people want to do a bit of hard wheeling. They set their purpose before what we were doing.",
            "17-prompt-female_vocals_16k": "That's why I've been testing something called RadarBot and honestly, it surprises me.",
            "33-prompt-female_vocals_16k-1": "There's a lot of on-track drama. There's also a lot of off-track drama that we are not privy to.",
            "The_Vulnerabilitea_House_spk-60-female-99_16k": "Claire, who I've known since high school, she said to me, I don't reckon you could just pop to the, I don't reckon you're sleeping very well.",
            "44-prompt-female.wav_vocals_16k": "Just a casual reminder that if you remember doing this, Minesweeper is the best by the way.",}
        self.prompt_speech_dir = "/gpfs01/nfs_share/data20250106/yuqiangz/master_models/CosyVoice/en_au_16k"

    def _convert_speech_tokens_to_str(self, speech_tokens):
        """Convert speech token IDs tensor/list to string like '<|s_N|>'."""
        if isinstance(speech_tokens, torch.Tensor):
            speech_tokens = speech_tokens.cpu().numpy().flatten().tolist()
        return "".join(f"<|s_{int(tid)}|>" for tid in speech_tokens)

    def _extract_speech_feat(self, speech):
        """Extract mel spectrogram from 24kHz speech for flow prompt."""
        speech_feat = mel_spectrogram(speech).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.unsqueeze(dim=0).to(self.device)
        return speech_feat

    async def forward_llm_streaming(self, target_text, reference_text, prompt_speech_tokens):
        """Async generator: stream LLM tokens via httpx SSE."""
        full_text = f"{reference_text}{target_text}"
        prompt_speech_tokens_str = self._convert_speech_tokens_to_str(prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            "max_tokens": 750,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": True,
        }

        buffer = ""
        async with self.http_client.stream("POST", self.api_base, json=payload, timeout=None) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    line_data = line[len("data: "):].strip()
                    if line_data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(line_data)
                        content = json_data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            buffer += content
                            while True:
                                match = re.search(r"<\|s_(\d+)\|>", buffer)
                                if not match:
                                    break
                                token_num = int(match.group(1))
                                # final_id = token_num + ORIGINAL_VOCAB_SIZE
                                yield token_num
                                buffer = buffer[match.end():]
                    except json.JSONDecodeError:
                        continue

        # Flush remaining tokens
        while True:
            match = re.search(r"<\|s_(\d+)\|>", buffer)
            if not match:
                break
            token_num = int(match.group(1))
            #final_id = token_num + ORIGINAL_VOCAB_SIZE
            yield token_num
            buffer = buffer[match.end():]

    async def forward_llm_offline(self, target_text, reference_text, prompt_speech_tokens):
        """Non-streaming LLM call, returns all speech token IDs at once."""
        full_text = f"{reference_text}{target_text}"
        prompt_speech_tokens_str = self._convert_speech_tokens_to_str(prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            "max_tokens": 750,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": False,
        }
        response = await self.http_client.post(self.api_base, json=payload, timeout=None)
        response.raise_for_status()
        response_json = response.json()
        generated_content = response_json['choices'][0]['message']['content']
        speech_ids = parse_speech_token_string(generated_content)
        # return [sid + ORIGINAL_VOCAB_SIZE for sid in speech_ids]
        return speech_ids

    def forward_audio_tokenizer(self, wav, wav_len):
        """BLS call to audio_tokenizer."""
        inference_request = pb_utils.InferenceRequest(
            model_name='audio_tokenizer',
            requested_output_names=['prompt_speech_tokens'],
            inputs=[wav, wav_len]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_speech_tokens = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_speech_tokens')
        return torch.utils.dlpack.from_dlpack(prompt_speech_tokens.to_dlpack()).cpu()

    def forward_speaker_embedding(self, wav):
        """BLS call to speaker_embedding."""
        inference_request = pb_utils.InferenceRequest(
            model_name='speaker_embedding',
            requested_output_names=['prompt_spk_embedding'],
            inputs=[pb_utils.Tensor.from_dlpack("reference_wav", to_dlpack(wav))]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_spk_embedding = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_spk_embedding')
        return torch.utils.dlpack.from_dlpack(prompt_spk_embedding.to_dlpack())

    async def forward_token2wav(self, target_speech_tokens, prompt_speech_tokens,
                                prompt_speech_feat, prompt_spk_embedding,
                                request_id, token_offset=None, finalize=True,
                                priority=100):
        """Async BLS call to token2wav (flow-only). Returns mel tensor."""
        target_tokens_pb = pb_utils.Tensor.from_dlpack(
            "target_speech_tokens", to_dlpack(target_speech_tokens))
        prompt_tokens_pb = pb_utils.Tensor.from_dlpack(
            "prompt_speech_tokens", to_dlpack(prompt_speech_tokens))
        prompt_feat_pb = pb_utils.Tensor.from_dlpack(
            "prompt_speech_feat", to_dlpack(prompt_speech_feat))
        prompt_emb_pb = pb_utils.Tensor.from_dlpack(
            "prompt_spk_embedding", to_dlpack(prompt_spk_embedding))

        inputs = [target_tokens_pb, prompt_tokens_pb, prompt_feat_pb, prompt_emb_pb]

        if token_offset is not None:
            inputs.append(pb_utils.Tensor("token_offset",
                          np.array([[token_offset]], dtype=np.int32)))
            inputs.append(pb_utils.Tensor("finalize",
                          np.array([[finalize]], dtype=np.bool_)))

        inference_request = pb_utils.InferenceRequest(
            model_name='token2wav',
            requested_output_names=['mel'],
            inputs=inputs,
            request_id=request_id,
            parameters={"priority": priority},
        )

        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        mel = pb_utils.get_output_tensor_by_name(inference_response, 'mel')
        return torch.utils.dlpack.from_dlpack(mel.to_dlpack())

    async def forward_vocoder(self, mel, finalize):
        """Async BLS call to vocoder. Returns speech tensor."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [80, T] -> [1, 80, T]
        mel_pb = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel.float()))
        finalize_pb = pb_utils.Tensor("finalize",
                      np.array([[finalize]], dtype=np.bool_))

        inference_request = pb_utils.InferenceRequest(
            model_name='vocoder',
            requested_output_names=['tts_speech'],
            inputs=[mel_pb, finalize_pb],
        )

        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        speech = pb_utils.get_output_tensor_by_name(inference_response, 'tts_speech')
        return torch.utils.dlpack.from_dlpack(speech.to_dlpack()).cpu()

    def _prepare_prompt(self, request):
        """Extract reference audio, tokenize, compute speaker embedding and mel feat."""
        # wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
        # wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")

        # spk_name = pb_utils.get_input_tensor_by_name(request, "spk_name")
        spk_name = pb_utils.get_input_tensor_by_name(request, "spk_name").as_numpy()
        spk_name = spk_name[0][0].decode('utf-8')

        print("spk_name : ",spk_name)
        assert spk_name in self.spk2text
        filename = self.spk2filename[spk_name]
        print("filename : ",filename)
        prompt_wav_path = os.path.join(self.prompt_speech_dir, filename)
        print("prompt_wav_path: ", prompt_wav_path)
        assert os.path.exists(prompt_wav_path)
        wav, sr = sf.read(prompt_wav_path)
        print('1')
        assert sr == 16000, "sample rate hardcoded in server"
        assert len(wav.shape) == 1, "waveform should be 1D"
        wav_len = np.array([[len(wav)]], dtype=np.int32)
        print("2")
        wav_len = torch.from_numpy(wav_len)
        print("3")
        wav = wav.reshape(1, -1).astype(np.float32)
        print("4")
        wav = torch.from_numpy(wav)
        print("5")
        


        # reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text")
        reference_text = self.spk2text[spk_name]
        print("6")
        # reference_text = reference_text.as_numpy()[0][0].decode('utf-8') if reference_text is not None else ""
        if '<|endofprompt|>' not in reference_text:
            reference_text = 'You are a helpful assistant.<|endofprompt|>' + reference_text
        print("7")
        # Check speaker cache
        if reference_text in self.speaker_cache:
            cached = self.speaker_cache[reference_text]
            return (cached['prompt_speech_tokens_for_llm'], cached['prompt_speech_tokens'],
                    cached['prompt_speech_feat'], cached['prompt_spk_embedding'], reference_text)
        print("8")
        # Audio tokenizer
        wav_np = wav.numpy()
        print("9")
        # wav_np = wav
        wav_len_val = wav_len.numpy()[0][0]
        print(wav_len_val)
        print(type(wav))
        print(type(wav_len))
        # wav_len_val = wav_len[0][0]
        pb_wav = pb_utils.Tensor("reference_wav", wav.cpu().numpy().astype(np.float32))
        pb_wav_len = pb_utils.Tensor("reference_wav_len", wav_len.cpu().numpy().astype(np.int32))
        prompt_speech_tokens = self.forward_audio_tokenizer(pb_wav, pb_wav_len)
        print("extract token done.")
        prompt_speech_tokens = prompt_speech_tokens.unsqueeze(0)  # [1, T]

        # Speaker embedding
        wav_tensor = torch.from_numpy(wav_np)
        wav_tensor = wav_tensor[:, :wav_len_val]
        prompt_spk_embedding = self.forward_speaker_embedding(wav_tensor)
        print("extract embedding done")

        # Mel extraction at 24kHz with CosyVoice3 params
        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=24000)(wav_tensor)
        speech_feat = self._extract_speech_feat(prompt_speech_resample)
        print("extrace feat done.")

        # Keep full tokens for LLM prefill (untruncated)
        prompt_speech_tokens_for_llm = prompt_speech_tokens.clone()

        # Align prompt speech feat and tokens to 2:1 ratio (for flow model only)
        orig_feat_len = speech_feat.shape[1]
        orig_token_len = prompt_speech_tokens.shape[-1]
        token_len = min(int(speech_feat.shape[1] / 2), prompt_speech_tokens.shape[-1])
        prompt_speech_feat = speech_feat[:, :2 * token_len].contiguous().half()
        prompt_speech_tokens = prompt_speech_tokens[:, :token_len].contiguous()

        # Cache
        self.speaker_cache[reference_text] = {
            'prompt_speech_tokens_for_llm': prompt_speech_tokens_for_llm,
            'prompt_speech_tokens': prompt_speech_tokens,
            'prompt_speech_feat': prompt_speech_feat,
            'prompt_spk_embedding': prompt_spk_embedding,
        }

        return prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, prompt_spk_embedding, reference_text

    async def _process_request_streaming(self, request):
        """Process a single request in streaming (decoupled) mode."""
        request_id = request.request_id()
        response_sender = request.get_response_sender()

        try:
            prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
                prompt_spk_embedding, reference_text = self._prepare_prompt(request)

            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')

            semantic_token_ids_arr = []
            token_offset = 0
            chunk_index = 0
            this_token_hop_len = self.token_hop_len
            accumulated_mel = None
            speech_offset = 0
            start_time = time.time()

            async for generated_id in self.forward_llm_streaming(
                target_text=target_text,
                reference_text=reference_text,
                prompt_speech_tokens=prompt_speech_tokens_for_llm,
            ):
                semantic_token_ids_arr.append(generated_id)

                while True:
                    pending_num = len(semantic_token_ids_arr) - token_offset
                    if pending_num < this_token_hop_len + self.flow_pre_lookahead_len:
                        break

                    # Prepare tokens for this chunk
                    end_idx = token_offset + this_token_hop_len + self.flow_pre_lookahead_len
                    this_tokens = torch.tensor(
                        semantic_token_ids_arr[:end_idx]
                    ).unsqueeze(0).to(torch.int32).to(self.device)

                    # Call token2wav (flow-only) -> mel_chunk
                    mel_chunk = await self.forward_token2wav(
                        this_tokens, prompt_speech_tokens,
                        prompt_speech_feat, prompt_spk_embedding,
                        request_id, token_offset=token_offset, finalize=False,
                        priority=chunk_index + 1,
                    )

                    # Accumulate mel
                    if mel_chunk.dim() == 2:
                        mel_chunk = mel_chunk.unsqueeze(0)
                    if accumulated_mel is None:
                        accumulated_mel = mel_chunk
                    else:
                        accumulated_mel = torch.cat([accumulated_mel, mel_chunk], dim=2)

                    # Call vocoder
                    speech = await self.forward_vocoder(accumulated_mel, finalize=False)

                    # Extract new speech
                    new_speech = speech[:, speech_offset:]
                    speech_offset += new_speech.shape[1]

                    if new_speech.shape[1] > 0:
                        audio_tensor = pb_utils.Tensor.from_dlpack(
                            "waveform", to_dlpack(new_speech))
                        inference_response = pb_utils.InferenceResponse(
                            output_tensors=[audio_tensor])
                        response_sender.send(inference_response)

                    token_offset += this_token_hop_len

                    # Dynamic chunk strategy
                    if self.dynamic_chunk_strategy == "exponential":
                        this_token_hop_len = self.token_frame_rate * (2 ** chunk_index)
                    elif self.dynamic_chunk_strategy == "time_based":
                        cost_time = time.time() - start_time
                        duration = token_offset / self.token_frame_rate
                        if chunk_index > 0 and cost_time > 0:
                            avg_chunk_time = cost_time / (chunk_index + 1)
                            if avg_chunk_time > 0:
                                multiples = (duration - cost_time) / avg_chunk_time
                                next_pending = len(semantic_token_ids_arr) - token_offset
                                if multiples > 4:
                                    this_token_hop_len = (next_pending // self.token_hop_len + 1) * self.token_hop_len
                                elif multiples > 2:
                                    this_token_hop_len = (next_pending // self.token_hop_len) * self.token_hop_len
                                else:
                                    this_token_hop_len = self.token_hop_len
                                this_token_hop_len = max(self.token_hop_len, this_token_hop_len)

                    chunk_index += 1

            # Final chunk with remaining tokens
            if len(semantic_token_ids_arr) > 0:
                remaining_tokens = torch.tensor(
                    semantic_token_ids_arr
                ).unsqueeze(0).to(torch.int32).to(self.device)

                mel_chunk = await self.forward_token2wav(
                    remaining_tokens, prompt_speech_tokens,
                    prompt_speech_feat, prompt_spk_embedding,
                    request_id, token_offset=token_offset, finalize=True,
                    priority=chunk_index + 1,
                )

                if mel_chunk.dim() == 2:
                    mel_chunk = mel_chunk.unsqueeze(0)
                if accumulated_mel is None:
                    accumulated_mel = mel_chunk
                else:
                    accumulated_mel = torch.cat([accumulated_mel, mel_chunk], dim=2)

                speech = await self.forward_vocoder(accumulated_mel, finalize=True)

                new_speech = speech[:, speech_offset:]
                if new_speech.shape[1] > 0:
                    audio_tensor = pb_utils.Tensor.from_dlpack(
                        "waveform", to_dlpack(new_speech))
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[audio_tensor])
                    response_sender.send(inference_response)

            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        except Exception as e:
            self.logger.log_error(f"Error in streaming request: {e}")
            error_response = pb_utils.InferenceResponse(
                error=pb_utils.TritonError(str(e)))
            response_sender.send(error_response)
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    async def _process_request_offline(self, request):
        """Process a single request in offline (non-decoupled) mode."""
        request_id = request.request_id()
        print("before prepare")
        prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
            prompt_spk_embedding, reference_text = self._prepare_prompt(request)
        print("pre done.")
        target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
        target_text = target_text[0][0].decode('utf-8')

        # Get all speech tokens at once (use full untruncated prompt tokens for LLM)
        all_token_ids = await self.forward_llm_offline(
            target_text=target_text,
            reference_text=reference_text,
            prompt_speech_tokens=prompt_speech_tokens_for_llm,
        )

        if len(all_token_ids) == 0:
            raise pb_utils.TritonModelException("LLM generated no speech tokens")

        all_tokens = torch.tensor(all_token_ids).unsqueeze(0).to(torch.int32).to(self.device)

        # token2wav (no token_offset, finalize=True) -> full mel
        mel = await self.forward_token2wav(
            all_tokens, prompt_speech_tokens,
            prompt_speech_feat, prompt_spk_embedding,
            request_id,
        )

        # vocoder -> full speech
        speech = await self.forward_vocoder(mel, finalize=True)

        audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(speech))
        return pb_utils.InferenceResponse(output_tensors=[audio_tensor])

    async def execute(self, requests):
        if self.decoupled:
            tasks = [
                asyncio.create_task(self._process_request_streaming(request))
                for request in requests
            ]
            await asyncio.gather(*tasks)
            return None
        else:
            responses = []
            for request in requests:
                try:
                    response = await self._process_request_offline(request)
                    responses.append(response)
                except Exception as e:
                    print("e : ", e)
                    self.logger.log_error(f"Error in offline request: {e}")
                    responses.append(pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))))
            return responses

    def finalize(self):
        self.logger.log_info("Finalizing CosyVoice3 BLS model")
        if hasattr(self, "http_client"):
            asyncio.run(self.http_client.aclose())
