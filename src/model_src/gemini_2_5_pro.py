import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from tqdm import tqdm

import pathlib
import soundfile as sf

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


# All services about google.generativeai is ended in November, 2025.


from google import genai
from google.genai import types


import tempfile


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


def gemini_2_5_pro_model_loader(self):
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        logger.error("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        return 

        # 환경 변수에서 읽어온 api_key 값으로 Client 초기화
    self.model = genai.Client(api_key=api_key)
    logger.info("Model loaded")


def do_sample_inference(self, audio_array, instruction, sampling_rate=16000):
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
    sf.write(audio_path.name, audio_array, sampling_rate)
    with open(audio_path.name, 'rb') as f:
        audio_bytes = f.read()
    response = self.model.models.generate_content(
                model="gemini-2.5-pro", 
                contents=[
                    instruction, 
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type='audio/mp3',
            )
            ]
            )
    response = response.text
    return response


def gemini_2_5_pro_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate
    instruction    = input["instruction"]

    os.makedirs('tmp', exist_ok=True)

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] in ['ASR', 'ASR-en', 'ASR-ko', 'ASR-CS']:
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
        model_predictions = [do_sample_inference(self, chunk_array, instruction) for chunk_array in tqdm(audio_chunks)]
        output = ' '.join(model_predictions)


    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')

        audio_array = audio_array[:30 * sampling_rate]
        output = do_sample_inference(self, audio_array, instruction)
    
    else: 
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        output = do_sample_inference(self, audio_array, instruction)

    return output

