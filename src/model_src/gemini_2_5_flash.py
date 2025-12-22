import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch
import time

from tqdm import tqdm

import pathlib
import soundfile as sf

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor




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


def gemini_2_5_flash_model_loader(self):
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        message = "GOOGLE_API_KEY not configured. Do `export GOOGLE_API_KEY=...` "
        logger.error(message)
        raise RuntimeError(message)

        # 환경 변수에서 읽어온 api_key 값으로 Client 초기화
    self.model = genai.Client(api_key=api_key)
    logger.info("Model loaded")


def do_sample_inference(self, audio_array, instruction, sampling_rate=16000):
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
    sf.write(audio_path.name, audio_array, sampling_rate)
    with open(audio_path.name, 'rb') as f:
        audio_bytes = f.read()
    last_error = None
    for attempt in range(1, 4):
        try:
            response = self.model.models.generate_content(
                        model="gemini-2.5-flash", 
                        contents=[
                            instruction, 
                            types.Part.from_bytes(
                                data=audio_bytes,
                                mime_type='audio/mp3',
                    )
                    ]
                    )
            return response.text
        except Exception as exc:
            last_error = exc
            logger.warning("Gemini request failed (attempt %s/3): %s", attempt, exc)
            if attempt < 3:
                time.sleep(10)  # brief pause before retry
    raise RuntimeError("Gemini request failed after retries") from last_error


def gemini_2_5_flash_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate
    instruction    = input["instruction"]

    if input['task_type'].startswith('CCFQA'):
        instruction = "You are a speech question answering assistant. Do not transcribe unless explicitly asked. Provide only the specific year, date, number, location, or name being asked for, return only an entity." + input["instruction"]

    os.makedirs('tmp', exist_ok=True)

    if audio_duration < 1:
        logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
        audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

    output = do_sample_inference(self, audio_array, instruction)

    return output
