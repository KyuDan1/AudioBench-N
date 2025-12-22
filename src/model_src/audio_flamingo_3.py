import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('/mnt/fr20tb/kyudan/audio-flamingo')

import logging
import numpy as np
import torch

from tqdm import tqdm

import soundfile as sf

from io import BytesIO
from urllib.request import urlopen
import tempfile

from huggingface_hub import snapshot_download
import llava
from llava import conversation as clib
from llava.media import Sound


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_base = "nvidia/audio-flamingo-3"

def audio_flamingo_3_model_loader(self):
    # Load Audio-Flamingo-3 model
    model_path = snapshot_download(model_base)
    self.model = llava.load(model_path, device_map=None)
    self.model = self.model.to("cuda")

    # Set conversation mode
    clib.default_conversation = clib.conv_templates["auto"].copy()

    logger.info("Model loaded: {}".format(model_base))


def audio_flamingo_3_model_generation(self, input):
    audio_array = input["audio"]["array"]
    sampling_rate = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    os.makedirs('tmp', exist_ok=True)

    # 임시 파일을 저장할 변수 초기화
    audio_path_to_delete = None

    try:
        # ASR 관련 모든 태스크 (ASR, ASR-en, ASR-ko)는 30초 넘으면 청크 처리
        if audio_duration > 30 and input['task_type'].startswith('ASR'):
            logger.info('Audio duration is more than 30 seconds for ASR task. Chunking.')
            audio_chunks = []
            for i in range(0, len(audio_array), 30 * sampling_rate):
                audio_chunks.append(audio_array[i:i + 30 * sampling_rate])

            model_predictions = []
            for chunk in tqdm(audio_chunks):
                chunk_audio_path = None
                try:
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
                    chunk_audio_path = temp_file.name
                    temp_file.close()

                    sf.write(chunk_audio_path, chunk, sampling_rate)

                    # Prepare prompt for Audio-Flamingo-3
                    prompt = []
                    prompt.append(Sound(chunk_audio_path))
                    prompt.append(input["instruction"])

                    # Generate response
                    response = self.model.generate_content(prompt)
                    model_predictions.append(response)

                finally:
                    # 청크 처리 후 파일 즉시 삭제
                    if chunk_audio_path and os.path.exists(chunk_audio_path):
                        os.remove(chunk_audio_path)

            output = ' '.join(model_predictions)

        # 30초 초과 및 비-ASR 태스크 -> 30초로 자르기
        elif audio_duration > 30:
            logger.info('Audio duration is more than 30 seconds (non-ASR). Taking first 30 seconds.')

            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
            audio_path_to_delete = temp_file.name
            temp_file.close()

            sf.write(audio_path_to_delete, audio_array[:30 * sampling_rate], sampling_rate)

            # Prepare prompt for Audio-Flamingo-3
            prompt = []
            prompt.append(Sound(audio_path_to_delete))
            text_prompt = input["instruction"]
            if input['task_type'] == 'ASR-en':
                text_prompt = "Please help me transcribe the English speech into text. " + text_prompt
            elif input['task_type'] == 'ASR-ko':
                text_prompt = "Please help me transcribe the Korean speech into text. " + text_prompt
            elif input['task_type'] == 'ASR-CS':
                text_prompt = "Transcribe the speech. " + text_prompt
            elif input['task_type'].startswith('CCFQA'):
                text_prompt = "You are a speech question answering assistant. Do not transcribe unless explicitly asked. Provide only the specific year, date, number, location, or name being asked for, return only an entity. " + text_prompt
            else:
                text_prompt = text_prompt

            prompt.append(text_prompt)
            # prompt.append(input["instruction"])

            # Generate response
            output = self.model.generate_content(prompt)

        # 30초 이하 모든 태스크
        else:
            if audio_duration < 1:
                logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
                audio_array = np.pad(audio_array, (0, sampling_rate - len(audio_array)), 'constant')

            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
            audio_path_to_delete = temp_file.name
            temp_file.close()

            sf.write(audio_path_to_delete, audio_array, sampling_rate)

            # Prepare prompt for Audio-Flamingo-3
            prompt = []
            prompt.append(Sound(audio_path_to_delete))

            # Add task-specific prefix if needed
            text_prompt = input["instruction"]
            if input['task_type'] == 'ASR-en':
                text_prompt = "Please help me transcribe the English speech into text. " + text_prompt
            elif input['task_type'] == 'ASR-ko':
                text_prompt = "Please help me transcribe the Korean speech into text. " + text_prompt
            elif input['task_type'] == 'ASR-CS':
                text_prompt = "Transcribe the speech. " + text_prompt
            elif input['task_type'].startswith('CCFQA'):
                text_prompt = "You are a speech question answering assistant. Do not transcribe unless explicitly asked. Provide only the specific year, date, number, location, or name being asked for, return only an entity. " + text_prompt
            else:
                text_prompt = text_prompt


            prompt.append(text_prompt)

            # Generate response
            output = self.model.generate_content(prompt)

        return output

    finally:
        # try 블록에서 예외가 발생하더라도 마지막에 사용된 파일 삭제 시도
        if audio_path_to_delete and os.path.exists(audio_path_to_delete):
            os.remove(audio_path_to_delete)
            logger.info(f"Cleaned up temp file: {audio_path_to_delete}")

