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

import soundfile as sf

from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

import tempfile


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "Qwen/Qwen2-Audio-7B-Instruct"

def qwen2_audio_7b_instruct_model_loader(self):

    self.processor = AutoProcessor.from_pretrained(model_path)
    self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto",
    torch_dtype=torch.float16)
    logger.info("Model loaded: {}".format(model_path))


def post_process_qwen2_asr(model_output):
    
    match = re.search(r'"((?:\\.|[^"\\])*)"', model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    if ":'" in model_output:
        model_output = "'" + model_output.split(":'")[1]
    elif ": '" in model_output:
        model_output = "'" + model_output.split(": '")[1]

    # Find the longest match of ''
    match = re.search(r"'(.*)'", model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    return model_output

def post_process_S2TT(model_output):
    try:
        # 작은따옴표로 둘러싸인 모든 부분을 찾습니다.
        matches = re.findall(r"'(.*?)'", model_output)
        
        if matches:
            # 매칭된 결과가 있다면, 그 중 가장 마지막 것을 반환
            return matches[-1]
        else:
            # 매칭된 것이 없으면 원본 반환
            return model_output
    except Exception as e:
        # 만약을 위한 예외 처리
        print(f"Error during post-processing: {e}")
        return model_output

def qwen2_audio_7b_instruct_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    os.makedirs('tmp', exist_ok=True)

    # 1. 임시 파일을 저장할 변수 초기화
    audio_path_to_delete = None 
    
    try:
        # 2. ASR 관련 모든 태스크 (ASR, ASR-en, ASR-ko)는 30초 넘으면 청크 처리
        if audio_duration > 30 and input['task_type'].startswith('ASR'):
            logger.info('Audio duration is more than 30 seconds for ASR task. Chunking.')
            audio_chunks = []
            for i in range(0, len(audio_array), 30 * sampling_rate):
                audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
            
            model_predictions = []
            for chunk in tqdm(audio_chunks):
                chunk_audio_path = None
                try:
                    # NamedTemporaryFile은 파일 핸들을 닫기 전에도 path를 제공합니다.
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
                    chunk_audio_path = temp_file.name
                    temp_file.close() # 핸들 닫기
                    
                    sf.write(chunk_audio_path, chunk, sampling_rate)

                    conversation = [
                        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                        {"role": "user", "content": [
                            {"type": "audio", "audio_url": chunk_audio_path},
                            {"type": "text", "text": input["instruction"]},
                        ]},
                    ]

                    text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    audios = []
                    for message in conversation:
                        if isinstance(message["content"], list):
                            for ele in message["content"]:
                                if ele["type"] == "audio":
                                    audios.append(
                                        librosa.load(
                                            ele['audio_url'],
                                            sr=self.processor.feature_extractor.sampling_rate)[0]
                                    )

                    inputs = self.processor(text=text, audio=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
                    inputs = inputs.to("cuda")

                    generate_ids = self.model.generate(**inputs, max_length=1024)
                    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

                    response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
                    # ASR 태스크는 항상 후처리
                    if input['task_type'].startswith('ASR'): 
                        response = post_process_qwen2_asr(response)

                    model_predictions.append(response)

                finally:
                    # 3. 청크 처리 후 파일 즉시 삭제
                    if chunk_audio_path and os.path.exists(chunk_audio_path):
                        os.remove(chunk_audio_path)

            output = ' '.join(model_predictions)

        # 30초 초과 및 비-ASR 태스크 (예: 캡셔닝) -> 30초로 자르기
        elif audio_duration > 30:
            logger.info('Audio duration is more than 30 seconds (non-ASR). Taking first 30 seconds.')
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
            audio_path_to_delete = temp_file.name
            temp_file.close()
            
            sf.write(audio_path_to_delete, audio_array[:30 * sampling_rate], sampling_rate)

            if input['task_type'] == 'ASR-en':
                print("task_type : ASR-en")
                prompt = "Please help me transcribe the English speech into text."
            elif input['task_type'] == 'ASR-ko':
                prompt = "Please help me transcribe the Korean speech into text."
            elif input['task_type'] == 'ASR-CS':
                prompt = "Transcribe the speech."
            elif input['task_type'].startswith('CCFQA'):
                prompt = "You are a speech question answering assistant. Do not transcribe unless explicitly asked. Provide only the specific year, date, number, location, or name being asked for, return only an entity."
            else:
                prompt = ""

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path_to_delete},
                    {"type": "text", "text": prompt + input["instruction"]},
                ]},
            ]
            
            # ... (이하 동일한 로직) ...
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    ele['audio_url'],
                                    sr=self.processor.feature_extractor.sampling_rate)[0]
                            )

            inputs = self.processor(text=text, audio=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
            inputs = inputs.to("cuda")

            generate_ids = self.model.generate(**inputs, max_length=1024)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            output = response
            # (이 블록은 이제 ASR이 아니므로 ASR 후처리 로직 제거)

            # if input['task_type']=='S2TTen2ko' or input['task_type']=='S2TTko2en':
            #     output = post_process_S2TT(output)
            # 4. ASR 관련 모든 태스크는 후처리
            if input['task_type'].startswith('ASR'): 
                output = post_process_qwen2_asr(output)
            if input['task_type']=='S2TTen2ko' or input['task_type']=='S2TTko2en':
                output = post_process_S2TT(output)

        # 30초 이하 모든 태스크
        else: 
            if audio_duration < 1:
                logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
                audio_array = np.pad(audio_array, (0, sampling_rate - len(audio_array)), 'constant')

            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
            audio_path_to_delete = temp_file.name
            temp_file.close()
            
            sf.write(audio_path_to_delete, audio_array, sampling_rate)

            if input['task_type'] == 'ASR-en':
                print("task_type : ASR-en")
                prompt = "Please help me transcribe the English speech into text."
            elif input['task_type'] == 'ASR-ko':
                prompt = "Please help me transcribe the Korean speech into text."
            elif input['task_type'] == 'ASR-CS':
                prompt = "Transcribe the speech."
            elif input['task_type'].startswith('CCFQA'):
                prompt = "You are a speech question answering assistant. Do not transcribe unless explicitly asked. Provide only the specific year, date, number, location, or name being asked for, return only an entity."
            else:
                prompt = "" # ASR (일반) 또는 기타 태스크

            conversation = [
                    {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": audio_path_to_delete},
                        {"type": "text", "text": prompt + input["instruction"]},
                    ]},
                ]
            
            # ... (이하 동일한 로직) ...
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    ele['audio_url'],
                                    sr=self.processor.feature_extractor.sampling_rate)[0]
                            )

            inputs = self.processor(text=text, audio=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
            inputs = inputs.to("cuda")

            generate_ids = self.model.generate(**inputs, max_length=1024)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            output = response

            # 4. ASR 관련 모든 태스크는 후처리
            if input['task_type'].startswith('ASR'): 
                output = post_process_qwen2_asr(output)
            if input['task_type']=='S2TTen2ko' or input['task_type']=='S2TTko2en':
                output = post_process_S2TT(output)

        return output

    finally:
        # 5. try 블록에서 예외가 발생하더라도 마지막에 사용된 파일 삭제 시도
        if audio_path_to_delete and os.path.exists(audio_path_to_delete):
            os.remove(audio_path_to_delete)
            logger.info(f"Cleaned up temp file: {audio_path_to_delete}")

