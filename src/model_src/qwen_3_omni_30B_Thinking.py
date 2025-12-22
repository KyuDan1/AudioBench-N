import os
# os.environ['VLLM_USE_V1'] = '0'
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import re
import warnings

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch
import soundfile as sf
print(f"PyTorch가 CUDA를 사용할 수 있나요? {torch.cuda.is_available()}")
print(f"PyTorch가 인식하는 GPU 개수: {torch.cuda.device_count()}")
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from tqdm import tqdm

import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import tempfile
import librosa
import audioread
from io import BytesIO
from urllib.request import urlopen
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
def qwen_3_omni_model_loader(self):
    #self.tokenizer               = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    

    # self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path,
    #                                                                     dtype='auto',
    #                                                                     attn_implementation='flash_attention_2',
    #                                                                     device_map="auto")
    # # else:
    self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, device_map="auto", dtype='auto')
    
    self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Model loaded: {}".format(model_path))


# 'audio_path' 인자를 'audio_array'로 변경
def inference(self, audio_path, prompt):
    messages = [
        {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(self.model.device).to(self.model.dtype)

    text_ids, audio = self.model.generate(**inputs, 
                                 thinker_return_dict_in_generate=True,
                                 use_audio_in_video=True, return_audio=False, speaker="Ethan", # If need TTS, change return_audio to True.
                                 thinker_max_new_tokens=8192, thinker_do_sample=False)

    text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if audio is not None:
        audio = np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16)
        return response, audio
    print(text)
    return text

def post_process_qwen_asr(model_output):
    
    match = re.search(r'"((?:\\.|[^"\\])*)"', model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    if ':"' in model_output:
        model_output = '"' + model_output.split(':"')[1]
    elif ': "' in model_output:
        model_output = '"' + model_output.split(': "')[1]

    # Find the longest match of ''
    match = re.search(r'"(.*)"', model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    return model_output


def qwen_3_omni_model_generation(self, input, task_type):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    os.makedirs('tmp', exist_ok=True)

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])

            # if len(audio_chunks) > 10:
            #     logger.info('More than 10 chunks. Taking first 10 chunks.')
            #     break
        
        model_predictions = []
        for chunk in tqdm(audio_chunks):
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
            sf.write(audio_path.name, chunk, sampling_rate)

            response = inference(audio_path.name, prompt=prompt, sys_prompt="You are a speech recognition model.")


            # Reprocess the results to get the output
            if 'ASR' in input['task_type']: response = post_process_qwen_asr(response)

            model_predictions.append(response)
        
        output = ' '.join(model_predictions)


    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array[:30 * sampling_rate], sampling_rate)
        prompt = "Transcribe the English audio into text without any punctuation marks."
        response = inference(self, audio_path.name, prompt=prompt, sys_prompt="You are a speech recognition model.")

        # Reprocess the results to get the output
        if 'ASR' in input['task_type']: response = post_process_qwen_asr(response)

        output = response
    
    else: 
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array, sampling_rate)
        
        # New task add placeholder
        if input['task_type'] =='ASR-ko':
            prompt = "Transcribe the Korean audio into text."
            response = inference(self, audio_path.name, prompt=prompt)
        elif input['task_type'] =='ASR-en':
            prompt = "Transcribe the English audio into text."
            response = inference(self, audio_path.name, prompt=prompt)
        elif input['task_type'] == 'S2TTen2ko':
            prompt = "Listen to the provided English speech and produce a translation in Korean text."
            response = inference(self, audio_path.name, prompt=prompt)
        elif input['task_type'] == 'S2TTko2en':
            prompt = "Listen to the provided Korean speech and produce a translation in English text."
            response = inference(self, audio_path.name, prompt=prompt)
        elif input['task_type'] == 'Audio-Understanding-Reasoning' or input['task_type'] == 'ASQA':
            prompt = "Listen to the provided question and select a correct Answer." + input["instruction"]
            response = inference(self, audio_path.name, prompt=prompt)
        elif input['task_type'].startswith('CCFQA'):
            # CCFQA benchmark - instruction already contains full prompt
            prompt = input["instruction"]
            response = inference(self, audio_path.name, prompt=prompt)
        else:
            print("Task doesn't defined.")
            prompt = input["instruction"]
            response = inference(self, audio_path.name, prompt=prompt)
            
        # Reprocess the results to get the output
        if 'ASR' in input['task_type']: response = post_process_qwen_asr(response)

        output = response



    return output


def qwen_3_omni_generation(self, input):
    """Alias for compatibility with evaluation pipeline."""
    return qwen_audio_chat_model_generation(self, input)
