import os
import re
import uuid

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from tqdm import tqdm

import soundfile as sf

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import tempfile
import librosa

from io import BytesIO
from urllib.request import urlopen

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "Qwen/Qwen2.5-Omni-3B"
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

def qwen_2_5_omni_model_loader(self):

    #self.tokenizer               = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    self.model                   = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                                        "Qwen/Qwen2.5-Omni-3B",
                                        torch_dtype="auto",
                                        device_map="auto",
                                        #attn_implementation="flash_attention_2",
                                    ).eval()
    self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Model loaded: {}".format(model_path))


# 'audio_path' 인자를 'audio_array'로 변경
def inference(self, audio_path, prompt, sys_prompt, tts=None):
    if tts:
        messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
                {"type": "text", "text": "Talk following sentence literally." + prompt},
            ]
        },
        ]

    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ]
            },
        ]
    if tts:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        output = self.model.generate(**inputs, use_audio_in_video=True, return_audio=True)

        text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        audio = output[1]
        return text, audio
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("text:", text)
        # image_inputs, video_inputs = process_vision_info([messages])
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        output = self.model.generate(**inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens=256, thinker_do_sample=False)

        text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(text)
        # text = text[0].split('assistant\n')[-1].strip()
        # Extract text after 'assistant' keyword
        raw_text = text[0]
        if 'assistant\n' in raw_text:
            text = raw_text.split('assistant\n')[-1].strip()
        elif 'assistant' in raw_text:
            text = raw_text.split('assistant')[-1].strip()
        else:
            text = raw_text.strip()

        print(f"Extracted text: {text}")
        return text

def post_process_qwen_asr(model_output):
    """
    Post-process ASR output by removing extra formatting like quotes.
    Handles double quotes ("") that appear in the output.
    """
    # First, replace consecutive double quotes with single quotes
    # e.g., ""tteokguk"" -> "tteokguk"
    processed = model_output.replace('""', '"')

    # Try to extract content within quotes
    match = re.search(r'"((?:\\.|[^"\\])*)"', processed)
    if match:
        processed = match.group(1)

    # Handle cases like 'field: "value"'
    if ':"' in processed:
        processed = '"' + processed.split(':"')[1]
    elif ': "' in processed:
        processed = '"' + processed.split(': "')[1]

    # Try one more time to extract from quotes
    match = re.search(r'"(.*)"', processed)
    if match:
        processed = match.group(1)

    # If all post-processing failed, return original output
    # This handles cases where there are no quotes at all
    if not processed or processed.isspace():
        return model_output

    return processed
    
    # match = re.search(r'"((?:\\.|[^"\\])*)"', model_output)
    # if match:
    #     model_output = match.group(1)
    # else:
    #     model_output = model_output

    # if ':"' in model_output:
    #     model_output = '"' + model_output.split(':"')[1]
    # elif ': "' in model_output:
    #     model_output = '"' + model_output.split(': "')[1]

    # # Find the longest match of ''
    # match = re.search(r'"(.*)"', model_output)
    # if match:
    #     model_output = match.group(1)
    # else:
    #     model_output = model_output

    # return model_output


def qwen_2_5_omni_model_generation(self, input, task_type, tts=None):
    tts_mode = bool(tts)
    if not tts_mode:
        task_type_value = str(input.get("task_type", "")).lower() if isinstance(input, dict) else ""
        if task_type_value.startswith("tts"):
            tts_mode = True

    if tts_mode:
        synthesis_text = (
            input.get("synthesis_text")
            or input.get("target_text")
            or input.get("text")
            or input.get("instruction")
        )
        if not synthesis_text:
            raise ValueError("TTS mode expects the input to contain text to synthesize.")

        prompt_audio_path = (
            input.get("prompt_audio_path")
            or input.get("prompt_wav_path")
            or input.get("prompt_wav")
        )
        if not (prompt_audio_path and os.path.exists(prompt_audio_path)):
            prompt_audio_payload = input.get("prompt_audio") if isinstance(input, dict) else None
            if isinstance(prompt_audio_payload, dict):
                audio_array = prompt_audio_payload.get("array")
                audio_sr = (
                    prompt_audio_payload.get("sampling_rate")
                    or prompt_audio_payload.get("sr")
                )
                if audio_array is not None and audio_sr:
                    tmp_dir = os.path.join("tmp", "seed_tts_prompt_refs")
                    os.makedirs(tmp_dir, exist_ok=True)
                    tmp_path = os.path.join(tmp_dir, f"prompt_{uuid.uuid4().hex}.wav")
                    sf.write(tmp_path, np.asarray(audio_array, dtype=np.float32), int(audio_sr))
                    prompt_audio_path = tmp_path
        user_content = []
        if prompt_audio_path and os.path.exists(prompt_audio_path):
            user_content.append({"type": "audio", "audio": prompt_audio_path})
        user_content.append(
            {"type": "text", "text": f"Repeat the following text exactly as it is written. Do not say anything else:\n\n{synthesis_text}"}
        )

        system_prompt = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with torch.no_grad():
            generation_outputs = self.model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=True,
                speaker="Chelsie",
            )

        if isinstance(generation_outputs, tuple):
            text_ids, audio_values = generation_outputs
        else:
            text_ids, audio_values = generation_outputs, None

        decoded_text = processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_text = decoded_text[0].strip() if decoded_text else ""

        if isinstance(audio_values, (list, tuple)):
            audio_tensor = audio_values[0]
        else:
            audio_tensor = audio_values

        if audio_tensor is None:
            raise RuntimeError("TTS generation did not return audio output.")

        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio_tensor)

        audio_np = np.asarray(audio_np).astype(np.float32).flatten()
        sample_rate = getattr(self.model.config, "audio_sample_rate", 24000)

        output_dir = os.path.join("tmp", "tts_outputs", "qwen2_5_omni_3b")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4().hex}.wav")
        sf.write(output_path, audio_np, sample_rate)

        logger.info("Saved TTS output to %s", output_path)

        return {
            "text": decoded_text,
            "audio_path": output_path,
            "sampling_rate": sample_rate,
            "speaker": "Chelsie",
        }
    else:
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

                response = inference(self, audio_path.name, prompt=prompt, sys_prompt="You are a speech recognition model.")


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
                prompt = "Transcribe the Korean audio into text without any punctuation marks."
                response = inference(self, audio_path.name, prompt=prompt, sys_prompt="You are a speech recognition model.")
            elif input['task_type'] =='ASR-en':
                prompt = "Transcribe the English audio into text without any punctuation marks."
                response = inference(self, audio_path.name, prompt=prompt, sys_prompt="You are a speech recognition model.")
            elif input['task_type'] == 'ASR-CS':
                prompt = "Transcribe the speech."
                response = inference(self, audio_path.name, prompt=prompt, sys_prompt="You are a speech recognition model.")
            elif input['task_type'] == 'S2TTen2ko':
                prompt = "Listen to the provided English speech and produce a translation in Korean text."
                response = inference(self, audio_path.name, prompt=prompt, sys_prompt="Listen to the provided English speech and produce a translation in Korean text.")
            elif input['task_type'] == 'S2TTko2en':
                prompt = "Listen to the provided Korean speech and produce a translation in English text."
                response = inference(self, audio_path.name, prompt=prompt, sys_prompt="Listen to the provided Korean speech and produce a translation in English text.")
            elif input['task_type'] == 'Audio-Understanding-Reasoning' or input['task_type'] == 'ASQA':
                prompt = "Listen to the provided question and select a correct Answer." + input["instruction"]
                response = inference(self, audio_path.name, prompt=prompt, sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.")
            elif input['task_type'].startswith('CCFQA'):
                # CCFQA benchmark - instruction already contains full prompt
                prompt = input["instruction"]
                sys_prompt = "You are a speech question answering assistant. Do not transcribe unless explicitly asked. Provide only the specific year, date, number, location, or name being asked for, return only an entity."
                response = inference(self, audio_path.name, prompt=prompt, sys_prompt=sys_prompt)
            else:
                print("Task doesn't defined.")
                prompt = input["instruction"]
                response = inference(self, audio_path.name, prompt=prompt, sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.")
                
            # Reprocess the results to get the output
            if 'ASR' in input['task_type']: response = post_process_qwen_asr(response)

            output = response



        return output


def qwen_2_5_omni_generation(self, input):
    """Alias for compatibility with evaluation pipeline."""
    return qwen_audio_chat_model_generation(self, input)
