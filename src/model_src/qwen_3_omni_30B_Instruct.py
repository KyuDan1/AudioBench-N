import os
# os.environ['VLLM_USE_V1'] = '0'
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import re
import uuid
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

model_path = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

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
def _run_qwen3_messages(self, messages, *, return_audio=False, speaker="Ethan", use_audio_in_video=True):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(self.model.device).to(self.model.dtype)

    generation_kwargs = dict(
        thinker_return_dict_in_generate=True,
        thinker_max_new_tokens=8192,
        thinker_do_sample=False,
        use_audio_in_video=use_audio_in_video,
        return_audio=return_audio,
    )
    if speaker:
        generation_kwargs["speaker"] = speaker

    text_ids, audio = self.model.generate(**inputs, **generation_kwargs)

    sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
    prompt_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    decoded = processor.batch_decode(
        sequences[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = decoded[0].strip() if decoded else ""

    audio_array = None
    if audio is not None:
        if isinstance(audio, (list, tuple)):
            audio_tensor = audio[0]
        else:
            audio_tensor = audio
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio_tensor)
        audio_array = np.asarray(audio_np).reshape(-1)
        audio_array = np.asarray(audio_array * 32767.0).clip(-32768, 32767).astype(np.int16)

    return response, audio_array


def inference(self, audio_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    response, _ = _run_qwen3_messages(self, messages, return_audio=False, speaker=None)
    print(response)
    return response

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


def _build_prompt(task_type, instruction=""):
    """Return task specific prompt fallback to provided instruction."""
    if task_type in ("ASR-en", "ASR"):
        return "Transcribe the English audio into text."
    if task_type == "ASR-ko":
        return "Transcribe the Korean audio into text."
    if task_type == "ASR-CS":
        return "Transcribe the speech."
    if task_type == "S2TTen2ko":
        return "Listen to the provided English speech and produce a translation in Korean text."
    if task_type == "S2TTko2en":
        return "Listen to the provided Korean speech and produce a translation in English text."
    if task_type in ("Audio-Understanding-Reasoning", "ASQA"):
        return "Listen to the provided question and select a correct Answer. " + (instruction or "")
    if task_type and task_type.startswith("CCFQA"):
        # CCFQA benchmark - add system instruction before user instruction
        sys_instruction = "Do not transcribe unless explicitly asked. Provide only the specific year, date, number, location, or name being asked for, return only an entity. "
        return sys_instruction + (instruction or "")
    return instruction or "Describe the audio."


def _generate_seed_tts_output(self, input_sample):
    synthesis_text = (
        input_sample.get("synthesis_text")
        or input_sample.get("target_text")
        or input_sample.get("text")
        or input_sample.get("instruction")
    )
    if not synthesis_text:
        raise ValueError("TTS mode expects text content to synthesize.")

    prompt_audio_path = (
        input_sample.get("prompt_audio_path")
        or input_sample.get("prompt_wav_path")
        or input_sample.get("prompt_wav")
    )
    if not (prompt_audio_path and os.path.exists(prompt_audio_path)):
        prompt_audio_payload = input_sample.get("prompt_audio")
        if isinstance(prompt_audio_payload, dict):
            audio_array = prompt_audio_payload.get("array")
            audio_sr = prompt_audio_payload.get("sampling_rate") or prompt_audio_payload.get("sr")
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
        {
            "type": "text",
            "text": (
                "Repeat the following text exactly as it is written. Do not say anything else:\n\n"
                f"{synthesis_text}"
            ),
        }
    )
    system_prompt = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
        "perceiving auditory and visual inputs, as well as generating text and speech."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    response_text, audio_array = _run_qwen3_messages(
        self, messages, return_audio=True, speaker="Ethan", use_audio_in_video=True
    )
    if audio_array is None:
        raise RuntimeError("TTS generation did not return audio output.")

    sample_rate = getattr(self.model.config, "audio_sample_rate", 24000)
    output_dir = os.path.join("tmp", "tts_outputs", "qwen3_omni_30b")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{uuid.uuid4().hex}.wav")
    sf.write(output_path, audio_array, sample_rate)
    logger.info("Saved TTS output to %s", output_path)

    return {
        "text": response_text,
        "audio_path": output_path,
        "sampling_rate": sample_rate,
        "speaker": "Ethan",
    }


def qwen_3_omni_model_generation(self, input, task_type):

    resolved_task_type = input.get("task_type", task_type) or task_type or ""
    task_type_lower = str(resolved_task_type).lower()

    if task_type_lower.startswith("tts"):
        return _generate_seed_tts_output(self, input)

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate
    instruction    = input.get("instruction", "")
    prompt         = _build_prompt(resolved_task_type, instruction)
    is_asr_task    = 'asr' in task_type_lower

    os.makedirs('tmp', exist_ok=True)

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and is_asr_task:
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

            response = inference(self, audio_path.name, prompt=prompt)


            # Reprocess the results to get the output
            if is_asr_task: response = post_process_qwen_asr(response)

            model_predictions.append(response)
        
        output = ' '.join(model_predictions)


    else: 
        if audio_duration > 30:
            logger.info('Audio duration is more than 30 seconds. Processing full audio without truncation.')
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array, sampling_rate)

        response = inference(self, audio_path.name, prompt=prompt)

        # Reprocess the results to get the output
        if is_asr_task: response = post_process_qwen_asr(response)

        output = response



    return output


def qwen_3_omni_generation(self, input):
    """Alias for compatibility with evaluation pipeline."""
    return qwen_audio_chat_model_generation(self, input)
