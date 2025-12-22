import random
import logging
import re

from jiwer import compute_measures, wer, cer

from dataset_src.prompts.prompts import asr_instructions, korean_asr_instruct

import re

def korean_clean(text):
    """
    1. Handle None or non-string inputs.
    2. Extract text after specific markers ("Final Transcription:", etc.).
    3. If quoted Korean text exists, extract the last one (supports both single and double quotes).
    4. Remove common English prefixes.
    5. Remove brackets, newlines, and specific punctuation.
    6. If no Korean characters remain (e.g., English only), return "".
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # --- 1. 특정 키워드 뒤의 내용 추출 ---
    # 순차적으로 체크하며, 해당 키워드가 있으면 그 뒤의 내용만 취합니다.
    markers = ["Final Transcription:", "output:", "transcribed text is:"]
    for marker in markers:
        if marker in text:
            # 해당 키워드 기준으로 나누고, 뒤쪽(마지막) 부분만 가져옴
            text = text.split(marker, 1)[-1]

    # --- 2. 따옴표("" 또는 '') 사이의 한글 추출 ---
    # 작은따옴표와 큰따옴표 모두 지원
    quote_patterns = [
        r"'([^']+)'",  # Single quotes
        r'"([^"]+)"',  # Double quotes
    ]

    korean_quotes = []
    for pattern in quote_patterns:
        quotes = re.findall(pattern, text)
        # 그 중 '한글'이 포함된 것만 필터링
        korean_quotes.extend([q for q in quotes if re.search(r'[가-힣]', q)])

    # 한글이 포함된 따옴표 내용이 있다면, 그 중 '맨 뒤의 것'을 선택합니다.
    if korean_quotes:
        text = korean_quotes[-1]

    # --- 3. 영어 프리픽스 제거 ---
    # Remove common English prefixes that models might add
    prefixes_to_remove = [
        r"^the\s+(transcription|transcript|text)\s+(of\s+(the|your)\s+audio\s+is|is)[\s:]*",
        r"^the\s+original\s+content\s+of\s+this\s+audio\s+is[\s:]*",
        r"^here\s+is\s+the\s+(transcription|transcript)[\s:]*",
        r"^transcription[\s:]*",
        r"^transcript[\s:]*",
    ]

    text_lower = text.lower()
    for prefix_pattern in prefixes_to_remove:
        text_lower = re.sub(prefix_pattern, "", text_lower, flags=re.IGNORECASE)

    # If prefix was removed, update text
    if len(text_lower) < len(text.lower()):
        text = text[len(text) - len(text_lower):]

    # --- 4. 기존 정제 로직 (줄바꿈, 특수문자, 괄호 제거) ---
    text = text.replace("\n", " ").replace("\r", " ").replace(".", "").replace(",", "").replace("*", "")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    # 공백 정리
    cleaned_text = re.sub(r"\s+", " ", text).strip()

    # --- 5. 영어만 있는 경우(한글이 아예 없는 경우) 빈 문자열 반환 ---
    # 정제된 텍스트에 한글(가-힣)이 하나라도 없으면 "" 리턴
    if not re.search(r'[가-힣]', cleaned_text):
        return ""

    return cleaned_text


class Ksponspeech_eval_clean_dataset(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        self.prompt   = korean_asr_instruct
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample in self.raw_data:
            audio       = sample['audio']
            instruction = random.choice(self.prompt)
            reference   = sample['text']
            input_data.append({
                                "audio"      : audio,
                                "instruction": instruction,
                                "reference"  : reference,
                                "task_type"  : "ASR-ko"
                                })

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data


    def format_model_predictions(self, input_data, model_predictions):

        data_with_model_predictions = []
        for sample in input_data:
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions


    def compute_score(self, data_with_model_predictions, metrics=None):

        supported_metrics = ['wer', 'cer', 'cer-wer']
        if metrics not in supported_metrics:
            raise ValueError(f"Unsupported metric: {metrics}. Supported metrics: {supported_metrics} for ASR")
        
        predictions=[]
        references=[]
        for item in data_with_model_predictions:
            model_prediction = korean_clean(item.get("model_prediction"))
            answer           = korean_clean(item.get("reference"))

            if len(model_prediction) == 0: model_prediction = "empty"
            if len(answer) == 0: answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)

        if metrics == 'wer':
            sample_wer = []
            incorrect  = 0
            total      = 0
            for prediction, reference in zip(predictions, references):
                measures   = compute_measures(reference, prediction)
                incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
                total     += measures["substitutions"] + measures["deletions"] + measures["hits"]

                wer_score = wer(reference, prediction)
                
                sample_wer_score = {
                    "reference" : reference,
                    "prediction": prediction,
                    "wer"       : wer_score,
                }

                sample_wer.append(sample_wer_score)

            total_wer = incorrect / total

            return {"wer": total_wer, "sample_wer": sample_wer}

        elif metrics == 'cer':
            sample_cer = []
            incorrect  = 0
            total      = 0
            for prediction, reference in zip(predictions, references):
                measures   = compute_measures(reference, prediction, char_level=True) 
                incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
                total     += measures["substitutions"] + measures["deletions"] + measures["hits"] # 총 문자 수

                cer_score = cer(reference, prediction) 
                
                sample_cer_score = {
                    "reference" : reference,
                    "prediction": prediction,
                    "cer"       : cer_score,
                }

                sample_cer.append(sample_cer_score)

            total_cer = incorrect / total

            return {"cer": total_cer, "sample_cer": sample_cer}
        
        elif metrics == 'cer-wer':
            sample_cer_wer = []
            
            for prediction, reference in zip(predictions, references):
                
                sample_cer_score = cer(reference, prediction) 
                sample_wer_score = wer(reference, prediction)
                
                sample_cer_wer_score = {
                    "reference" : reference,
                    "prediction": prediction,
                    "cer"       : sample_cer_score,
                    "wer"       : sample_wer_score
                }
                sample_cer_wer.append(sample_cer_wer_score)
                
            
            total_wer = wer(references, predictions)
            total_cer = cer(references, predictions)

            return {"wer": total_wer, "cer": total_cer, "sample_cer_wer": sample_cer_wer}
