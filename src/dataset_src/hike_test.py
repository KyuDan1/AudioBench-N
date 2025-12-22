"""
HiKE: Hierarchical Evaluation Framework for Korean-English Code-Switching Speech Recognition

Dataset: thetaone-ai/HiKE (Hugging Face)
Metrics: MER (Mixed Error Rate), PIER (Point of Interest Error Rate)
Paper: https://arxiv.org/abs/2509.24613
"""

import json
import logging
import random
import re
from typing import Tuple, Dict, Any, List

import jiwer
import regex
from jiwer.transforms import (
    Compose,
    ExpandCommonEnglishContractions,
    ReduceToListOfListOfWords,
    ReduceToSingleSentence,
    RemoveKaldiNonWords,
    RemoveMultipleSpaces,
    RemovePunctuation,
    RemoveWhiteSpace,
    Strip,
    SubstituteWords,
    ToLowerCase,
)

# Fixed instruction to match HiKE main.py default behavior
# from dataset_src.prompts.prompts import cs_asr_instructions  # No longer needed

# Import PIER from eval_methods
from dataset_src.eval_methods.pier.measures import pier as hike_pier


# ============================================================================
# Audio extraction helper (from HiKE/src/models/__init__.py)
# ============================================================================

def _extract_audio_array(audio):
    """Extract numpy array from various audio input types."""
    import numpy as np
    
    # Case 1: dict with 'array' key (legacy datasets format)
    if isinstance(audio, dict) and "array" in audio:
        return {"array": audio["array"], "sampling_rate": audio.get("sampling_rate", 16000)}
    
    # Case 2: AudioDecoder from newer datasets versions (torchcodec)
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        arr = samples.data.squeeze(0).numpy()
        return {"array": arr, "sampling_rate": samples.sample_rate}
    
    # Case 3: Already a numpy array
    if isinstance(audio, np.ndarray):
        return {"array": audio, "sampling_rate": 16000}
    
    # Fallback: return as-is
    return audio


# ============================================================================
# HiKE-specific utility functions (from HiKE/src/utils.py)
# ============================================================================

class SpaceKoreanChars:
    """Add space after all Korean characters to consider Korean characters as a single token."""
    def __call__(self, s: str) -> str:
        s = re.sub(r"([\uAC00-\uD7A3])", r" \1 ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s


def replace_loanword(text: str, loanwords: str) -> str:
    """Replace Korean loanwords with their English equivalents."""
    if isinstance(text, dict):
        if "text" in text:
            text = text["text"]
    try:
        if isinstance(loanwords, str):
            loanwords_list = json.loads(loanwords)
        else:
            loanwords_list = loanwords
        for loanword in loanwords_list:
            text = text.replace(loanword["Korean"], loanword["English"])
    except Exception:
        pass
    return text


def add_space(text: str) -> str:
    """Add space between English and Korean characters (e.g., 'bug는' -> 'bug 는')."""
    return regex.sub(r"([A-Za-z0-9]+)(?=\p{Script=Hangul})", r"\1 ", text)


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation.
    Exactly matches HiKE/src/utils.py normalize_text function.
    """
    if isinstance(text, dict):
        text = text["text"]
    
    wer_standardize_contiguous = Compose([
        ToLowerCase(),
        ExpandCommonEnglishContractions(),
        RemoveKaldiNonWords(),
        SubstituteWords({"—": " "}),
        RemovePunctuation(),
        RemoveWhiteSpace(replace_by_space=True),
        RemoveMultipleSpaces(),
        Strip(),
        ReduceToSingleSentence(),
        ReduceToListOfListOfWords(),
    ])
    return " ".join(wer_standardize_contiguous(text)[0])


# ============================================================================
# MER Calculation (from HiKE/src/metrics/mer.py)
# ============================================================================

def compute_mer(ref_text: str, hyp_text: str) -> Tuple[float, dict]:
    """
    Compute Mixed Error Rate (MER) for Korean-English code-switching.
    Exactly matches HiKE/src/metrics/mer.py mer function.
    """
    steps = [SpaceKoreanChars(), Strip()]
    transform = Compose(steps)
    
    ref_t = transform(ref_text)
    hyp_t = transform(hyp_text)
    
    if not ref_t or len(ref_t.split()) == 0:
        return 0.0, {"S": 0, "I": 0, "D": 0, "N": 0}
    
    # jiwer v3.0.0+ uses process_words() returning dataclass
    # jiwer v2.x uses compute_measures() returning dict
    if hasattr(jiwer, 'process_words'):
        # v3.0.0+
        result = jiwer.process_words(ref_t, hyp_t)
        S = result.substitutions
        I = result.insertions
        D = result.deletions
        mer_percent = result.wer * 100
    elif hasattr(jiwer, 'compute_measures'):
        # v2.x fallback
        result = jiwer.compute_measures(ref_t, hyp_t)
        S = result["substitutions"]
        I = result["insertions"]
        D = result["deletions"]
        mer_percent = result["wer"] * 100
    else:
        raise RuntimeError("jiwer version not supported. Need v2.x or v3.0.0+")
    
    N = len(ref_t.split())
    
    if N == 0:
        return mer_percent, {"S": 0, "I": I, "D": 0, "N": 0}
    
    return mer_percent, {"S": S, "I": I, "D": D, "N": N}


# ============================================================================
# PIER Calculation (from HiKE/src/metrics/__init__.py)
# ============================================================================

def pier_fixed(ref: str, pred: str) -> float:
    """
    Compute PIER (Point of Interest Error Rate) score.
    Exactly matches HiKE/src/metrics/__init__.py pier_fixed function.
    """
    # If there's no "rest" in reference, PIER always returns 0.
    # To prevent this, add dummy token out of poi to reference
    _ref = ref + " 뷁"
    _pred = pred + " 뷁"
    
    try:
        pier_result = hike_pier(_ref, _pred)
        pier_poi = pier_result["poi"]
        pier_score = pier_poi["PIER"]
        return pier_score
    except Exception as e:
        logging.error(f"PIER calculation error: {e}")
        return 0.0


# ============================================================================
# HiKE Dataset Processor
# ============================================================================

class hike_test_dataset:
    """
    HiKE (Hierarchical Korean-English Code-Switching) Dataset Processor.
    
    Evaluation process exactly matches HiKE/src/main.py:
    1. replace_loanword on prediction, ref_pier, ref_mer
    2. normalize_text on prediction
    3. add_space on prediction (for PIER only)
    4. compute MER(ref_mer, pred_normalized)
    5. compute PIER(ref_pier, pier_pred)
    """
    
    def __init__(self, raw_data, number_of_samples: int):
        if number_of_samples != -1 and number_of_samples < len(raw_data):
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        # Fixed instruction to match HiKE main.py default behavior
        self.instruction = "Transcribe the speech."
        self.replace_loanword_flag = True
        
        logging.info(f'HiKE Dataset: {len(self.raw_data)} samples loaded')
        logging.info(f'Using fixed instruction: {self.instruction}')
    
    def prepare_model_input(self) -> List[Dict[str, Any]]:
        input_data = []
        
        for sample in self.raw_data:
            # Extract audio with sampling_rate (matches HiKE BaseASR behavior)
            audio = _extract_audio_array(sample['audio'])
            # Use fixed instruction (matches HiKE main.py default)
            instruction = self.instruction
            
            input_data.append({
                "audio": audio,
                "instruction": instruction,
                "reference": sample['text'],
                "text_normalized": sample['text_normalized'],
                "text_pier_labeled": sample['text_pier_labeled'],
                "cs_level": sample['cs_level'],
                "loanwords": sample['loanwords'],
                "sample_id": sample.get('sample_id', ''),
                "task_type": "ASR-CS",
            })
        
        logging.info('\n=  =  =  HiKE Dataset Sample  =  =  =')
        if input_data:
            sample_log = {k: v for k, v in random.choice(input_data).items() if k != 'audio'}
            logging.info(sample_log)
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')
        
        return input_data
    
    def format_model_predictions(self, input_data: List[Dict], model_predictions: List[str]) -> List[Dict]:
        data_with_predictions = []
        
        for sample in input_data:
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_predictions.append(new_sample)
        
        return data_with_predictions
    
    def compute_score(self, data_with_model_predictions: List[Dict], metrics: str = None) -> Dict[str, Any]:
        """
        Compute evaluation scores for HiKE benchmark.
        Exactly matches HiKE/src/main.py evaluation process.
        """
        supported_metrics = ['mer', 'pier', 'mer-pier']
        if metrics is not None and metrics not in supported_metrics:
            raise ValueError(f"Unsupported metric: {metrics}. Supported: {supported_metrics}")
        
        if metrics is None:
            metrics = 'mer-pier'
        
        results = {
            "overall": {},
            "by_cs_level": {"word": {}, "phrase": {}, "sentence": {}},
            "details": [],
        }
        
        all_mer_scores = []
        all_pier_scores = []
        
        for item in data_with_model_predictions:
            prediction = item.get("model_prediction", "")
            loanwords = item.get("loanwords", "[]")
            cs_level = item.get("cs_level", "unknown")
            
            sample_result = {
                "sample_id": item.get("sample_id", ""),
                "cs_level": cs_level,
                "reference": item.get("reference", ""),
                "prediction": prediction,
            }
            
            # Step 1: Apply loanword replacement (matches HiKE main.py lines 86-97)
            if self.replace_loanword_flag:
                pred = replace_loanword(prediction, loanwords)
                ref_pier = replace_loanword(item.get("text_pier_labeled", ""), loanwords)
                ref_mer = replace_loanword(item.get("text_normalized", ""), loanwords)
            else:
                pred = prediction
                ref_pier = item.get("text_pier_labeled", "")
                ref_mer = item.get("text_normalized", "")
            
            # Step 2: Normalize prediction (matches HiKE main.py line 99)
            pred = normalize_text(pred)
            
            # Step 3: Add space for PIER (matches HiKE main.py lines 100-101)
            pier_pred = add_space(pred)
            
            # Step 4: Compute MER (matches HiKE main.py line 104)
            if metrics in ['mer', 'mer-pier']:
                mer_score, mer_stats = compute_mer(ref_mer, pred)
                sample_result["mer"] = mer_score
                sample_result["mer_stats"] = mer_stats
                all_mer_scores.append({"score": mer_score, "cs_level": cs_level})
            
            # Step 5: Compute PIER (matches HiKE main.py line 103)
            if metrics in ['pier', 'mer-pier']:
                pier_score = pier_fixed(ref_pier, pier_pred)
                sample_result["pier"] = pier_score
                all_pier_scores.append({"score": pier_score, "cs_level": cs_level})
            
            results["details"].append(sample_result)
        
        # Calculate scores by cs_level (matches HiKE main.py lines 114-120)
        cs_levels = ["word", "phrase", "sentence"]
        for level in cs_levels:
            level_mer = [s["score"] for s in all_mer_scores if s["cs_level"] == level]
            level_pier = [s["score"] for s in all_pier_scores if s["cs_level"] == level]
            
            if level_mer:
                results["by_cs_level"][level]["mer"] = sum(level_mer) / len(level_mer)
            if level_pier:
                results["by_cs_level"][level]["pier"] = sum(level_pier) / len(level_pier)
            results["by_cs_level"][level]["count"] = len(level_mer) if level_mer else len(level_pier)
        
        # Calculate overall scores (matches HiKE main.py lines 122-125)
        if all_mer_scores:
            overall_mer = sum(s["score"] for s in all_mer_scores) / len(all_mer_scores)
            results["overall"]["mer"] = overall_mer
            results["mer"] = overall_mer
        
        if all_pier_scores:
            overall_pier = sum(s["score"] for s in all_pier_scores) / len(all_pier_scores)
            results["overall"]["pier"] = overall_pier
            results["pier"] = overall_pier
        
        return results
    
    def compute_score_omni_mmpt(self, data_with_model_predictions: List[Dict], metrics: str = None) -> Dict[str, Any]:
        converted_data = []
        for item in data_with_model_predictions:
            converted_item = item.copy()
            if "generated_text" in item and "model_prediction" not in item:
                converted_item["model_prediction"] = item["generated_text"]
            if "ref_text" in item and "reference" not in item:
                converted_item["reference"] = item["ref_text"]
            converted_data.append(converted_item)
        
        return self.compute_score(converted_data, metrics)
