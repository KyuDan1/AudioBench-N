"""
CCFQA English Test Dataset (Monolingual)
Audio: English -> Answer: English

CCFQA is a cross-lingual cross-modal factuality evaluation benchmark.
This processor handles English monolingual evaluation.

Paper: https://arxiv.org/abs/2508.07295
Dataset: yxdu/ccfqa
Metrics: Character-level F1 Score + LLM-based ACC (Gemma-3-27b-it)
"""

import random
import logging
from typing import List, Dict, Any


class ccfqa_eng_test_dataset(object):
    """
    CCFQA English Monolingual Dataset Processor
    Audio Language: English
    Answer Language: English
    Task: Factual Question Answering
    """

    def __init__(self, raw_data, number_of_samples: int):
        if number_of_samples != -1 and number_of_samples < len(raw_data):
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))

        self.raw_data = raw_data

        # CCFQA official prompt template (from audio_language_ccfqa.py:182)
        # For Qwen2-Audio/Qwen2.5-Omni models
        self.language_mapping = {
            'eng': 'English', 'kor': 'Korean', 'cmn': 'Chinese',
            'fra': 'French', 'jpn': 'Japanese', 'rus': 'Russian',
            'spa': 'Spanish', 'yue': 'Traditional Chinese'
        }

        logging.info(f'CCFQA English Test: {len(self.raw_data)} samples loaded')

    def prepare_model_input(self) -> List[Dict[str, Any]]:
        """
        Prepare model input with audio and question in English
        Expected answer is in English
        Uses CCFQA official prompt template
        """
        input_data = []

        for sample in self.raw_data:
            # Audio is in English, Question is in English
            audio = sample['audio']
            question = sample['eng_q']  # English question
            reference_answer = sample['eng_a']  # English answer
            src_lang = "eng"
            tgt_lang = "eng"

            # CCFQA official instruction (text only, chat template applied by model)
            # Based on audio_language_ccfqa.py:182
            instruction = (
                f"Answer this {self.language_mapping[src_lang]} speech question in {self.language_mapping[tgt_lang]}. "
                f"Answer this Question."
            )

            input_data.append({
                "audio": audio,
                "instruction": instruction,
                "reference": reference_answer,
                "question": question,
                "sample_id": sample['id'],
                "label": sample['label'],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "task_type": "CCFQA-QA-eng",
            })

        logging.info('\n=  =  =  CCFQA English Dataset Sample  =  =  =')
        if input_data:
            sample_log = {k: v for k, v in random.choice(input_data).items() if k != 'audio'}
            logging.info(sample_log)
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data

    def format_model_predictions(self, input_data: List[Dict], model_predictions: List[str]) -> List[Dict]:
        """Format model predictions by removing audio and adding predictions"""
        data_with_predictions = []

        for sample in input_data:
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_predictions.append(new_sample)

        return data_with_predictions

    def calculate_f1_score(self, target_text: str, prediction_text: str) -> float:
        """
        Calculate character-level F1 score (matching CCFQA official implementation)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        Based on character set intersection
        """
        if not target_text and not prediction_text:
            return 100.0  # Both empty, F1 is 100%

        target_chars = set(list(target_text))
        prediction_chars = set(list(prediction_text))

        common_chars = target_chars.intersection(prediction_chars)

        tp = len(common_chars)

        # If one is empty but the other is not
        if (not target_text and prediction_text) or (target_text and not prediction_text):
            return 0.0

        precision = tp / len(prediction_chars) if len(prediction_chars) > 0 else 0
        recall = tp / len(target_chars) if len(target_chars) > 0 else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1 * 100, 1)  # Return as percentage with 1 decimal

    def compute_score(self, data_with_model_predictions: List[Dict], metrics: str = None) -> Dict[str, Any]:
        """
        Compute evaluation scores for CCFQA benchmark
        Metrics: Character-level F1 Score or LLM-based ACC (Gemma-3-27b-it)
        """
        if metrics is None:
            metrics = 'f1'

        supported_metrics = ['f1', 'gemma3_27b_judge']
        if metrics not in supported_metrics:
            raise ValueError(f"Unsupported metric: {metrics}. Supported: {supported_metrics}")

        # Handle LLM-based ACC metric (Gemma-3-27b-it judge)
        if metrics == 'gemma3_27b_judge':
            from dataset_src.eval_methods.eval_gemma3_27b_ccfqa import gemma3_27b_as_judge_ccfqa

            # Prepare data for LLM judge
            questions = [item.get("question", "") for item in data_with_model_predictions]
            references = [item.get("reference", "") for item in data_with_model_predictions]
            predictions = [item.get("model_prediction", "") for item in data_with_model_predictions]
            src_langs = [item.get("src_lang", "eng") for item in data_with_model_predictions]
            tgt_langs = [item.get("tgt_lang", "eng") for item in data_with_model_predictions]

            input_data_judge = (questions, references, predictions, src_langs, tgt_langs)

            # Run LLM judge evaluation
            judge_results, judge_details = gemma3_27b_as_judge_ccfqa(
                model_path="google/gemma-3-27b-it",
                input_data=input_data_judge
            )

            return judge_results

        # Handle F1 metric (default)
        results = {
            "overall": {},
            "by_category": {},
            "details": [],
        }

        total_f1 = 0
        category_stats = {}

        for item in data_with_model_predictions:
            try:
                prediction = item.get("model_prediction", "").strip()
            except AttributeError:
                # Handle case where model_prediction is None
                prediction = ""

            try:
                reference = item.get("reference", "").strip()
            except AttributeError:
                # Handle case where reference is None
                reference = ""

            category = item.get("label", "unknown")

            # Calculate character-level F1 score
            f1_score = self.calculate_f1_score(reference, prediction)
            total_f1 += f1_score

            # Track by category
            if category not in category_stats:
                category_stats[category] = {"f1_total": 0.0, "count": 0}
            category_stats[category]["f1_total"] += f1_score
            category_stats[category]["count"] += 1

            results["details"].append({
                "sample_id": item.get("sample_id", ""),
                "category": category,
                "question": item.get("question", ""),
                "reference": reference,
                "prediction": prediction,
                "f1": f1_score,
                "src_lang": "eng",
                "tgt_lang": "eng",
            })

        # Calculate overall scores
        total_samples = len(data_with_model_predictions)
        if total_samples > 0:
            results["overall"]["f1"] = round(total_f1 / total_samples, 1)
            results["f1"] = round(total_f1 / total_samples, 1)  # For compatibility

        # Calculate category-wise scores
        for category, stats in category_stats.items():
            results["by_category"][category] = {
                "f1": round(stats["f1_total"] / stats["count"], 1) if stats["count"] > 0 else 0.0,
                "count": stats["count"]
            }

        return results
