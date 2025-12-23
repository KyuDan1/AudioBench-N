#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Gemma-3-27b-it as Judge for CCFQA Benchmark
Based on CCFQA official evaluation method (vllm_eval.py)

This evaluator uses Gemma-3-27b-it model to determine if the model's response
correctly answers the question with a binary yes/no judgment.
"""

import os
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI


# Language mapping for CCFQA (from audio_language_ccfqa.py)
LANGUAGE_MAPPING = {
    'ara': 'Arabic', 'arz': 'Arabic', 'ben': 'Bengali', 'ces': 'Czech',
    'deu': 'German', 'eng': 'English', 'spa': 'Spanish', 'fas': 'Persian',
    'pes': 'Persian', 'fra': 'French', 'heb': 'Hebrew', 'hin': 'Hindi',
    'ind': 'Indonesian', 'ita': 'Italian', 'jpn': 'Japanese', 'khm': 'Khmer',
    'kor': 'Korean', 'lao': 'Lao', 'msa': 'Malay', 'zsm': 'Malay',
    'mya': 'Burmese', 'nld': 'Dutch', 'pol': 'Polish', 'por': 'Portuguese',
    'rus': 'Russian', 'tha': 'Thai', 'tgl': 'Tagalog', 'tur': 'Turkish',
    'urd': 'Urdu', 'vie': 'Vietnamese', 'zho': 'Chinese', 'cmn': 'Chinese',
    'yue': 'Traditional Chinese', 'ceb': 'Cebuan', 'oci': 'Occitan',
    'mon': 'Mongolian', 'khk': 'Mongolian',
}


def _build_failure_detail(question, reference, prediction, src_lang, tgt_lang, message):
    """Build error response when evaluation fails"""
    return {
        'question': question,
        'reference': reference,
        'model_prediction': prediction,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'judge_response': f"Error: {message}",
        'acc': 'no',
        'success': 0,
    }


def gemma3_27b_as_judge_one_sample(args):
    """
    Evaluate one sample using Gemma-3-27b-it model
    Based on CCFQA official prompt (vllm_eval.py:78-88)
    """
    question, reference, prediction, src_lang, tgt_lang = args

    # CCFQA official LLM judge prompt (from vllm_eval.py:78-88)
    prompt_content = (
        f"Based on the {LANGUAGE_MAPPING[src_lang]} question and "
        f"{LANGUAGE_MAPPING[tgt_lang]} reference and response, "
        f"determine if the response correctly answers the question in {LANGUAGE_MAPPING[tgt_lang]}. "
        f"If correct, return \"yes\"; otherwise, return \"no\", without any additional explanation.\n\n"
        f"Question: {question}\n"
        f"{LANGUAGE_MAPPING[tgt_lang]} reference: {reference}\n"
        f"{LANGUAGE_MAPPING[tgt_lang]} response: {prediction}"
    )

    # Connect to vLLM server running Gemma-3-27b-it
    openai_api_base = "http://10.169.39.24:10130/v1"
    openai_api_key = "EMPTY"

    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    except Exception as exc:
        return _build_failure_detail(question, reference, prediction, src_lang, tgt_lang,
                                    f"client init failed: {exc}")

    try:
        models = client.models.list()
        if not getattr(models, "data", None):
            raise RuntimeError("no models returned")
        model = models.data[0].id
    except Exception as exc:
        return _build_failure_detail(question, reference, prediction, src_lang, tgt_lang,
                                    f"models.list failed: {exc}")

    try:
        # Use chat completions API - vLLM will apply correct Gemma-3 chat template
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=64,
            temperature=0,
            top_p=1,
            n=1,
        )
        output = completion.choices[0].message.content.strip()
    except Exception as exc:
        return _build_failure_detail(question, reference, prediction, src_lang, tgt_lang,
                                    f"completion failed: {exc}")

    # Parse response (should be "yes" or "no")
    output_lower = output.lower().replace("\n", "")

    if "yes" in output_lower:
        acc_result = 'yes'
        success = 1
    elif "no" in output_lower:
        acc_result = 'no'
        success = 1
    else:
        acc_result = 'other'
        success = 0

    sample_detail = {
        'question': question,
        'reference': reference,
        'model_prediction': prediction,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'judge_response': output,
        'acc': acc_result,
        'success': success,
    }

    return sample_detail


def gemma3_27b_as_judge_ccfqa(model_path, input_data):
    """
    Evaluate CCFQA predictions using Gemma-3-27b-it as judge

    Args:
        model_path: Path to Gemma-3-27b-it model (not used, kept for API compatibility)
        input_data: Tuple of (questions, references, predictions, src_langs, tgt_langs)

    Returns:
        judge_results: Dictionary with 'acc_score' (percentage of "yes") and 'success_rate'
        all_details: List of per-sample evaluation details
    """

    questions, references, predictions, src_langs, tgt_langs = input_data

    # Use multiprocessing for parallel evaluation
    num_processes = min(8, len(questions))

    with Pool(processes=num_processes) as pool:
        all_details = list(
            tqdm(
                pool.imap(
                    gemma3_27b_as_judge_one_sample,
                    zip(questions, references, predictions, src_langs, tgt_langs)
                ),
                total=len(questions),
                desc="Gemma-3-27b-it Judge"
            )
        )

    # Calculate ACC score (percentage of "yes")
    yes_count = sum(1 for detail in all_details if detail['acc'] == 'yes')
    no_count = sum(1 for detail in all_details if detail['acc'] == 'no')
    other_count = sum(1 for detail in all_details if detail['acc'] == 'other')

    total_valid = yes_count + no_count + other_count
    acc_score = (yes_count / total_valid * 100) if total_valid > 0 else 0.0

    success_rate = sum(detail['success'] for detail in all_details) / len(all_details)

    judge_results = {
        'acc_score': round(acc_score, 1),  # Percentage with 1 decimal
        'success_rate': success_rate,
        'yes_count': yes_count,
        'no_count': no_count,
        'other_count': other_count,
        'total_samples': len(all_details)
    }

    return judge_results, all_details
