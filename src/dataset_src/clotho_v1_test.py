import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from huggingface_hub import hf_hub_download

from dataset_src.eval_methods.metrics import CocoTokenizer
from dataset_src.eval_methods.bleu import Bleu
from dataset_src.eval_methods.meteor import Meteor
from dataset_src.eval_methods.rouge import Rouge
from dataset_src.eval_methods.cider import Cider
from dataset_src.eval_methods.spice import Spice

caption_instructions = [
    "Listen to the clip and write a concise caption describing what you hear.",
    "Please provide a natural-language description of this audio scene.",
    "Describe the key events happening in the provided audio recording.",
    "Summarize the audible content of this sound clip in one sentence.",
    "Write an audio caption that captures the important sounds in this clip."
]

class clotho_v1_test_dataset(object):

    def __init__(self, raw_data, number_of_samples, repo_id="Kyudan/clotho_v1_test"):
        if not isinstance(raw_data, (list, tuple)):
            raise TypeError("Expected raw_data to be a sequence of Hugging Face dataset file paths.")

        file_list = list(raw_data)
        if number_of_samples != -1:
            rng = random.Random(42)
            rng.shuffle(file_list)
            file_list = file_list[:min(number_of_samples, len(file_list))]

        self.repo_id = repo_id
        self.file_list = file_list
        self.prompt = caption_instructions
        logging.info('Number of samples: {}'.format(len(self.file_list)))

        # === ADDED: Load ground truths from CSV ===
        try:
            # Assumes this script is in 'dataset_src' and the CSV is also in 'dataset_src'
            csv_path = Path(__file__).resolve().parent / 'clotho_captions_evaluation.csv'

            if not csv_path.exists():
                # Fallback to the hardcoded path from your reference if relative path fails
                user_csv_path = Path('AudioBench-application/src/dataset_src/clotho_captions_evaluation.csv')
                if user_csv_path.exists():
                    csv_path = user_csv_path
                else:
                    logging.info("Local ground-truth CSV not found. Downloading from Hugging Face hub.")
                    download_path = hf_hub_download(
                        repo_id=self.repo_id,
                        filename='clotho_captions_evaluation.csv',
                        repo_type='dataset'
                    )
                    csv_path = Path(download_path)

            captions_df = pd.read_csv(csv_path)
            self.gt_dict = self._prepare_ground_truths(captions_df)
            logging.info(f"Loaded ground truths for {len(self.gt_dict)} files from {csv_path}")
        except Exception as e:
            logging.error(f"Failed to load or process ground truth CSV: {e}")
            self.gt_dict = {}
        # === END ADDED SECTION ===


    # === ADDED: Helper method to process CSV ===
    def _prepare_ground_truths(self, captions_df):
        """
        Prepare ground truth captions from the development CSV.
        (Adapted from your reference script)
        """
        gt_dict = {}
        
        # Check for expected columns
        if 'file_name' not in captions_df.columns or 'caption_1' not in captions_df.columns:
            logging.error(f"CSV columns are incorrect: {captions_df.columns.tolist()}. Expected 'file_name' and 'caption_1' ... 'caption_5'.")
            return {}
            
        for idx, row in captions_df.iterrows():
            file_name = row['file_name']
            
            captions = []
            for i in range(1, 6):
                caption_col = f'caption_{i}'
                if caption_col in row and pd.notna(row[caption_col]):
                    captions.append(str(row[caption_col]))
            
            gt_dict[file_name] = captions
        
        return gt_dict
    # === END ADDED HELPER ===


    def _extract_metric_summary(self, scores, metric_name):
        """Map overall score dict to requested metric summary."""
        if not scores:
            return None

        metric = metric_name.lower()
        if metric == 'bleu':
            return {k: scores[k] for k in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"] if k in scores}
        if metric == 'meteor':
            return scores.get("METEOR")
        if metric in ('rouge', 'rouge_l'):
            return scores.get("ROUGE_L")
        if metric in ('cider', 'cider_d'):
            return scores.get("CIDEr")
        if metric == 'spice':
            return scores.get("SPICE")
        if metric == 'spider':
            # Prefer directly-computed SPIDEr if it already exists.
            spider_score = scores.get("SPIDEr")
            if spider_score is not None:
                return spider_score

            # Fall back to recomputing from CIDEr / SPICE when available.
            cider = scores.get("CIDEr")
            spice = scores.get("SPICE")
            if cider is not None and spice is not None:
                return (cider + spice) / 2.0

            # If only one component is available, return a conservative estimate.
            if cider is not None:
                return cider / 2.0
            if spice is not None:
                return spice / 2.0
            return None
        return None


    def prepare_model_input(self):
        input_data = []

        if not self.gt_dict:
            logging.error("Ground truth dictionary is empty. Cannot prepare model inputs.")
            return []

        for file_path in tqdm(self.file_list, desc="Preparing model inputs"):
            file_name = os.path.basename(file_path)
            gt = self.gt_dict.get(file_name)

            if not gt:
                logging.warning(f"No ground truth found for file: {file_name}. Skipping.")
                continue

            try:
                local_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=file_path,
                    repo_type='dataset'
                )
                audio_array, sampling_rate = sf.read(local_path, always_2d=False)
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                audio_array = np.asarray(audio_array, dtype=np.float32)
            except Exception as err:
                logging.warning(f"Failed to load audio for {file_path}: {err}")
                continue

            instruction = random.choice(self.prompt)

            input_data.append({
                "audio": {
                    "array": audio_array,
                    "sampling_rate": sampling_rate
                },
                "source": "clotho_v1_test",
                "gt": gt,
                "instruction": instruction,
                "task_type": "caption",
                "file_name": file_name
            })
            # === END MODIFIED SECTION ===

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        if input_data:
            logging.info(random.sample(input_data, 1)[0])
        else:
            logging.info("No data prepared (or no GTs found).")
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data

    def format_model_predictions(self, input_data, model_predictions):
        data_with_model_predictions = []
        for sample, prediction in zip(input_data, model_predictions):
            new_sample = sample.copy()
            # Remove audio to reduce file size
            if 'audio' in new_sample:
                audio_obj = new_sample['audio']
                sampling_rate = None
                array_shape = None

                # Hugging Face AudioDecoder implements __getitem__ for 'array' and 'sampling_rate'
                if isinstance(audio_obj, dict):
                    sampling_rate = audio_obj.get('sampling_rate')
                    array = audio_obj.get('array')
                else:
                    try:
                        sampling_rate = audio_obj['sampling_rate']
                    except Exception:
                        sampling_rate = None
                    try:
                        array = audio_obj['array']
                    except Exception:
                        array = None

                if array is not None:
                    array_shape = getattr(array, 'shape', None)

                new_sample['audio_object_details'] = {
                    'sampling_rate': sampling_rate,
                    'array_shape': array_shape
                }
                new_sample.pop('audio', None)
            new_sample['model_prediction'] = prediction
            data_with_model_predictions.append(new_sample)
            
        return data_with_model_predictions

    def compute_score(self, data_with_model_predictions, metrics=None):
        # Group by source
        results_dict = {}
        for item in data_with_model_predictions:
            source = item["source"]
            results_dict.setdefault(source, []).append(item)

        # Compute scores for each source
        all_scores = {}
        for source in results_dict:
            refs, hyps = [], []
            results_list = results_dict[source]
            for result in results_list:
                gt = result["gt"]
                response = result["model_prediction"]
                refs.append(gt)
                hyps.append(response)
            
            logging.info(f"Computing scores for source: {source} ({len(refs)} samples)")
            score_dict = self._compute_caption_metrics(refs, hyps)
            all_scores[source] = score_dict
            logging.info(f"source: {source}\tcnt: {len(refs)}\tRes: {score_dict}")

        # Compute overall scores
        all_refs = [item["gt"] for item in data_with_model_predictions]
        all_hyps = [item["model_prediction"] for item in data_with_model_predictions]
        
        overall_scores = {}
        if all_refs and all_hyps:
            logging.info(f"Computing overall scores ({len(all_refs)} samples)")
            overall_scores = self._compute_caption_metrics(all_refs, all_hyps)
        else:
            logging.warning("No data to compute overall scores.")

        results = {
            'overall': overall_scores,
            'by_source': all_scores,
            'details': data_with_model_predictions[:20]  # First 20 samples
        }

        if metrics:
            if not overall_scores:
                metric_summary = {}
            else:
                metric_summary = self._extract_metric_summary(overall_scores, metrics)
                if metric_summary is None:
                    available = [#"bleu", 
                                #"meteor", 
                                "rouge_l", 
                                "cider", 
                                "spice",
                                "spider"]
                    raise ValueError(f"Unsupported or unavailable metric '{metrics}'. Available metrics: {available}")
            results[metrics] = metric_summary

        return results

    def _compute_caption_metrics(self, gts, res):
        """Compute caption evaluation metrics"""
        preds_str = res
        references = gts
        
        # Ensure gts is in the correct format: list[list[str]]
        # gts is currently list[list[str]] (one list of refs per sample)
        # res is currently list[str] (one pred per sample)
        
        # The tokenizer needs gts as {idx: [ref1, ref2,...]} and res as {idx: [pred]}
        gts_dict = {i: gts[i] for i in range(len(gts))}
        res_dict = {i: [res[i]] for i in range(len(res))}
        
        tokenizer = CocoTokenizer(preds_str, references)
        tokenized_res, tokenized_gts = tokenizer.tokenize()

        scorers = [
            #(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #(Meteor(), "METEOR"),
            #(Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]
        
        f_res = {}
        for scorer, method in scorers:
            try:
                score, scores = scorer.compute_score(tokenized_gts, tokenized_res)
                if type(method) == list:
                    for sc, m in zip(score, method):
                        f_res[m] = sc
                else:
                    f_res[method] = score
            except Exception as e:
                logging.error(f"Error computing metric {method}: {e}")

        
        if 'CIDEr' in f_res and 'SPICE' in f_res:
            f_res["SPIDEr"] = (f_res['CIDEr'] + f_res['SPICE']) / 2.
        else:
            f_res["SPIDEr"] = 0.0
            
        return f_res
