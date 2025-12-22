import logging
import os
from io import BytesIO
import pandas as pd
import soundfile as sf
from typing import Any, Tuple

INSTRUCTION = "Which of the following acoustic scenes best describes the audio?"
CHOICES = [
    "Bus",
    "Cafe",
    "Car",
    "CrowdedIndoor",
    "Elevator",
    "Kitchen",
    "Park",
    "ResidentialArea",
    "Restaurant",
    "Restroom",
    "Street",
    "Subway",
    "SubwayStation",
]

class cochlscene_test_dataset(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        self.samples_metadata = []
        logging.info('Number of samples: {}'.format(len(self.raw_data)))

    def _resolve_audio_payload(self, context: Any) -> Tuple[Any, int]:
        """
        Convert the multimodal context into (audio_array, sampling_rate).

        Qwen-Audio models expose AudioDecoder objects (torchcodec) during
        evaluation, so we need to gracefully handle objects that are not
        dict-like but still support `__getitem__` for array / sampling rate.
        """

        if context is None:
            raise KeyError("Audio context is missing from the sample.")

        audio_array = None
        sampling_rate = None
        audio_path = None
        audio_bytes = None

        if isinstance(context, dict):
            audio_array = context.get('array')
            sampling_rate = context.get('sampling_rate')
            audio_path = context.get('path')
            audio_bytes = context.get('bytes')
        else:
            if hasattr(context, '__getitem__'):
                try:
                    audio_array = context['array']
                    sampling_rate = context['sampling_rate']
                except Exception:
                    audio_array = None
                    sampling_rate = None

            if hasattr(context, 'get'):
                audio_path = audio_path or context.get('path')
                audio_bytes = audio_bytes or context.get('bytes')

            encoded_payload = getattr(context, '_hf_encoded', None)
            if isinstance(encoded_payload, dict):
                audio_path = audio_path or encoded_payload.get('path')
                audio_bytes = audio_bytes or encoded_payload.get('bytes')

            metadata = getattr(context, 'metadata', None)
            if metadata is not None and getattr(metadata, 'path', None):
                audio_path = audio_path or metadata.path

            if isinstance(context, str):
                audio_path = audio_path or context

        if audio_array is not None and sampling_rate is None:
            sampling_rate = getattr(context, 'sampling_rate', None)

        if audio_array is not None and sampling_rate is not None:
            return audio_array, sampling_rate

        if audio_path and os.path.exists(audio_path):
            return sf.read(audio_path)

        if audio_bytes:
            return sf.read(BytesIO(audio_bytes))

        raise ValueError(f"Unable to resolve audio data from context type {type(context)}.")


    def prepare_model_input(self):

        def data_generator():
            for idx, sample in enumerate(self.raw_data):
                context = sample.get('context', sample.get('audio'))
                audio_array, sampling_rate = self._resolve_audio_payload(context)
                audio = {
                    "array": audio_array,
                    "sampling_rate": sampling_rate,
                }
                instruction = 'Question:\n' + INSTRUCTION + '\nChoices:\n' + " ".join(CHOICES)
                label_value = sample['label']
                label_feature = None
                features = getattr(self.raw_data, "features", None)
                if features and "label" in features:
                    label_feature = features["label"]
                if isinstance(label_value, int) and label_feature and hasattr(label_feature, "int2str"):
                    reference = label_feature.int2str(label_value)
                else:
                    reference = label_value

                metadata = {
                    "instruction": instruction,
                    "reference"  : reference,
                    "task_type"  : "Audio-Understanding-Reasoning",
                }

                self.samples_metadata.append(metadata)

                if idx == 0:
                    logging.info('\n=  =  =  Dataset Sample  =  =  =')
                    logging.info(metadata)
                    logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

                yield {
                        "audio"      : audio,
                        "instruction": instruction,
                        "reference"  : reference,
                        "task_type"  : "Audio-Understanding-Reasoning",
                      }

        return data_generator()
        

    def format_model_predictions(self, input_data, model_predictions):

        data_with_model_predictions = []
        for sample_metadata, prediction in zip(self.samples_metadata, model_predictions):
            new_sample = sample_metadata.copy()
            new_sample['model_prediction'] = prediction
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions


    def compute_score(self, data_with_model_predictions, metrics=None):
        
        questions   = [item["instruction"] for item in data_with_model_predictions]
        references  = [item["reference"] for item in data_with_model_predictions]
        predictions = [item["model_prediction"] for item in data_with_model_predictions]

        if metrics == 'llama3_70b_judge':
            from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge_binary
            llama3_70b_judge_results, all_details = llama3_70b_as_judge_binary("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions])

            # Grouping the data by 'task' and calculating the average rate_score for each task
            # for result, sample_other_attributes in zip(all_details, self.raw_data['other_attributes']):
            #     result['task'] = sample_other_attributes['task']
            # df = pd.DataFrame(all_details)
            # task_scores = df.groupby('task')['rate_score'].mean().to_dict()
            
            return {'llama3_70b_judge': llama3_70b_judge_results, 'details': all_details}


        if metrics == 'string_match':
            choices = [item for item in CHOICES]
            from dataset_src.eval_methods.string_match import mmau_string_match
            string_match_results, all_details = mmau_string_match([questions, references, predictions, choices])

            # Grouping the data by 'task' and calculating the average rate_score for each task


            return {'string_match': string_match_results, 'details': all_details}


        # elif metrics == 'llama3_8b_judge':
        #     from dataset_src.eval_methods.eval_llama3_8b import llama3_8b_as_judge
        #     llama3_8b_judge_results = llama3_8b_as_judge("../prepared_models/Meta-Llama-3-8B-Instruct-hf", [questions, references, predictions])
        #     return {'llama3_8b_judge': llama3_8b_judge_results}
        
        # elif metrics == 'prometheus2_judge':
        #     from dataset_src.eval_methods.eval_prometheus2 import prometheus2_as_judge
        #     prometheus2_judge_results = prometheus2_as_judge("../prepared_models/prometheus-7b-v2.0", [questions, references, predictions])
        #     return {'prometheus2_judge': prometheus2_judge_results}
        
        elif metrics == 'gpt4o_judge':
            from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
            gpt4o_judge_results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])


            # Grouping the data by 'task' and calculating the average rate_score for each task
            for result, sample_other_attributes in zip(all_details, self.raw_data['other_attributes']):
                result['task'] = sample_other_attributes['task']
            df = pd.DataFrame(all_details)
            task_scores = df.groupby('task')['rate_score'].mean().to_dict()

            return {'gpt4o_judge': gpt4o_judge_results, "task_scores": task_scores, 'details': all_details}
        
        # elif metrics == 'gpt4o_judge_binary':
        #     from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
        #     gpt4o_judge_binary_results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])
        #     return {'gpt4o_judge_binary': gpt4o_judge_binary_results, 'details': all_details}

        else:
            raise ValueError("Invalid metrics: {}".format(metrics))
