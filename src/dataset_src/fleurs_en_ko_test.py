import random
import logging

from jiwer import compute_measures, wer

import evaluate

from dataset_src.text_normalizer.preprocess_text import preprocess_text_asr

# Ten instructions, random select one for each sample
st_instructions = [
    "Listen to the speech clip and translate it into Korean.",
    "Play the speech recording and translate it to Korean.",
    "Hear the audio clip and convert it to Korean.",
    "Listen to the speech and translate it into Korean.",
    "Play the recorded speech and provide a translation in Korean.",
    "Hear the speech audio and translate it into Korean.",
    "Listen to the speech audio and translate it to Korean.",
    "Play the speech clip and translate it into Korean.",
    "Listen to the speech recording and translate it into Korean.",
    "Hear the audio clip and translate it to Korean.",
    "Listen to the speech and provide a translation in Korean.",
    "Play the audio of the speech and translate it into Korean.",
    "Hear the speech and convert it to Korean.",
    "Listen to the speech audio clip and translate it to Korean.",
    "Play the speech recording and translate it into Korean.",
    "Listen to the speech clip and provide a translation in Korean.",
    "Hear the recorded speech and translate it to Korean.",
    "Play the audio speech and translate it into Korean.",
    "Listen to the speech and translate it to Korean.",
    "Hear the audio of the speech and translate it into Korean."
]

class fleurs_en_ko_test_dataset(object):

    def __init__(self, raw_data, raw_data_en, raw_data_ko, number_of_samples):

        raw_data_en_sorted = raw_data_en.sort('id')
        raw_data_ko_sorted = raw_data_ko.sort('id')
        raw_data_sorted = raw_data.sort('id') 

        if number_of_samples != -1:
            self.raw_data = raw_data_sorted.select(range(number_of_samples))
            self.raw_data_en = raw_data_en_sorted.select(range(number_of_samples))
            self.raw_data_ko = raw_data_ko_sorted.select(range(number_of_samples))
        else:
            self.raw_data = raw_data_sorted
            self.raw_data_en = raw_data_en_sorted
            self.raw_data_ko = raw_data_ko_sorted
        
        self.prompt   = st_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample_en, sample_ko in zip(self.raw_data_en, self.raw_data_ko):
            audio       = sample_en['audio']
            source      = sample_ko['transcription']
            reference   = sample_ko['transcription']
            instruction = random.choice(self.prompt)

            input_data.append({
                                "audio"      : audio,
                                "source"     : source,
                                "instruction": instruction,
                                "reference"  : reference,
                                "task_type"  : "S2TTen2ko" # ST-EN-ZH to ST-EN-KO changed
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
        if metrics not in (None, 'bleu', 'comet'):
            raise ValueError(f"Unsupported metric: {metrics}. Supported metrics: 'bleu' or 'comet'")

        predictions = []
        references  = []
        sources = []
        for item in data_with_model_predictions:
            # model_prediction = preprocess_text_asr(item["model_prediction"])
            # answer           = preprocess_text_asr(item["answer"])
            model_prediction = item.get("model_prediction")
            answer           = item.get("reference")
            source           = item.get("source")

            if not isinstance(model_prediction, str):
                model_prediction = str(model_prediction)
            if not isinstance(answer, str):
                answer = str(answer)
            if not isinstance(source, str):
                source = str(source)

            if len(model_prediction) == 0: model_prediction = "empty"
            if len(answer) == 0: answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)
            sources.append(source)

        scores = {}
        if metrics in (None, 'bleu'):
            sacrebleu = evaluate.load("sacrebleu")
            # Updated to flores101 tokenizer (Thanks Chenyang Lv)
            # results = sacrebleu.compute(predictions=predictions, references=references, tokenize='13a')
            bleu_score = sacrebleu.compute(predictions=predictions, references=references, tokenize='ko-mecab')
            scores["bleu"] = bleu_score['score']

        if metrics in (None, 'comet'):
            comet_metric = evaluate.load('comet')
            comet_score = comet_metric.compute(predictions=predictions, references=references, sources=sources)
            if "mean_score" in comet_score:
                scores["comet"] = comet_score["mean_score"]
            elif "system_score" in comet_score:
                scores["comet"] = comet_score["system_score"]
            else:
                raise KeyError(f"COMET output missing mean_score/system_score keys: {list(comet_score.keys())}")

        return scores
