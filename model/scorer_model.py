#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for sequence classification."""


import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import datasets
import numpy as np
import json

from difflib import SequenceMatcher#导入库

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    AutoModelForSequenceClassification,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging
from t5_pegasus_tokenizer import T5PegasusTokenizer

hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()#引用ratio方法，返回序列相似性的度量


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_file: str = field(default=None, metadata={"help": "The path of the training file"})
    dev_file: Optional[str] = field(default=None, metadata={"help": "The path of the development file"})
    test_file: Optional[str] = field(default=None, metadata={"help": "The path of the test file"})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cons_model_name_or_path: str = field(
        metadata={"help": "Path to constraint model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cons_config_name: Optional[str] = field(
        default=None, metadata={"help": "constraint model config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(training_args)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(f"Training/evaluation parameters {training_args}")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if "chinese_t5_pegasus" in model_args.model_name_or_path:
        tokenizer = T5PegasusTokenizer.from_pretrained(model_args.model_name_or_path)
        # tokenizer.bos_token = tokenizer.cls_token
        # tokenizer.eos_token = tokenizer.sep_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.cons_model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

    constrant_tokenizer = AutoTokenizer.from_pretrained(
            model_args.cons_model_name_or_path,
            cache_dir=model_args.cache_dir,
        )



    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
    )

    constrant_config = AutoConfig.from_pretrained(
        model_args.cons_config_name if model_args.cons_config_name else model_args.cons_model_name_or_path,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
    )



    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    constrant_model = BertForTokenClassification.from_pretrained(
        model_args.cons_model_name_or_path,
        config=constrant_config,
    )

    tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_0>", "<extra_1>", "<extra_2>", "<extra_3>",
                                                                "<extra_4>", "<extra_5>", "<extra_6>", "<extra_7>"]})
    model.resize_token_embeddings(len(tokenizer))

    def get_tfds(
            train_file: str,
            eval_file: str,
            test_file: str,
            tokenizer: PreTrainedTokenizer,
            max_seq_length: Optional[int] = None,
    ):
        data_files = {}
        extension = 'json'
        if train_file is not None:
            data_files["train"] = train_file
        if eval_file is not None:
            data_files["validation"] = eval_file
        if test_file is not None:
            data_files["test"] = test_file

        ds = datasets.load_dataset(extension, data_files=data_files)
        features_name = ds["train"].column_names[:-1]
        label_name = ds["train"].column_names[-1]
        label_list = ['0', '1']
        label2id = {label: i for i, label in enumerate(label_list)}


        def preprocess_function(examples):
            inputs = examples[features_name[0]]
            # inputs = preprocess_input(inputs)
            # targets = [PenmanSerializer(target).get_graph_string() for target in targets]
            # targets = [x.replace('(', '<extra_id_0>').replace(')', '<extra_id_1>') for x in targets]
            token_type = []
            for i in range(len(inputs)):
                model_inputs = constrant_tokenizer(inputs[i], padding=False, return_tensors="pt")
                outputs = constrant_model(**model_inputs).logits
                type = [1 if outputs[0][j, 1] > 0 else 0 for j in range(len(outputs[0]))]
                type.insert(0, 0)
                type = type + [0 for _ in range(max_seq_length-len(type))]
                token_type.append(type)

            model_inputs = tokenizer(inputs, max_length=max_seq_length, padding="max_length", truncation=True)
            model_inputs["label"] = [label2id[item] for item in examples[label_name]]
            # model_inputs['token_type_ids'] = token_type

            return model_inputs

        train_dataset = ds["train"].map(
            preprocess_function,
            remove_columns=ds["train"].column_names,
            batched=True,
        )

        eval_dataset = ds["validation"].map(
            preprocess_function,
            remove_columns=ds["validation"].column_names,
            batched=True,
        )

        test_dataset = ds["test"].map(
            preprocess_function,
            remove_columns=ds["test"].column_names,
            batched=True,
        )

        return train_dataset, eval_dataset, test_dataset, label2id

    train_dataset, eval_dataset, test_dataset, label2id = get_tfds(
        train_file=data_args.train_file,
        eval_file=data_args.dev_file,
        test_file=data_args.test_file,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)

        return {"acc": (preds == p.label_ids).mean()}

    # Initialize our Trainer
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(training_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        # train_pred_results = trainer.predict(
        #     train_dataset,
        #     metric_key_prefix="train",
        #     max_length=data_args.val_max_target_length,
        #     num_beams=data_args.num_beams,
        # )

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")

        eval_results = trainer.predict(
            eval_dataset,
            metric_key_prefix="eval"
        )

        output_test_preds_file = os.path.join(training_args.output_dir, "eval_preds_seq2seq.txt")

        input = []
        with open(data_args.dev_file, "r") as r:
            data = r.readlines()
        for line in data:
            line = json.loads(line)
            input.append(line['text'])

        with open(output_test_preds_file, "w") as writer:
            for i in range(len(eval_results.predictions)):
                writer.write('input:' + input[i] + '\n')
                preds = np.argmax(eval_results.predictions[i])
                writer.write('score:' + str(eval_results.predictions[i][1]) + '\n')
                writer.write('label:' + str(preds) + '\n')

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")

            for key, value in result.items():
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

            results.update(result)

    if training_args.do_predict:
        logger.info("*** Test ***")
        result = trainer.evaluate()

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test"
        )

        output_test_preds_file = os.path.join(training_args.output_dir, "test_preds_seq2seq.txt")

        unlabel_file = os.path.join(training_args.output_dir, "test_label.txt")

        input = []
        with open(data_args.test_file, "r") as r:
            data = r.readlines()
        for line in data:
            line = json.loads(line)
            input.append(line['text'])

        with open(output_test_preds_file, "w") as writer:
            for i in range(len(test_results.predictions)):
                writer.write('input:' + input[i] + '\n')
                preds = np.argmax(test_results.predictions[i])
                writer.write('score:' + str(test_results.predictions[i][1]) + '\n')
                writer.write('label:' + str(preds) + '\n')

            with open(unlabel_file, "w") as writer:
                for i in range(len(test_results.predictions)):
                    preds = np.argmax(test_results.predictions[i])
                    if preds == 1:
                        writer.write('input:' + input[i] + '\n')
                        writer.write('label:' + str(preds) + '\n')

            results.update(result)

    return results


if __name__ == "__main__":
    main()