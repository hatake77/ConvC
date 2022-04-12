#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License atcompute_metrics
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import torch
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    BertForTokenClassification,
    AutoModelForSequenceClassification,
    set_seed,
    Trainer,
    MT5ForConditionalGeneration,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import re
import random
import math
from stanfordcorenlp import StanfordCoreNLP

from mbr_decoding import mbr_decoding

from seq2seq.t5_pegasus_tokenizer import T5PegasusTokenizer


from difflib import SequenceMatcher#导入库

logger = logging.getLogger(__name__)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()#引用ratio方法，返回序列相似性的度量

meaningful = ['NN', 'NR', 'VV']

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
    score_model_name_or_path: str = field(
        metadata={"help": "Path to score model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cons_config_name: Optional[str] = field(
        default=None, metadata={"help": "constraint model config name or path if not the same as model_name"}
    )
    score_config_name: Optional[str] = field(
        default=None, metadata={"help": "score model config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # !!! must use non-fast version
    # fast: "<extra_id_0> <extra_id_1>" -> [32099, 3, 32098, 1]
    # non-fast: "<extra_id_0> <extra_id_1>" -> [32099, 32098, 1]
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
                    "pegasus) or translation (or translation_{xx}_to_{yy})."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    source_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if not self.task.startswith("summarization") and not self.task.startswith(
                "translation") and not self.task.startswith('event') and not self.task.startswith('amr'):
            raise ValueError(
                "`task` should be summarization, summarization_{dataset}, translation or translation_{xx}_to_{yy}."
            )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length



summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

event_extraction_name_mapping = {
    "ace2005": ("text", "event")
}

amr_extraction_name_mapping = {
    "ace2005": ("text", "amr")
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)
    print(training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        mirror='tuna',
    )

    constrant_config = AutoConfig.from_pretrained(
        model_args.cons_config_name if model_args.cons_config_name else model_args.cons_model_name_or_path,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
    )

    score_config = AutoConfig.from_pretrained(
        model_args.score_config_name if model_args.score_config_name else model_args.score_model_name_or_path,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
    )

    # !!!
    config.max_length = data_args.max_target_length
    constrant_config.max_length = data_args.max_target_length

    if "chinese_t5_pegasus" in model_args.model_name_or_path:
        tokenizer = T5PegasusTokenizer.from_pretrained(model_args.model_name_or_path)
        # tokenizer.bos_token = tokenizer.cls_token
        # tokenizer.eos_token = tokenizer.sep_token
        print('chinese!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            mirror='tuna',
        )

    constrant_tokenizer = AutoTokenizer.from_pretrained(
            model_args.cons_model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

    to_remove_token_list = list()
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]


# AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        mirror='tuna',
    )

    constrant_model = BertForTokenClassification.from_pretrained(
        model_args.cons_model_name_or_path,
        config=constrant_config,
    )

    score_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.score_model_name_or_path,
        config=score_config,
    )

    # model = MT5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    if tokenizer.encode("<extra_0> <extra_1>") != [32099, 32098, 1]:
        # For non-t5 tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_0>", "<extra_1>", "<extra_2>", "<extra_3>", "<extra_4>", "<extra_5>", "<extra_6>", "<extra_7>"]})
        model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if data_args.task.startswith("translation"):
        if data_args.source_lang is not None:
            tokenizer.src_lang = data_args.source_lang
        if data_args.target_lang is not None:
            tokenizer.tgt_lang = data_args.target_lang

    ### Start Code for Event Extraction
    # if data_args.task.startswith("event"):
    #     decoding_type_schema = EventSchema.read_from_file(data_args.event_schema)
    # else:
    decoding_type_schema = None
    ### End Code for Event Extraction

    # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
    # them all).
    source_lang, target_lang, text_column, summary_column = None, None, None, None

    if data_args.task.startswith("summarization"):
        # Get the column names for input/target.
        dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
        if data_args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = data_args.text_column
        if data_args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = data_args.summary_column
    ### Start Code for Event Extraction
    elif data_args.task.startswith("event"):
        dataset_columns = event_extraction_name_mapping.get(data_args.dataset_name, None)
        if data_args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = data_args.text_column
        if data_args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = data_args.summary_column

    else:
        # Get the language codes for input/target.
        lang_search = re.match("translation_([a-z]+)_to_([a-z]+)", data_args.task)
        if data_args.source_lang is not None:
            source_lang = data_args.source_lang.split("_")[0]
        else:
            assert (
                    lang_search is not None
            ), "Provide a source language via --source_lang or rename your task 'translation_xx_to_yy'."
            source_lang = lang_search.groups()[0]

        if data_args.target_lang is not None:
            target_lang = data_args.target_lang.split("_")[0]
        else:
            assert (
                    lang_search is not None
            ), "Provide a target language via --target_lang or rename your task 'translation_xx_to_yy'."
            target_lang = lang_search.groups()[1]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.error(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )


    nlp = StanfordCoreNLP(r'http://localhost', port=9111, lang='zh')

    def all_postive(str, token_label):
        for word in str:
            if word in token_label.keys() and token_label[word] == 0:
                return False
        return True

    def extract_span(inputs):
        all_candidate = []
        for i in range(len(inputs)):
            sentence = inputs[i].replace('[CLS]','').replace('[SEP]','')
            candidate = []
            model_inputs = constrant_tokenizer(inputs[i], padding=False, return_tensors="pt")
            tokenizer_output = constrant_tokenizer(inputs[i], padding=False)["input_ids"]
            outputs = constrant_model(**model_inputs).logits
            token_label = {}
            for j in range(len(tokenizer_output)):
                token_label[constrant_tokenizer.decode(tokenizer_output[j])] = 1 if outputs[0][j, 1] > 0 else 0
            span = ''
            for index in range(len(sentence)):
                if sentence[index] in token_label.keys() and token_label[sentence[index]] == 1:
                    span += sentence[index]
                elif len(span) > 0:
                    candidate.append(span)
                    span = ''
            if len(span) > 0:
                candidate.append(span)
            candidate = list(set(candidate))
            random.shuffle(candidate)
            num = math.floor(random.random() * 4) + 2
            inputs[i] = inputs[i] + '[CLS]' + '[SEP]'.join(candidate[:num])
            all_candidate.append(candidate)
        return inputs, all_candidate


    def add_gold_constraint(inputs, targets):
        all_candidate = []
        for i in range(len(inputs)):
            candidate = []
            pos = nlp.pos_tag(targets[i].replace('[CLS]', '').replace('[SEP]', ''))

            for p in pos:
                if p[1] in meaningful:
                    candidate.append(p[0])

            candidate = list(set(candidate))
            random.shuffle(candidate)
            num = math.floor(random.random()*4) + 2
            inputs[i] = inputs[i] + '[CLS]' + '[SEP]'.join(candidate[:num])
            all_candidate.append(candidate)
        return inputs, all_candidate

    def add_constraint(inputs):
        all_candidate = []
        for i in range(len(inputs)):
            candidate = []
            model_inputs = constrant_tokenizer(inputs[i], padding=False, return_tensors="pt")
            tokenizer_output = constrant_tokenizer(inputs[i], padding=False)["input_ids"]
            outputs = constrant_model(**model_inputs).logits
            token_label = {}
            for j in range(len(tokenizer_output)):
                token_label[constrant_tokenizer.decode(tokenizer_output[j])] = 1 if outputs[0][j, 1] > 0 else 0
            pos = nlp.pos_tag(inputs[i].replace('[CLS]','').replace('[SEP]',''))

            for p in pos:
                if p[1] in meaningful and all_postive(p[0], token_label):
                    candidate.append(p[0])


            candidate = list(set(candidate))
            random.shuffle(candidate)
            num = math.floor(random.random() * 4) + 2
            inputs[i] = inputs[i] + '[CLS]' + '[SEP]'.join(candidate[:num])
            all_candidate.append(candidate)
        return inputs, all_candidate

    def preprocess_function_for_train(examples):
        if data_args.task.startswith("translation"):
            inputs = [ex[source_lang] for ex in examples["translation"]]
            targets = [ex[target_lang] for ex in examples["translation"]]
        else:
            inputs = examples[text_column]
            targets = examples[summary_column]

        inputs,_ = extract_span(inputs)

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def preprocess_function(examples):
        if data_args.task.startswith("translation"):
            inputs = [ex[source_lang] for ex in examples["translation"]]
            targets = [ex[target_lang] for ex in examples["translation"]]
        else:
            inputs = examples[text_column]
            targets = examples[summary_column]

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function_for_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function_for_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function_for_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric_name = "rouge" if data_args.task.startswith("summarization") else "sacrebleu"
    metric = load_metric(metric_name)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        if metric_name == "rouge":
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        else:  # sacrebleu
            labels = [[label] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

        def clean_str(x_str):
            for to_remove_token in to_remove_token_list:
                x_str = x_str.replace(to_remove_token, '')
            return x_str.strip()

        def extract_pos(x_str):
            pos = nlp.pos_tag(x_str)
            return ' '.join([x[1] for x in pos])

        decoded_preds = [clean_str(x) for x in decoded_preds]
        decoded_labels = [clean_str(x) for x in decoded_labels]

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if metric_name == "rouge":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            # result2 = metric.compute(predictions=pos_preds, references=pos_labels)
            result = {"bleu": result["score"]}
        return result

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        # if last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        # elif os.path.isdir(model_args.model_name_or_path):
        #     checkpoint = model_args.model_name_or_path
        # else:
        #     checkpoint = None
        # TODO fix better about max_length
        checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload


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

    def score_with_Roberta(input, candidate):
        input = input.replace('[CLS]', '').replace('[SEP]', '')
        input = input + '[SEP]' + candidate
        token_type = []
        model_inputs = constrant_tokenizer(input, padding=False, return_tensors="pt")
        outputs = constrant_model(**model_inputs).logits
        type = [1 if outputs[0][j, 1] > 0 else 0 for j in range(len(outputs[0]))]
        token_type.append(type)

        model_inputs['token_type_ids'] = torch.tensor(token_type)
        score = float(score_model(**model_inputs).logits[0][1])

        return score

    with open('中文停用词表.txt', 'r', encoding='UTF-8') as f1:
        data_list = f1.readlines()
    stop_words = [word.strip() for word in data_list]

    def compute_penalty(input, candidate, key_info):
        penalty = 0
        key_info = ''.join(key_info)
        last_sentence = input.split('[SEP]')[-2]
        candidate = candidate.replace('[CLS]', '').replace('[SEP]', '')
        for id,word in enumerate(candidate):
            if word not in input and word not in stop_words:
                penalty -= 3
            if word in candidate[:id] and word not in stop_words:
                penalty -= 3
            if word in last_sentence and word in key_info:
                penalty += 1
        return penalty




    if training_args.do_eval:
        logger.info("*** Evaluate ***")



    if training_args.do_predict:
        logger.info("*** Test ***")

       
        if trainer.is_world_process_zero():
            

 

            output_test_preds_file = os.path.join(training_args.output_dir, "train_preds_seq2seq.txt")


            model_input = torch.tensor(
                tokenizer.encode('[CLS]', max_length=data_args.max_source_length, padding=padding,
                                 truncation=True)).unsqueeze(0).to("cuda:0")

            with open(output_test_preds_file, "w") as writer:
                for id in range(len(datasets["test"]['text'])):
                    sentence = '[CLS]'+datasets["test"]['text'][id].split('[CLS]')[1]
                    input, candidate = extract_span([sentence])
                    # input = sentence
                    candidate = candidate[0]
                    count = 0
                    writer.write('candidate:' + ' '.join(candidate) + '\n')
                    data_list = []
                    topk_score = []
                    while len(data_list) < 6 and count < 5:
                        random.shuffle(candidate)
                        num = math.floor(random.random() * 4) + 2
                        input = sentence + '[CLS]' + '[SEP]'.join(candidate[:num])

                        input_ids = torch.tensor(
                            tokenizer.encode(input, max_length=data_args.max_source_length, padding=padding,
                                             truncation=True)).unsqueeze(0).to("cuda:0")
                        last_ids = tokenizer.encode(input.split('[CLS]')[1].split('[SEP]')[-2],
                                                    add_special_tokens=False)

                        sample_outputs = model.generate(
                                    input_ids,
                                    decoder_input_ids=model_input,
                                    do_sample=True,
                                    max_length=30,
                                    top_p=0.95,
                                    num_return_sequences=3,
                                    return_dict_in_generate=True,
                                    output_scores=True
                                )

                        score_step = []
                        for step, score in enumerate(sample_outputs.scores):
                            probs = torch.nn.functional.softmax(score, dim=-1)
                            values, indices = torch.topk(probs, 20)
                            score_step.append(list())
                            values = values.cpu().numpy().tolist()
                            indices = indices.cpu().numpy().tolist()
                            for i in range(len(indices)):  # i indicates the top i sentence
                                score_step[step].append(dict())
                                for j in range(len(indices[i])):
                                    score_step[step][i][indices[i][j]] = values[i][j]
                        count += 1

                        best_sample, best_utility, utilities = mbr_decoding(
                            tokenizer.batch_decode(sample_outputs.sequences, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=True))
                        for i, sample in enumerate(sample_outputs.sequences):  # i indicates the top i sentence
                            sample_output = tokenizer.decode(sample)
                            tmp = ''.join(sample_output.split(" ")[3:])
                            end = tmp.find('[SEP]')
                            tmp = tmp[:end]
                            s = utilities[i]
                            embs = tokenizer.encode(tmp, add_special_tokens=False)
                            for k, t in enumerate(embs):
                                # if t in last_ids and t in score_step[k][i]:
                                #     s += np.log(2 * score_step[k][i][t])
                                # el
                                if t == 200:
                                    break
                                else:
                                    if t not in score_step[k][i]:
                                        s += np.log(1e-15)
                                    else:
                                        s += np.log(score_step[k][i][t])

                            if len(tmp) > 0 and tmp not in data_list:
                                # print("sample: " + tmp)
                                data_list.append(tmp)
                                roberta_score = score_with_Roberta(sentence, tmp)
                                penalty = compute_penalty(sentence, tmp, candidate)
                                final_score = 0.8 * roberta_score + 0.3 * penalty
                                topk_score.append(final_score)

                    for i, sample_output in enumerate(data_list):
                        writer.write(
                            "{}: {}\n".format(i, sample_output))
                    writer.write(str(topk_score) + "\n")
                    writer.write("\n")



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

