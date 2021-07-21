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

"""
This script can be used to train and evaluate either a regular supervised model or a PET/iPET model on
one of the supported tasks and datasets.
"""

import argparse
import os
import json
from typing import Tuple

import torch

from pet.tasks import ClassProcessor, PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from pet.utils import eq_div
from pet.wrapper import WRAPPER_TYPES, MODEL_CLASSES, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig
import pet
import log

logger = log.get_logger('root')


def load_pet_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=args.wrapper_type, task_name=args.task_name, label_list=args.label_list,
                              max_seq_length=args.pet_max_seq_length, verbalizer_file=args.verbalizer_file,
                              cache_dir=args.cache_dir, verbalizer=args.verbalizer, pattern=args.pattern)

    train_cfg = pet.TrainConfig(device=args.device, per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
                                per_gpu_unlabeled_batch_size=args.pet_per_gpu_unlabeled_batch_size, n_gpu=args.n_gpu,
                                num_train_epochs=args.pet_num_train_epochs, max_steps=args.pet_max_steps,
                                gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm, lm_training=args.lm_training, alpha=args.alpha)

    eval_cfg = pet.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size,
                              decoding_strategy=args.decoding_strategy, priming=args.priming)

    return model_cfg, train_cfg, eval_cfg


def load_sequence_classifier_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for a regular sequence classifier from the given command line
    arguments. This classifier can either be used as a standalone model or as the final classifier for PET/iPET.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=SEQUENCE_CLASSIFIER_WRAPPER, task_name=args.task_name,
                              label_list=args.label_list, max_seq_length=args.sc_max_seq_length,
                              verbalizer_file=args.verbalizer_file, cache_dir=args.cache_dir,
                              verbalizer=args.verbalizer, pattern=args.pattern)

    train_cfg = pet.TrainConfig(device=args.device, per_gpu_train_batch_size=args.sc_per_gpu_train_batch_size,
                                per_gpu_unlabeled_batch_size=args.sc_per_gpu_unlabeled_batch_size, n_gpu=args.n_gpu,
                                num_train_epochs=args.sc_num_train_epochs, max_steps=args.sc_max_steps,
                                temperature=args.temperature,
                                gradient_accumulation_steps=args.sc_gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm, use_logits=args.method != 'sequence_classifier')

    eval_cfg = pet.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.sc_per_gpu_eval_batch_size)

    return model_cfg, train_cfg, eval_cfg


def load_ipet_config(args) -> pet.IPetConfig:
    """
    Load the iPET config from the given command line arguments.
    """
    ipet_cfg = pet.IPetConfig(generations=args.ipet_generations, logits_percentage=args.ipet_logits_percentage,
                              scale_factor=args.ipet_scale_factor, n_most_likely=args.ipet_n_most_likely)
    return ipet_cfg


def main(args):

    args.wrapper_type = "mlm"
    args.lm_training = True
    args.alpha = 0.9999
    args.temperature = 2
    args.verbalizer_file = None
    args.decoding_strategy = "default"
    args.pet_gradient_accumulation_steps = 1
    args.pet_max_steps = -1
    args.sc_repetitions = 1
    args.sc_gradient_accumulation_steps = 1
    args.sc_max_steps = -1
    args.split_examples_evenly = False
    args.cache_dir = ""
    args.weight_decay = 0.01
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 1.0
    args.warmup_steps = 0
    args.logging_steps = 50
    args.no_cuda = False
    args.overwrite_output_dir = False
    args.priming = False

    logger.info("Parameters: {}".format(args))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    #args.task_name = args.task_name.lower()
    #if args.task_name not in PROCESSORS:
    #    raise ValueError("Task '{}' not found".format(args.task_name))
    processor = ClassProcessor(TRAIN_FILE_NAME=args.train_file_name, DEV_FILE_NAME=args.dev_file_name,
                               TEST_FILE_NAME=args.test_file_name, UNLABELED_FILE_NAME=args.unlabeled_file_name,
                               LABELS=args.labels, TEXT_A_COLUMN=args.text_a_column,
                               TEXT_B_COLUMN=args.text_b_column, LABEL_COLUMN=args.label_column)
    args.label_list = processor.get_labels()
    with open(os.path.join(args.output_dir, "label_list.json"), "w") as outfile:
        json.dump(args.label_list, outfile)

    train_ex_per_label, test_ex_per_label = None, None
    train_ex, test_ex = args.train_examples, args.test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
        train_ex, test_ex = None, None

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET

    train_data = load_examples(
        processor, args.data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)
    val_data = load_examples(
        processor, args.data_dir, DEV_SET, num_examples=-1, num_examples_per_label=test_ex_per_label)
    eval_data = load_examples(
        processor, args.data_dir, eval_set, num_examples=test_ex, num_examples_per_label=test_ex_per_label)
    unlabeled_data = load_examples(
        processor, args.data_dir, UNLABELED_SET, num_examples=args.unlabeled_examples)

    # args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)
    sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(args)
    ipet_cfg = load_ipet_config(args)

    if args.method == 'pet':
        pet.train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                      pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                      ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                      reduction=args.reduction, train_data=train_data, val_data=val_data, unlabeled_data=unlabeled_data,
                      eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval,
                      no_distillation=args.no_distillation, seed=args.seed)

    elif args.method == 'ipet':
        pet.train_ipet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, ipet_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                       pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                       ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                       reduction=args.reduction, train_data=train_data, val_data=val_data, unlabeled_data=unlabeled_data,
                       eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed)

    elif args.method == 'sequence_classifier':
        pet.train_classifier(sc_model_cfg, sc_train_cfg, sc_eval_cfg, output_dir=args.output_dir,
                             repetitions=args.sc_repetitions, train_data=train_data, val_data=val_data,
                             unlabeled_data=unlabeled_data,
                             eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed)

    else:
        raise ValueError(f"Training method '{args.method}' not implemented")


def test(args, eval_set="test", output_dir_final_model=None):

    if output_dir_final_model is None:
        output_dir_final_model = os.path.join(args.output_dir, 'final')

    args.wrapper_type = "mlm"
    args.lm_training = True
    args.alpha = 0.9999
    args.temperature = 2
    args.verbalizer_file = None
    args.decoding_strategy = "default"
    args.pet_gradient_accumulation_steps = 1
    args.pet_max_steps = -1
    args.sc_repetitions = 1
    args.sc_gradient_accumulation_steps = 1
    args.sc_max_steps = -1
    args.split_examples_evenly = False
    args.cache_dir = ""
    args.weight_decay = 0.01
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 1.0
    args.warmup_steps = 0
    args.logging_steps = 50
    args.no_cuda = False
    args.overwrite_output_dir = False
    args.priming = False

    logger.info("Parameters: {}".format(args))

    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory ({}) doesn't exist".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    f = open(os.path.join(output_dir_final_model, "label_map.json"), "r")
    label_map = json.load(f)

    processor = ClassProcessor(TRAIN_FILE_NAME=args.train_file_name, DEV_FILE_NAME=args.dev_file_name,
                               TEST_FILE_NAME=args.test_file_name, UNLABELED_FILE_NAME=args.unlabeled_file_name,
                               LABELS=args.labels, TEXT_A_COLUMN=args.text_a_column,
                               TEXT_B_COLUMN=args.text_b_column, LABEL_COLUMN=args.label_column)

    eval_data = load_examples(processor, args.data_dir, eval_set, num_examples=-1, num_examples_per_label=None)

    sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(args)

    logits_dict = pet.test(output_dir_final_model, eval_data, sc_eval_cfg, label_map, type_dataset=eval_set, priming_data=None)

    return logits_dict


if __name__ == "__main__":

    from pet.flags import Flags
    args = Flags()

    flags_dict_info = {
                      "method": 'pet',  # ['pet', 'ipet', 'sequence_classifier']
                      "data_dir": '/content/',
                      "model_type": "camembert",
                      "model_name_or_path": "camembert-base",
                      "task_name": "binary-polarity",
                      "output_dir": "/content/test0/",
                      "pattern_ids": [0, 1, 2, 3],
                      "pet_repetitions": 1,
                      "pet_max_seq_length": 400,
                      "pet_num_train_epochs": 3,
                      "sc_max_seq_length": 400,
                      "sc_num_train_epochs": 3,
                      "eval_set": "test",
                      "verbalizer": {"1": ["nul"], "2": ["bien"]},
                      "pattern": {0: "C'est MASK ! TEXT_A"}
                      }

    args = args.update(flags_dict_info)

    main(args)
