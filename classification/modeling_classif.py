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
import ast
import json
import os
import random
import statistics
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers.data.metrics import simple_accuracy

import logg
from classification.utils_classif import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from classification.wrapper_classif import TransformerModelWrapper, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig

logger = logg.get_logger('root')


class PetConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    """Configuration for training a model."""

    def __init__(self, device: str = None, per_gpu_train_batch_size: int = 8, per_gpu_unlabeled_batch_size: int = 8,
                 n_gpu: int = 1, num_train_epochs: int = 3, max_steps: int = -1, gradient_accumulation_steps: int = 1,
                 weight_decay: float = 0.0, learning_rate: float = 5e-5, adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0, max_grad_norm: float = 1, lm_training: bool = False, use_logits: bool = False,
                 alpha: float = 0.9999, temperature: float = 1):
        """
        Create a new training config.

        :param device: the device to use ('cpu' or 'gpu')
        :param per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train for
        :param max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
        :param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the maximum learning rate to use
        :param adam_epsilon: the epsilon value for Adam
        :param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        :param max_grad_norm: the maximum norm for the gradient
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use each training example's logits instead of its label (used for distillation)
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for distillation
        """
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.per_gpu_unlabeled_batch_size = per_gpu_unlabeled_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.lm_training = lm_training
        self.use_logits = use_logits
        self.alpha = alpha
        self.temperature = temperature


class EvalConfig(PetConfig):
    """Configuration for evaluating a model."""

    def __init__(self, device: str = None, n_gpu: int = 1, per_gpu_eval_batch_size: int = 8,
                 metrics: List[str] = None, decoding_strategy: str = 'default', priming: bool = False):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr', or 'parallel')
        :param priming: whether to use priming
        """
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics
        self.decoding_strategy = decoding_strategy
        self.priming = priming


def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config)
    return model


def train_classifier(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig, output_dir: str,
                     repetitions: int = 3, train_data: List[InputExample] = None, val_data: List[InputExample] = None,
                     unlabeled_data: List[InputExample] = None, eval_data: List[InputExample] = None,
                     do_train: bool = True, do_eval: bool = True, seed: int = 42):
    """
    Train and evaluate a sequence classification model.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    logger.info("\n--- CLASSIFICATION ---")

    # Step 2: Merge the annotations created by each individual model
    logger.info("\n--- Load Logits ---")
    logits_file = os.path.join(output_dir, 'unlabeled_logits.txt')
    logits = LogitsList.load(logits_file).logits
    assert len(logits) == len(unlabeled_data)
    logger.info("Got {} logits from file {}".format(len(logits), logits_file))
    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    # Step 3: Train the final sequence classifier model
    model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    model_config.use_logits = True

    train_pet_ensemble(model_config, train_config, eval_config, pattern_ids=[0], output_dir=output_dir,
                       repetitions=repetitions, train_data=train_data, val_data=val_data, unlabeled_data=unlabeled_data,
                       eval_data=eval_data, do_train=do_train, do_eval=do_eval, seed=seed, apply_classification=True)


def train_pet_ensemble(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig,
                       pattern_ids: List[int], output_dir: str, ipet_data_dir: str = None, repetitions: int = 3,
                       train_data: List[InputExample] = None, val_data: List[InputExample] = None,
                       unlabeled_data: List[InputExample] = None,
                       eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True,
                       save_unlabeled_logits: bool = False, seed: int = 42, apply_classification: bool = False):
    """
    Train and evaluate an ensemble of PET models without knowledge distillation.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ipet_data_dir: optional directory containing additional training data for iPET
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param save_unlabeled_logits: whether logits for unlabeled examples should be saved in a file ``logits.txt``. This
           is required for both iPET and knowledge distillation.
    :param seed: the random seed to use
    :param apply_classification: apply a Classification model
    """

    results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    logger.info("\n--- VERBALIZER : {} ---".format(model_config.verbalizer))

    for pattern_id in pattern_ids:

        if not apply_classification:
            logger.info("\n--- PATTERN {} : {} ---".format(pattern_id, model_config.pattern[pattern_id]))

        for iteration in range(repetitions):

            logger.info("Iteration : {}/{}".format(iteration, repetitions))

            model_config.pattern_id = pattern_id
            results_dict = {}

            if apply_classification:
                pattern_iter_output_dir = "{}/final_model".format(output_dir)
            else:
                pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, iteration)

            if os.path.exists(pattern_iter_output_dir):
                logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
                continue

            if not os.path.exists(pattern_iter_output_dir):
                os.makedirs(pattern_iter_output_dir)

            wrapper = init_model(model_config)

            # Training
            if do_train:
                if ipet_data_dir:
                    p = os.path.join(ipet_data_dir, 'p{}-i{}-train.bin'.format(pattern_id, iteration))
                    ipet_train_data = InputExample.load_examples(p)
                    for example in ipet_train_data:
                        example.logits = None
                else:
                    ipet_train_data = None

                results_dict.update(train_single_model(wrapper, train_data, train_config, eval_config,
                                                       val_data=val_data,
                                                       ipet_train_data=ipet_train_data,
                                                       unlabeled_data=unlabeled_data))

                with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                logger.info("Saving trained model at {}...".format(pattern_iter_output_dir))
                wrapper.save(pattern_iter_output_dir)
                train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
                eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
                logger.info("Saving complete")

                if save_unlabeled_logits:
                    logits = evaluate(wrapper, unlabeled_data, eval_config, type_dataset="unlabeled")['logits']
                    save_logits(os.path.join(pattern_iter_output_dir, 'logits.txt'), logits)

                if not do_eval:
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

            # Evaluation
            if do_eval:
                logger.info("Starting evaluation on eval set...")
                if not wrapper:
                    wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

                eval_result = evaluate(wrapper, eval_data, eval_config, type_dataset="eval_set", priming_data=train_data)

                save_predictions(os.path.join(pattern_iter_output_dir, 'predictions.jsonl'), wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

                scores = eval_result['scores']
                if apply_classification:
                    logger.info("--- RESULT (iteration={}) ---".format(iteration))
                else:
                    logger.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logger.info(scores)

                results_dict['test_set_after_training'] = scores
                with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                    json.dump(results_dict, fh)

                for metric, value in scores.items():
                    results[metric][pattern_id].append(value)

                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

    if do_eval:
        logger.info("=== OVERALL RESULTS ===")
        _write_results(os.path.join(output_dir, 'result_test.txt'), results)
    else:
        logger.info("=== ENSEMBLE TRAINING COMPLETE ===")


def train_single_model(model: TransformerModelWrapper, train_data: List[InputExample], config: TrainConfig,
                       eval_config: EvalConfig = None, val_data: List[InputExample] = None,
                       ipet_train_data: List[InputExample] = None,
                       unlabeled_data: List[InputExample] = None, return_train_set_results: bool = True):
    """
    Train a single model.

    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :param ipet_train_data: an optional list of iPET training examples to use
    :param unlabeled_data: an optional list of unlabeled examples to use
    :param return_train_set_results: whether results on the train set before and after training should be computed and
           returned
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    results_dict = {}

    model.model.to(device)

    if train_data and return_train_set_results:
        logger.info("Evaluation on Train set before training")
        scores = evaluate(model, train_data, eval_config, type_dataset="train")['scores']
        results_dict['train_set_before_training'] = scores['acc']
        logger.info("Scores :")
        logger.info(scores)

    all_train_data = train_data + ipet_train_data

    if not all_train_data and not config.use_logits:
        logger.warning('Training method was called without training examples')
    else:
        global_step, tr_loss = model.train(
            all_train_data, device,
            val_data=val_data,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            per_gpu_unlabeled_batch_size=config.per_gpu_unlabeled_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            unlabeled_data=unlabeled_data if config.lm_training or config.use_logits else None,
            lm_training=config.lm_training,
            use_logits=config.use_logits,
            alpha=config.alpha,
            temperature=config.temperature
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    # if train_data and return_train_set_results:
    #    logger.info("Evaluation on Train set after training")
    #    results_dict['train_set_after_training'] = evaluate(model, train_data, eval_config, type_dataset="train")['scores']['acc']

    return results_dict


def evaluate(model: TransformerModelWrapper, eval_data: List[InputExample], config: EvalConfig,
             type_dataset: str = 'unlabeled', priming_data: List[InputExample] = None) -> Dict:
    """
    Evaluate a model.

    :param model: the model to evaluate
    :param eval_data: the examples for evaluation
    :param config: the evaluation config
    :param priming_data: an optional list of priming data to use
    :param type_dataset: 'train', 'dev', 'unlabeled', 'test'
    :return: a dictionary containing the model's logits, predictions and (if any metrics are given) scores
    """

    if config.priming:
        for example in eval_data:
            example.meta['priming_data'] = priming_data

    metrics = config.metrics if config.metrics else ['acc']
    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    model.model.to(device)
    results = model.eval(eval_data, device, per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu, decoding_strategy=config.decoding_strategy,
                         type_dataset=type_dataset, priming=config.priming)

    predictions = np.argmax(results['logits'], axis=1)
    scores = {}

    for metric in metrics:
        if metric == 'acc':
            scores[metric] = simple_accuracy(predictions, results['labels'])
        elif metric == 'f1':
            scores[metric] = f1_score(results['labels'], predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(results['labels'], predictions, average='macro')
        elif metric == 'f1-weighted':
            scores[metric] = f1_score(results['labels'], predictions, average='weighted')
        elif metric == 'recall':
            scores[metric] = recall_score(results['labels'], predictions)
        elif metric == 'recall-macro':
            scores[metric] = recall_score(results['labels'], predictions, average='macro')
        elif metric == 'recall-weighted':
            scores[metric] = recall_score(results['labels'], predictions, average='weighted')
        elif metric == 'precision':
            scores[metric] = precision_score(results['labels'], predictions)
        elif metric == 'precision-macro':
            scores[metric] = precision_score(results['labels'], predictions, average='macro')
        elif metric == 'precision-weighted':
            scores[metric] = precision_score(results['labels'], predictions, average='weighted')
        elif metric == 'em':
            scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
        else:
            raise ValueError(f"Metric '{metric}' not implemented")

    results['scores'] = scores
    results['predictions'] = predictions
    return results


def test(output_dir: str, eval_data: List[InputExample], config: EvalConfig, label_map: dict,
         type_dataset: str = 'unlabeled', priming_data: List[InputExample] = None) -> Dict:
    """
    Test a model.

    :param model: the model to evaluate
    :param eval_data: the examples for evaluation
    :param config: the evaluation config
    :param priming_data: an optional list of priming data to use
    :param type_dataset: 'train', 'dev', 'unlabeled', 'test'
    :return: a dictionary containing the model's logits, predictions and (if any metrics are given) scores
    """

    TransformerModelWrapper_output_dir = "{}/final_model".format(output_dir)
    model = TransformerModelWrapper.from_pretrained(TransformerModelWrapper_output_dir)

    if config.priming:
        for example in eval_data:
            example.meta['priming_data'] = priming_data

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    model.model.to(device)
    results = model.eval(eval_data, device, per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu, decoding_strategy=config.decoding_strategy,
                         type_dataset=type_dataset, priming=config.priming)

    predictions = results['logits']

    logits_dict = {}
    for i in label_map.keys():
        logits_dict[label_map[i]] = predictions[:, int(i)]

    return logits_dict


def _write_results(path: str, results: Dict):
    with open(path, 'w') as fh:
        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')