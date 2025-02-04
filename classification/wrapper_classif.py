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
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import json
import jsonpickle
import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, \
    XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, \
    BertConfig, BertForSequenceClassification, BertTokenizer, RobertaConfig, \
    RobertaForSequenceClassification, RobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, \
    XLMRobertaTokenizer, AlbertForSequenceClassification, AlbertTokenizer, AlbertConfig, \
    GPT2Config, GPT2Tokenizer, CamembertConfig, CamembertTokenizer, CamembertForSequenceClassification
from transformers import __version__ as transformers_version

import logg
from classification import preprocessor_classif
from classification.tasks_classif import TASK_HELPERS
from classification.utils_classif import InputFeatures, DictDataset, distillation_loss

logger = logg.get_logger('root')

CONFIG_NAME = 'wrapper_config.json'
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER]

PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: preprocessor_classif.SequenceClassifierPreprocessor
}

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: RobertaForSequenceClassification
    },
    'xlm-roberta': {
        'config': XLMRobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: XLMRobertaForSequenceClassification
    },
    'camembert': {
        'config': CamembertConfig,
        'tokenizer': CamembertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: CamembertForSequenceClassification
    },
    'xlnet': {
        'config': XLNetConfig,
        'tokenizer': XLNetTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: XLNetForSequenceClassification
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: AlbertForSequenceClassification
    },
    'gpt2': {
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer
    },
}

EVALUATION_STEP_FUNCTIONS = {
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_eval_step,
}

TRAIN_STEP_FUNCTIONS = {
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_train_step,
}


class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(self, model_type: str, model_name_or_path: str, wrapper_type: str, task_name: str, max_seq_length: int,
                 label_list: List[str], pattern_id: int = 0, verbalizer_file: str = None, cache_dir: str = None,
                 verbalizer: dict = None, pattern: dict = None):
        """
        Create a new config.

        :param model_type: the model type (e.g., 'bert', 'roberta', 'albert')
        :param model_name_or_path: the model name (e.g., 'roberta-large') or path to a pretrained model
        :param wrapper_type: the wrapper type (one of 'mlm', 'plm' and 'sequence_classifier')
        :param task_name: the task to solve
        :param max_seq_length: the maximum number of tokens in a sequence
        :param label_list: the list of labels for the task
        :param pattern_id: the id of the pattern to use
        :param verbalizer_file: optional path to a verbalizer file
        :param cache_dir: optional path to a cache dir
        """
        if verbalizer is None:
            verbalizer = dict()
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.wrapper_type = wrapper_type
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.verbalizer_file = verbalizer_file
        self.cache_dir = cache_dir
        self.verbalizer = verbalizer
        self.pattern = pattern


class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        """Create a new wrapper from the given config."""
        self.config = config
        config_class = MODEL_CLASSES[self.config.model_type]['config']
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]

        model_config = config_class.from_pretrained(
            config.model_name_or_path, num_labels=len(config.label_list), finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None, use_cache=False)

        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)  # type: PreTrainedTokenizer

        if self.config.model_type == 'gpt2':
            self.tokenizer.pad_token, self.tokenizer.mask_token = self.tokenizer.eos_token, self.tokenizer.eos_token

        self.model = model_class.from_pretrained(config.model_name_or_path, config=model_config,
                                                 cache_dir=config.cache_dir if config.cache_dir else None)

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](self)
        self.task_helper = TASK_HELPERS[self.config.task_name](self) if self.config.task_name in TASK_HELPERS else None

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](wrapper)
        wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](wrapper) \
            if wrapper.config.task_name in TASK_HELPERS else None
        return wrapper

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def train(self, task_train_data: List[InputExample], device, val_data: List[InputExample] = None,
              per_gpu_train_batch_size: int = 8, n_gpu: int = 1,
              num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
              learning_rate: float = 5e-5, adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
              logging_steps: int = 50, per_gpu_unlabeled_batch_size: int = 8, unlabeled_data: List[InputExample] = None,
              lm_training: bool = False, use_logits: bool = False, alpha: float = 0.8, temperature: float = 1,
              max_steps=-1, **_):
        """
        Train the underlying language model.

        :param task_train_data: the training examples to use
        :param device: the training device (cpu/gpu)
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param unlabeled_data: the unlabeled examples to use
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use the example's logits instead of their labels to compute the loss
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for knowledge distillation
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        logger.info('--- Train Dataset Generation ---')
        train_dataset = self._generate_dataset(task_train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if val_data is not None:
            logger.info('--- Val Dataset Generation ---')
            val_dataset = self._generate_dataset(val_data)
            val_sampler = RandomSampler(val_dataset)
            val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_batch_size)

        unlabeled_dataloader, unlabeled_iter = None, None

        if lm_training or use_logits:
            # we need unlabeled data both for auxiliary language modeling and for knowledge distillation
            assert unlabeled_data is not None
            unlabeled_batch_size = per_gpu_unlabeled_batch_size * max(1, n_gpu)
            logger.info(f'--- Unlabeled Dataset Generation ---')
            unlabeled_dataset = self._generate_dataset(unlabeled_data, labelled=False)
            unlabeled_sampler = RandomSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(unlabeled_dataset, sampler=unlabeled_sampler,
                                              batch_size=unlabeled_batch_size)
            unlabeled_iter = unlabeled_dataloader.__iter__()

        if use_logits:
            train_dataloader = unlabeled_dataloader

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        # multi-gpu training
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        losses = []
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            running_loss = 0.0
            for _, batch in enumerate(train_dataloader):   # enumerate(epoch_iterator):
                self.model.train()

                batch = {k: t.to(device) for k, t in batch.items()}

                train_step_inputs = {
                    'unlabeled_batch': None, 'lm_training': False, 'alpha': alpha,
                    'use_logits': use_logits, 'temperature': temperature
                }
                loss = self.task_helper.train_step(batch, **train_step_inputs) if self.task_helper else None

                if loss is None:
                    loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch, **train_step_inputs)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                running_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        print(json.dumps({**logs, **{'step': global_step}}))

                if 0 < max_steps < global_step:
                    # train_dataloader.close()
                    break
                step += 1

            train_epoch_loss = running_loss / len(train_dataloader)

            if val_data is not None:
                self.model.eval()

                running_loss = 0.0
                for _, batch in enumerate(val_dataloader):
                    batch = {k: t.to(device) for k, t in batch.items()}

                    train_step_inputs = {
                        'unlabeled_batch': None, 'lm_training': False, 'alpha': alpha,
                        'use_logits': use_logits, 'temperature': temperature
                    }
                    loss = self.task_helper.train_step(batch, **train_step_inputs) if self.task_helper else None

                    if loss is None:
                        loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch, **train_step_inputs)

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training

                    running_loss += loss.item()

                val_epoch_loss = running_loss / len(val_dataloader)
                print(f'Score Epoch, Loss: {train_epoch_loss:.5f} val loss: {val_epoch_loss:.5f}')
                losses.append([train_epoch_loss, val_epoch_loss])
            else:
                print(f'Score Epoch, Loss: {train_epoch_loss:.4f} ')
                losses.append(train_epoch_loss)

            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        return global_step, (tr_loss / global_step if global_step > 0 else -1)

    def eval(self, eval_data: List[InputExample], device, per_gpu_eval_batch_size: int = 8, n_gpu: int = 1,
             priming: bool = False, decoding_strategy: str = 'default', type_dataset: str = 'unlabeled') -> Dict:
        """
        Evaluate the underlying language model.

        :param eval_data: the evaluation examples to use
        :param device: the evaluation device (cpu/gpu)
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param priming: whether to use priming
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr' or 'parallel')
        :param type_dataset: 'train', 'dev', 'unlabeled', 'test'
        :return: a dictionary of numpy arrays containing the indices, logits, labels, and (optional) question_ids for
                 each evaluation example.
        """

        logger.info('--- Evaluation on {} Dataset Generation ---'.format(type_dataset))
        eval_dataset = self._generate_dataset(eval_data, priming=priming, type_dataset=type_dataset)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            with torch.no_grad():

                # some tasks require special evaluation
                logits = self.task_helper.eval_step(batch,
                                                    decoding_strategy=decoding_strategy) if self.task_helper else None

                if logits is None:
                    logits = EVALUATION_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)

        return {
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }

    def _generate_dataset(self, data: List[InputExample], labelled: bool = True, priming: bool = False,
                          type_dataset: str = "test"):
        features = self._convert_examples_to_features(data, labelled=labelled, priming=priming,
                                                      type_dataset=type_dataset)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long)
        }

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True,
                                      priming: bool = False, type_dataset: str = "test") -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {} - {}".format(ex_index, min(ex_index+10000, len(examples))))
            input_features = self.preprocessor.get_input_features(example, labelled=labelled, priming=priming)
            if self.task_helper:
                self.task_helper.add_special_input_features(example, input_features)
            features.append(input_features)
            if type_dataset == "train":
                if ex_index < 2:
                    logger.info(f'--- Example {ex_index} ---')
                    logger.info(input_features.pretty_print(self.tokenizer))
        return features

    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs

    def sequence_classifier_train_step(self, batch: Dict[str, torch.Tensor], use_logits: bool = False,
                                       temperature: float = 1, **_) -> torch.Tensor:
        """Perform a sequence classifier training step."""

        inputs = self.generate_default_inputs(batch)
        if not use_logits:
            inputs['labels'] = batch['labels']

        outputs = self.model(**inputs)

        if use_logits:
            logits_predicted, logits_target = outputs[0], batch['logits']
            return distillation_loss(logits_predicted, logits_target, temperature)
        else:
            return outputs[0]

    def sequence_classifier_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a sequence classifier evaluation step."""
        inputs = self.generate_default_inputs(batch)
        return self.model(**inputs)[0]
