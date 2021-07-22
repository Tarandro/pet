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

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from classification.utils_classif import InputFeatures, InputExample


class Preprocessor(ABC):
    """
    A preprocessor that transforms an :class:`InputExample` into a :class:`InputFeatures` object so that it can be
    processed by the model being used.
    """

    def __init__(self, wrapper):
        """
        Create a new preprocessor.

        :param wrapper: the wrapper for the language model to use
        """
        self.wrapper = wrapper
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}

    @abstractmethod
    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass


class SequenceClassifierPreprocessor(Preprocessor):
    """Preprocessor for a regular sequence classification model."""

    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:
        inputs = self.wrapper.task_helper.get_sequence_classifier_inputs(example) if self.wrapper.task_helper else None
        if inputs is None:
            inputs = self.wrapper.tokenizer.encode_plus(
                example.text_a if example.text_a else None,
                example.text_b if example.text_b else None,
                add_special_tokens=True,
                max_length=self.wrapper.config.max_seq_length,
                truncation=True
            )
        input_ids, token_type_ids = inputs["input_ids"], inputs.get("token_type_ids")

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        if not token_type_ids:
            token_type_ids = [0] * self.wrapper.config.max_seq_length
        else:
            token_type_ids = token_type_ids + ([0] * padding_length)
        mlm_labels = [-1] * len(input_ids)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100
        # logits = example.logits if example.logits else [-1]
        if example.logits:
            logits = example.logits
        else:
            logits = [1 if k == example.label else 0 for k in self.label_map.keys()]

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, logits=logits, idx=example.idx)
