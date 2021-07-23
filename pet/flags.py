from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
from typing import Any, Union
from yaml import dump, full_load


@dataclass
class Flags:
    """ Class to instantiate parameters """
    ### Required parameters

    # The training method to use. Either regular sequence classification, PET or iPET.
    method: str = 'pet'  # ['pet', 'ipet', 'sequence_classifier']
    # "The input data dir. Should contain the data files for the task."
    data_dir: str = field(default_factory=str)
    # The type of the pretrained language model to use
    model_type: str = "camembert"
    # Path to the pre-trained model or shortcut name
    model_name_or_path: str = field(default_factory=str)
    # The output directory where the model predictions and checkpoints will be written
    output_dir: str = field(default_factory=str)

    ### Dataset

    train_file_name: str = "train.csv"
    dev_file_name: str = "dev.csv"
    test_file_name: str = "test.csv"
    unlabeled_file_name: str = "unlabeled.csv"
    labels: list = field(default_factory=lambda: ["1", "2"])
    text_a_column: int = 1
    text_b_column: int = -1
    label_column: int = 0

    ### Pattern
    verbalizer: dict = field(default_factory=lambda: {"1": ["nul"], "2": ["bien"]})
    pattern: dict = field(default_factory=lambda: {0: "C'est MASK ! TEXT_A"})

    ### PET-specific optional parameters

    # The ids of the PVPs to be used (only for PET)
    pattern_ids: list = field(default_factory=lambda: [0])
    # Reduction strategy for merging predictions from multiple PET models
    reduction: str = 'wmean'   # ['wmean', 'mean']
    # The number of times to repeat PET training and testing with different seeds
    pet_repetitions: int = 1
    # The maximum total input sequence length after tokenization for PET
    pet_max_seq_length: int = 256
    # Total number of training epochs to perform in PET
    pet_num_train_epochs: int = 3
    # Batch size per GPU/CPU for PET training
    pet_per_gpu_train_batch_size: int = 4
    # Batch size per GPU/CPU for PET evaluation
    pet_per_gpu_eval_batch_size: int = 8
    # Batch size per GPU/CPU for auxiliary language modeling examples in PET
    pet_per_gpu_unlabeled_batch_size: int = 4

    ### SequenceClassifier-specific optional parameters (also used for the final PET classifier)

    # The maximum total input sequence length after tokenization for sequence classification
    sc_max_seq_length: int = 256
    # Batch size per GPU/CPU for sequence classifier training
    sc_per_gpu_train_batch_size: int = 4
    # Batch size per GPU/CPU for sequence classifier evaluation
    sc_per_gpu_eval_batch_size: int = 8
    # Batch size per GPU/CPU for unlabeled examples used for distillation
    sc_per_gpu_unlabeled_batch_size: int = 4
    # Total number of training epochs to perform for sequence classifier training
    sc_num_train_epochs: int = 3
    sc_max_steps: int = -1

    ### iPET-specific optional parameters

    # The number of generations to train
    ipet_generations: int = 3
    # The percentage of models to choose for annotating new training sets
    ipet_logits_percentage: float = 0.25
    # The factor by which to increase the training set size per generation
    ipet_scale_factor: int = 5
    # If >0, in the first generation the n_most_likely examples per label are chosen even
    # if their predicted label is different
    ipet_n_most_likely: int = -1

    ### Other optional parameters

    # The total number of train examples to use, where -1 equals all examples
    train_examples: int = -1
    # The total number of test examples to use, where -1 equals all examples
    test_examples: int = -1
    # The total number of unlabeled examples to use, where -1 equals all examples
    unlabeled_examples: int = -1
    # Metrics
    metrics: list = field(default_factory=lambda: ["acc", "f1", "f1-macro", "f1-weighted"])

    # The initial learning rate for Adam
    learning_rate: float = 1e-5
    # random seed for initialization
    seed: int = 15

    # Whether to perform training
    do_train: bool = True
    # Whether to perform evaluation
    do_eval: bool = True
    # Whether to perform evaluation on the dev set or the test set
    eval_set: str = "test"   # ['dev', 'test']
    # Whether to perform distillation
    no_distillation: bool = False

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self


def save_yaml(filepath: Union[str, Path], content: Any, width: int = 120):
    with open(filepath, "w") as f:
        dump(content, f, width=width)


def load_yaml(filepath: Union[str, Path]) -> Any:
    with open(filepath, "r") as f:
        content = full_load(f)
    return content