import numpy as np
import pandas as pd
import plotly.express as px

import os

from pet.flags import save_yaml, load_yaml
import dataclasses
from joblib import load
from glob import glob
from shutil import rmtree, copytree, copyfile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from preparation.cleaner import Preprocessing_NLP
from preparation.prepare_data import Prepare

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
from pet.flags import save_yaml
import dataclasses

logger = log.get_logger('root')

# logging.getLogger().setLevel(verbosity_to_loglevel(2))

# create requirement.txt file : !pipreqs ./ --ignore .venv --encoding=utf8 --force

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


def add_fix_params(args):
    args.task_name = "Verbalizer"
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

    return args


class Pet:
    """Class for compile full pipeline of Pet task.
        Pet steps:
            - Preprocessing data :class:Preprocessing_NLP
            - Prepare data :class:Prepare
            - Compute Pet model
            - Returns prediction/metric results on validation data.
            - Returns prediction/metric results on test data.
        """

    def __init__(self, flags_parameters):
        """
        Args:
            flags_parameters : Instance of Flags class object

        From flags_parameters:
            column_text (str) : name of the column with texts (only one column)
            frac_trainset (float) : (]0;1]) fraction of dataset to use for train set
            seed (int) : seed used for train/test split, cross-validation split and fold choice
            outdir (str) : path of output logs
        """
        self.flags_parameters = flags_parameters
        self.column_text = flags_parameters.column_text
        self.frac_trainset = flags_parameters.frac_trainset
        self.seed = flags_parameters.seed
        self.outdir = self.flags_parameters.outdir

        self.flags_parameters = add_fix_params(self.flags_parameters)

        logger.info("Parameters: {}".format(self.flags_parameters))

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Setup CUDA, GPU & distributed training
        self.flags_parameters.device = "cuda" if torch.cuda.is_available() and not self.flags_parameters.no_cuda else "cpu"
        self.flags_parameters.n_gpu = torch.cuda.device_count()

        flags_dict = dataclasses.asdict(self.flags_parameters)

        save_yaml(os.path.join(self.outdir, "flags.yaml"), flags_dict)

        self.pre = None
        self.info_scores = {self.flags_parameters.model_type: {}}  # variable to store all information scores

    def data_preprocessing(self, data=None):
        """ Apply :class:Preprocessing_NLP from preprocessing_nlp.py
            and :class:Prepare from prepare_data.py
        Args :
            data (Dataframe)
        """
        # Read data
        if data is None:
            logger.info("\nRead data...")
            data = pd.read_csv(self.flags_parameters.path_data)

        # Preprocessing
        logger.info("\nBegin preprocessing of {} data :".format(len(data)))
        self.pre = Preprocessing_NLP(data, self.flags_parameters)
        self.data = self.pre.transform(data)

        if self.flags_parameters.path_data_validation == '' or self.flags_parameters.path_data_validation == 'empty' or self.flags_parameters.path_data_validation is None:
            self.dataset_val = None
        # Validation
        # use a loaded validation dataset :
        else:
            self.dataset_val = pd.read_csv(self.flags_parameters.path_data_validation)
            self.dataset_val = self.pre.transform(self.dataset_val)

        self.prepare = Prepare(self.flags_parameters)
        self.column_text, self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.folds = self.prepare.get_datasets(
            self.data, self.dataset_val)

        self.target = self.prepare.target

        # unlabeled data :
        self.dataset_unlabeled = pd.read_csv(self.flags_parameters.path_data_unlabeled)
        self.dataset_unlabeled = self.pre.transform(self.dataset_unlabeled)
        self.X_unlabeled, self.Y_unlabeled, self.column_text = self.prepare.separate_X_Y(self.dataset_unlabeled)
        if self.Y_unlabeled is None:
            self.Y_unlabeled = pd.DataFrame({self.Y_train.columns[0]: [self.Y_train.iloc[0, 0] for i in range(len(self.X_unlabeled))]})

    def preprocess_test_data(self, data_test):
        """ Apply same transformation as in the function self.data_preprocessing for data_test
        Args:
            data_test (str, list, dict, dataframe)
        Returns:
            data_test (Dataframe)
            doc_spacy_data_test (List) : documents from data_test preprocessed by spacy nlp
            y_test (DataFrame) (Optional)
        """
        if isinstance(data_test, str):
            data_test = pd.DataFrame({self.flags_parameters.column_text: [data_test]})
        elif isinstance(data_test, list):
            data_test = pd.DataFrame({self.flags_parameters.column_text: data_test})
        elif isinstance(data_test, dict):
            data_test = pd.DataFrame(data_test)

        if self.pre is not None:
            data_test = self.pre.transform(data_test)
            data_test, y_test, self.column_text = self.prepare.separate_X_Y(data_test)
        else:
            self.pre = Preprocessing_NLP(data_test, self.flags_parameters)
            data_test = self.pre.transform(data_test)
            self.prepare = Prepare(self.flags_parameters)
            data_test, y_test, self.column_text = self.prepare.separate_X_Y(data_test)
            self.target = self.prepare.target
        if y_test is None:
            return data_test
        else:
            return data_test, y_test

    def prepare_model(self, x=None, y=None, x_val=None, y_val=None, type_model="classifier"):
        """ Instantiate self.x_train, self.y_train, self.x_val, self.y_val and models
        Args:
              x (Dataframe) (Optional)
              y (Dataframe) (Optional)
              x_val (Dataframe) (Optional)
              y_val (Dataframe) (Optional)
              type_model (str) 'embedding', 'clustering' or 'classifier'
        """

        # if x and y are None use self.X_train and self.Y_train else use x and y :
        if x is not None:
            self.x_train = x
        else:
            self.x_train = self.X_train
        if y is not None:
            self.y_train = y
        else:
            self.y_train = self.Y_train
        if x_val is not None:
            self.x_val = x_val
        else:
            if self.X_val is not None:
                self.x_val = self.X_val
            else:
                self.x_val = None
        if y_val is not None:
            self.y_val = y_val
        else:
            if self.Y_val is not None:
                self.y_val = self.Y_val
            else:
                self.y_val = None

        if self.y_train is not None:
            assert isinstance(self.y_train, pd.DataFrame), "y/self.y_train must be a DataFrame type"

    def train_single_model(self, data_dir, train_file_name, dev_file_name, test_file_name, unlabeled_file_name,
                           text_x_column=0, text_y_column=1):

        output_dir_train = os.path.join(self.outdir, "train_single_model")
        if os.path.exists(output_dir_train) and os.listdir(output_dir_train) \
                and self.flags_parameters.do_train and not self.flags_parameters.overwrite_output_dir:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir_train))
        os.makedirs(output_dir_train)

        processor = ClassProcessor(TRAIN_FILE_NAME=train_file_name, DEV_FILE_NAME=dev_file_name,
                                   TEST_FILE_NAME=test_file_name, UNLABELED_FILE_NAME=unlabeled_file_name,
                                   LABELS=self.flags_parameters.labels, TEXT_A_COLUMN=text_x_column,
                                   TEXT_B_COLUMN=-1, LABEL_COLUMN=text_y_column)
        self.flags_parameters.label_list = processor.get_labels()
        with open(os.path.join(output_dir_train, "label_list.json"), "w") as outfile:
            json.dump(self.flags_parameters.label_list, outfile)

        args_dict = dataclasses.asdict(self.flags_parameters)
        save_yaml(os.path.join(output_dir_train, "flags.yaml"), args_dict)

        train_ex_per_label, test_ex_per_label = None, None
        train_ex, test_ex = self.flags_parameters.train_examples, self.flags_parameters.test_examples
        if self.flags_parameters.split_examples_evenly:
            train_ex_per_label = eq_div(self.flags_parameters.train_examples, len(self.flags_parameters.label_list)) if self.flags_parameters.train_examples != -1 else -1
            test_ex_per_label = eq_div(self.flags_parameters.test_examples, len(self.flags_parameters.label_list)) if self.flags_parameters.test_examples != -1 else -1
            train_ex, test_ex = None, None

        eval_set = TEST_SET if self.flags_parameters.eval_set == 'test' else DEV_SET

        train_data = load_examples(
            processor, data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)
        val_data = load_examples(
            processor, data_dir, DEV_SET, num_examples=-1, num_examples_per_label=test_ex_per_label)
        eval_data = load_examples(
            processor, data_dir, eval_set, num_examples=test_ex, num_examples_per_label=test_ex_per_label)
        unlabeled_data = load_examples(
            processor, data_dir, UNLABELED_SET, num_examples=self.flags_parameters.unlabeled_examples)

        # args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

        pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(self.flags_parameters)
        sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(self.flags_parameters)
        ipet_cfg = load_ipet_config(self.flags_parameters)

        if self.flags_parameters.method == 'pet':
            pet.train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                          pattern_ids=self.flags_parameters.pattern_ids, output_dir=output_dir_train,
                          ensemble_repetitions=self.flags_parameters.pet_repetitions, final_repetitions=self.flags_parameters.sc_repetitions,
                          reduction=self.flags_parameters.reduction, train_data=train_data, val_data=val_data,
                          unlabeled_data=unlabeled_data,
                          eval_data=eval_data, do_train=self.flags_parameters.do_train, do_eval=self.flags_parameters.do_eval,
                          no_distillation=self.flags_parameters.no_distillation, seed=self.flags_parameters.seed)

        elif self.flags_parameters.method == 'ipet':
            pet.train_ipet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, ipet_cfg, sc_model_cfg, sc_train_cfg,
                           sc_eval_cfg,
                           pattern_ids=self.flags_parameters.pattern_ids, output_dir=output_dir_train,
                           ensemble_repetitions=self.flags_parameters.pet_repetitions, final_repetitions=self.flags_parameters.sc_repetitions,
                           reduction=self.flags_parameters.reduction, train_data=train_data, val_data=val_data,
                           unlabeled_data=unlabeled_data,
                           eval_data=eval_data, do_train=self.flags_parameters.do_train, do_eval=self.flags_parameters.do_eval, seed=self.flags_parameters.seed)

        elif self.flags_parameters.method == 'sequence_classifier':
            pet.train_classifier(sc_model_cfg, sc_train_cfg, sc_eval_cfg, output_dir=output_dir_train,
                                 repetitions=self.flags_parameters.sc_repetitions, train_data=train_data, val_data=val_data,
                                 unlabeled_data=unlabeled_data,
                                 eval_data=eval_data, do_train=self.flags_parameters.do_train,
                                 do_eval=self.flags_parameters.do_eval, seed=self.flags_parameters.seed)
        else:
            raise ValueError(f"Training method '{self.flags_parameters.method}' not implemented")

        output_dir_final_model = os.path.join(output_dir_train, 'final')

        i = 0
        while os.path.exists(os.path.join(self.outdir, 'final_'+str(i))):
            i += 1
        copytree(output_dir_final_model, os.path.join(self.outdir, 'final_'+str(i)))
        rmtree(output_dir_train)

        return os.path.join(self.outdir, 'final_'+str(i))

    def predict_single_model(self, eval_set, output_dir_final_model, data_dir,
                             train_file_name, dev_file_name, test_file_name, unlabeled_file_name,
                             text_x_column=0, text_y_column=1):
        args = add_fix_params(self.flags_parameters)

        if output_dir_final_model is None:
            output_dir_final_model = os.path.join(self.outdir, 'final')

        logger.info("Parameters: {}".format(args))

        if not os.path.exists(self.outdir):
            raise ValueError("Output directory ({}) doesn't exist".format(self.outdir))

        f = open(os.path.join(output_dir_final_model, "label_map.json"), "r")
        label_map = json.load(f)

        processor = ClassProcessor(TRAIN_FILE_NAME=train_file_name, DEV_FILE_NAME=dev_file_name,
                                   TEST_FILE_NAME=test_file_name, UNLABELED_FILE_NAME=unlabeled_file_name,
                                   LABELS=args.labels, TEXT_A_COLUMN=text_x_column,
                                   TEXT_B_COLUMN=-1, LABEL_COLUMN=text_y_column)

        eval_data = load_examples(processor, data_dir, eval_set, num_examples=-1, num_examples_per_label=None)

        sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(args)

        logits_dict = pet.test(output_dir_final_model, eval_data, sc_eval_cfg, label_map, type_dataset=eval_set,
                               priming_data=None)

        return logits_dict

    def get_y_pred(self, logits_dict_test):
        map_label = {i: k for i, k in enumerate(logits_dict_test.keys())}
        y_pred = list(np.argmax(np.array([f for f in logits_dict_test.values()]).T, axis=1))
        y_pred = [map_label[p] for p in y_pred]
        return y_pred

    def get_y_confidence(self, logits_dict_test):
        y_confidence = list(np.max(np.array([f for f in logits_dict_test.values()]).T, axis=1))
        return y_confidence

    def calcul_metric_classification(self, y_true, y_pred, print_score=True):
        """ Compute for multi-class variable (y_true, y_pred) accuracy, recall_macro, precision_macro and f1_macro
        Args:
            y_true (Dataframe or array)
            y_pred (Dataframe or array)
            print_score (Boolean)
        Returns:
            acc, f1, recall, precision (float)
        """

        acc = np.round(accuracy_score(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)), 4)
        f1 = np.round(f1_score(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1), average='macro'), 4)
        recall = np.round(recall_score(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1), average='macro'), 4)
        precision = np.round(precision_score(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1), average='macro'), 4)

        if print_score:
            logger.info('\nScores :')
            logger.info('accuracy = {}'.format(acc))
            logger.info('precision {} = {}'.format('macro', precision))
            logger.info('recall {} = {}'.format('macro', recall))
            logger.info('f1 score {} = {}'.format('macro', f1))
            logger.info('\n')

        return acc, f1, recall, precision

    def train(self, x=None, y=None, x_val=None, y_val=None):
        """ Careful : Train for classification
        Args:
              x (Dataframe) (Optional)
              y (Dataframe) (Optional)
              x_val (Dataframe) (Optional)
              y_val (Dataframe) (Optional)
        """

        self.prepare_model(x=x, y=y, x_val=x_val, y_val=y_val, type_model="classifier")

        # variable to store best Hyperparameters with specific "seed" and "scoring"
        dict_models_parameters = {"seed": self.seed}

        if self.x_val is None:
            self.fold_id = np.ones((len(self.y_train),)) * -1
        else:
            self.fold_id = np.ones((len(self.y_val),)) * -1

        self.oof_val = {}

        data_dir = self.outdir
        train_file_name = "train"
        dev_file_name = "dev.csv"
        test_file_name = "test.csv"
        unlabeled_file_name = "unlabeled.csv"

        first_fold = True
        for num_fold, (train_index, val_index) in enumerate(self.folds):
            logger.info("Fold {}:".format(num_fold))

            if train_index == 'all':
                # validation
                x_train, x_val = self.x_train, self.x_val
                y_train, y_val = self.y_train, self.y_val
            else:
                # cross-validation
                x_train, x_val = self.x_train.iloc[train_index, :], self.x_train.iloc[val_index, :]
                y_train, y_val = self.y_train.iloc[train_index, :], self.y_train.iloc[val_index, :]

            text_x_column = 0
            text_y_column = 1

            pd.concat([x_train, y_train], axis=1).to_csv(os.path.join(data_dir, train_file_name), index=False)
            pd.concat([x_val, y_val], axis=1).to_csv(os.path.join(data_dir, dev_file_name), index=False)
            pd.concat([self.X_unlabeled, self.Y_unlabeled], axis=1).to_csv(os.path.join(data_dir, unlabeled_file_name), index=False)

            output_dir_final_model = self.train_single_model(data_dir, train_file_name, dev_file_name,
                                                             dev_file_name, unlabeled_file_name,
                                                             text_x_column=text_x_column, text_y_column=text_y_column)

            logits_dict_val = self.predict_single_model(eval_set="dev", output_dir_final_model=output_dir_final_model,
                                                        data_dir=data_dir,
                                                        train_file_name=train_file_name, dev_file_name=dev_file_name,
                                                        test_file_name=dev_file_name, unlabeled_file_name=unlabeled_file_name,
                                                        text_x_column=text_x_column, text_y_column=text_y_column)
            y_val_pred = self.get_y_pred(logits_dict_val)

            for it, idx in enumerate(val_index):
                self.oof_val[idx] = y_val_pred[it]
            self.fold_id[val_index] = num_fold

        sd = sorted(self.oof_val.items())
        prediction_oof_val = []
        for k, v in sd:
            prediction_oof_val.append(v)
        prediction_oof_val = np.array(prediction_oof_val)

        if self.X_val is None:
            # cross-validation
            y_true_sample = self.y_train.values[np.where(self.fold_id >= 0)[0]].copy()
        else:
            # validation
            y_true_sample = self.Y_val.copy()

        acc, f1, recall, precision = self.calcul_metric_classification(y_true_sample, prediction_oof_val, True)

        self.info_scores[self.flags_parameters.model_type]["accuracy_val"] = acc
        self.info_scores[self.flags_parameters.model_type]["f1_macro_val"] = f1
        self.info_scores[self.flags_parameters.model_type]["recall_macro_val"] = recall
        self.info_scores[self.flags_parameters.model_type]["precision_macro_val"] = precision

    def get_leaderboard(self, dataset='val', sort_by=None, ascending=False, info_models=None):
        """ Metric scores for each model of self.models or info_models
            if no optimization and validation : you need to give info_model (dictionary with Model class)
        Args:
            dataset (str) : 'val' or 'test', which prediction to use
            sort_by (str) : metric column name to sort
            ascending (Boolean)
            info_models (dict) : dictionary with Model class
        Return:
             self.leaderboard (Dataframe)
        """

        metrics = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']
        scores = {}

        leaderboard = {"name": list(self.info_scores.keys())}
        for metric in metrics:
            scores[metric + '_' + dataset] = [self.info_scores[name_model][metric + '_' + dataset]
                                                            for name_model in self.info_scores.keys()]
            leaderboard[metric + '_' + dataset] = np.round(scores[metric + '_' + dataset], 4)
        if sort_by in ['recall', 'precision', 'f1']:
            sort_by = sort_by + '_macro'

        leaderboard = pd.DataFrame(leaderboard)
        if sort_by:
            leaderboard = leaderboard.sort_values(by=sort_by + '_' + dataset, ascending=ascending)
        return leaderboard

    def prediction(self, on_test_data=True, x=None, y=None, proba=True):
        """ prediction on X_test if on_test_data else x
        """

        if on_test_data and x is None:  # predict on self.X_test
            x = self.X_test
            y = self.Y_test

        data_dir = self.outdir
        test_file_name = "test.csv"
        text_x_column = 0
        text_y_column = 1
        pd.concat([x, y], axis=1).to_csv(os.path.join(data_dir, test_file_name), index=False)

        model_paths = sorted(glob(os.path.join(self.outdir, 'final*')))
        n_model = len(model_paths)

        result_dict = {}
        for model_path in model_paths:
            logger.info("model from : {}".format(model_path))

            logits_dict_test = self.predict_single_model(eval_set="test", output_dir_final_model=model_path,
                                                         data_dir=data_dir,
                                                         train_file_name="", dev_file_name="",
                                                         test_file_name=test_file_name, unlabeled_file_name="",
                                                         text_x_column=text_x_column, text_y_column=text_y_column)

            for label in logits_dict_test.keys():
                if label not in result_dict.keys():
                    result_dict[label] = logits_dict_test[label]
                else:
                    result_dict[label] = result_dict[label] + logits_dict_test[label]

            y_test_pred = self.get_y_pred(logits_dict_test)
            _, _, _, _ = self.calcul_metric_classification(y, y_test_pred, True)

        for label in result_dict.keys():
            result_dict[label] = result_dict[label] / n_model

        y_test_pred = self.get_y_pred(result_dict)
        y_test_confidence = self.get_y_confidence(result_dict)

        acc, f1, recall, precision = self.calcul_metric_classification(y, y_test_pred, True)

        self.info_scores[self.flags_parameters.model_type]["accuracy_test"] = acc
        self.info_scores[self.flags_parameters.model_type]["f1_macro_test"] = f1
        self.info_scores[self.flags_parameters.model_type]["recall_macro_test"] = recall
        self.info_scores[self.flags_parameters.model_type]["precision_macro_test"] = precision

        return y_test_pred, y_test_confidence, result_dict

