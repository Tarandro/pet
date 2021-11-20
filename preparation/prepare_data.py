import pandas as pd
import numpy as np
import random as rd
from sklearn.model_selection import KFold, StratifiedKFold

import logg
logger = logg.get_logger('root')


class Prepare:
    """Class to compile full pipeline : Prepare data
            steps:
                - separate column text and target -> X and Y
                - Split data in train/test according to frac_trainset
                - create cross validation split or prepare validation dataset
    """

    def __init__(self, flags_parameters):
        """
        Args:
            flags_parameters : Instance of Flags class object
        From flags_parameters:
            column_text (str) : name of the column with texts (only one column)
            column_text_b (str) : name of the column_b with texts (only one column)
            target (str or List) : names of target columns
            frac_trainset (float) pourcentage of data for train set
            map_label (dict) dictionary to map label to integer
            debug (bool) if True use only 50 data rows for training
            nfolds (int) number of folds to split dataset
            nfolds_train (int) number of folds to train during optimization/validation
            cv_strategy ("StratifiedKFold" or "KFold")
        """
        self.column_text = flags_parameters.column_text
        self.column_text_b = flags_parameters.column_text_b
        self.frac_trainset = flags_parameters.frac_trainset
        self.debug = flags_parameters.debug
        self.seed = flags_parameters.seed
        self.nfolds = flags_parameters.nfolds
        self.nfolds_train = flags_parameters.nfolds_train
        self.cv_strategy = flags_parameters.cv_strategy

        # self.target need to be a List
        self.target = flags_parameters.target
        if isinstance(self.target, list):
            self.target = self.target
        else:
            self.target = [self.target]

    def separate_X_Y(self, data):
        """ separate column text and target -> X and Y
        Args:
             data (DataFrame)
        Return:
            data (DataFrame) data from input with the column_text and without target columns
            Y (DataFrame) data from input with only target columns
            column_text (int) the column number of self.column_text (str) in data
        """

        if len([col for col in self.target if col in data.columns]) > 0:
            col_Y = [col for col in self.target if col in data.columns]
            Y = data[col_Y]
            data = data.drop(col_Y, axis=1)

        else:
            Y = None

        # for X, keep only the column 'self.column_text'
        # WARNING : self.column_text (int) is now the column number of self.column_text (str) in self.data
        column_text = list(data[[self.column_text]].columns).index(self.column_text)
        if self.column_text_b is not None:
            data_ = data[[self.column_text, self.column_text_b]]
            column_text = list(data_.columns).index(self.column_text)
            column_text_b = list(data_.columns).index(self.column_text_b)
        else:
            data_ = data[[self.column_text]]
            column_text = list(data_.columns).index(self.column_text)
            column_text_b = None

        return data_, Y, column_text, column_text_b

    def split_data(self, data, Y, frac_trainset):
        """ split data, Y -> X_train, X_test, Y_train, Y_test
        Args:
            data (DataFrame) data with the column_text
            Y (DataFrame) data with target columns
            frac_trainset (float) fraction for training set
        Return:
            X_train (DataFrame) train data with column text
            doc_spacy_data_train (array) train documents from column_text preprocessed by spacy
            Y_train (DataFrame) train data with target columns
            X_test (DataFrame) test data with column text
            Y_test (DataFrame) test data with target columns
        """
        # DEBUG
        if self.debug:
            logger.info("\n DEBUG MODE : only a small portion is use for training set")
            train_data = data.sample(n=min(50, len(data)), random_state=self.seed)
        else:
            train_data = data.sample(frac=frac_trainset, random_state=self.seed)

        # Train set
        X_train = train_data.copy()
        logger.info("\nTraining set size : {}".format(len(X_train)))
        if Y is not None:
            Y_train = Y.loc[train_data.index, :]
        else:
            Y_train = None

        # Test set
        if self.frac_trainset < 1:
            test_data = data.drop(train_data.index)
            X_test = test_data.copy()
            if Y is not None:
                Y_test = Y.drop(train_data.index)
            else:
                Y_test = None
            logger.info("Test set size : {}".format(len(X_test)))
        else:
            X_test, doc_spacy_data_test, Y_test = None, None, None
            logger.info("Test set size : 0")

        return X_train, Y_train, X_test, Y_test

    def create_validation(self, dataset_val):
        """ separate column text and target -> X and Y
        Args:
            dataset_val (DataFrame) validation dataset
        Return:
            dataset_val_copy (DataFrame) dataset_val from input with the column_text and without target columns
            Y_val (DataFrame) dataset_val from input with only target columns
            folds (List) length 1, format = [('all', [all index of dataset_val])], 'all' means that all train set
                        will be used for training and validated on dataset_val
        """

        if len([col for col in self.target if col in dataset_val.columns]) > 0:
            Y_val = dataset_val[[col for col in self.target if col in dataset_val.columns]]
            dataset_val_copy = dataset_val.drop([col for col in self.target if col in dataset_val.columns], axis=1)
        else:
            dataset_val_copy = dataset_val.copy()
            Y_val = None

        folds = [('all', [i for i in range(Y_val.shape[0])])]

        return dataset_val_copy, Y_val, folds

    def create_cross_validation(self, X_train, Y_train):
        """ Create Cross-validation scheme
        Cross-validation split in self.nfolds but train only on self.nfolds_train chosen randomly
        Args:
            X_train (DataFrame)
            Y_train (DataFrame)
        Return:
            folds (List[tuple]) list of length self.nfolds_train with tuple (train_index, val_index)
        """
        # Cross-validation split in self.nfolds but train only on self.nfolds_train chosen randomly
        rd.seed(self.seed)
        fold_to_train = rd.sample([i for i in range(self.nfolds)], k=max(min(self.nfolds_train, self.nfolds), 1))

        if self.column_text_b is None:
            if self.cv_strategy == "StratifiedKFold" and Y_train is not None:
                skf = StratifiedKFold(n_splits=self.nfolds, random_state=self.seed, shuffle=True)
                folds_sklearn = skf.split(np.array(Y_train), np.array(Y_train))
            else:
                if Y_train is None:
                    kf = KFold(n_splits=self.nfolds, random_state=self.seed, shuffle=True)
                    folds_sklearn = kf.split(np.array(X_train))
                else:
                    kf = KFold(n_splits=self.nfolds, random_state=self.seed, shuffle=True)
                    folds_sklearn = kf.split(Y_train)
        else:
            org = X_train.copy()
            unique_comments = np.unique(np.concatenate([X_train.iloc[:, 0], X_train.iloc[:, 1]]))
            comment_to_fold = {}

            kf_gen = KFold(self.nfolds, random_state=self.seed).split(unique_comments)
            for fold, (_, comments_idx) in enumerate(kf_gen):
                for comment in unique_comments[comments_idx]:
                    comment_to_fold[comment] = fold

            org['A_fold'] = org.iloc[:, 0].map(comment_to_fold)
            org['B_fold'] = org.iloc[:, 1].map(comment_to_fold)

            folds_sklearn = []
            for fold in range(self.nfolds):
                train = org[(org.A_fold != fold) & (org.B_fold != fold)]
                valid = org[(org.A_fold == fold) & (org.B_fold == fold)]
                folds_sklearn.append((list(train.index), list(valid.index)))

        folds = []
        for num_fold, (train_index, val_index) in enumerate(folds_sklearn):
            if num_fold not in fold_to_train:
                continue
            folds.append((train_index, val_index))

        return folds

    def get_datasets(self, data, doc_spacy_data, dataset_val=None, doc_spacy_data_val=None):
        """ Use previous function of the class to prepare all needed dataset
        Args:
            data (DataFrame)
            doc_spacy_data (array) documents from column_text data preprocessed by spacy
            dataset_val (DataFrame)
            doc_spacy_data_val (array) documents from column_text dataset_val preprocessed by spacy
        """

        data, Y, column_text, column_text_b = self.separate_X_Y(data)

        if dataset_val is None:
            X_train, Y_train, X_test, Y_test = self.split_data(data, Y, self.frac_trainset)
            X_val, Y_val = None, None
            folds = self.create_cross_validation(X_train, Y_train)
        else:
            # if a validation dataset is provided, data is not split in train/test and validation data will
            # also be the test set -> frac_trainset = 1
            frac_trainset = 1
            X_train, Y_train, X_test, Y_test = self.split_data(data, Y, frac_trainset)
            X_val, Y_val, folds = self.create_validation(dataset_val)
            X_test, Y_test = X_val, Y_val

        return column_text, column_text_b, X_train, Y_train, X_val, Y_val, X_test, Y_test, folds