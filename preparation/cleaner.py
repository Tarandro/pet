import pandas as pd

import re
import string

import log
logger = log.get_logger('root')


####################
# clean and spacy preprocessing
####################

def small_clean_text(text):
    """ Clean text : Remove '\n', '\r', URL, '’', numbers and double space
    Args:
        text (str)
    Return:
        text (str)
    """
    text = re.sub('\n', ' ', text)
    text = re.sub('\r', ' ', text)

    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('’', ' ', text)

    #text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub(' +', ' ', text)
    return text


def clean_text(text):
    """ Clean text : lower text + Remove '\n', '\r', URL, '’', numbers and double space + remove Punctuation
    Args:
        text (str)
    Return:
        text (str)
    """
    text = str(text).lower()

    text = re.sub('\n', ' ', text)
    text = re.sub('\r', ' ', text)

    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuation
    text = re.sub('’', ' ', text)

    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub(' +', ' ', text)

    return text


#############################
#############################
#############################

class Preprocessing_NLP:
    """Class for compile full pipeline of NLP preprocessing task.
            Preprocessing_NLP steps:
                - (Optional) can apply a small cleaning on text column
    """

    def __init__(self, data, flags_parameters):
        """
        Args:
            data (Dataframe)
            flags_parameters : Instance of Flags class object
        From flags_parameters:
            column_text (str) : name of the column with texts (only one column)
            apply_small_clean (Boolean) step 1 of transform
        """
        self.column_text = flags_parameters.column_text
        self.apply_small_clean = flags_parameters.apply_small_clean

        assert isinstance(data, pd.DataFrame), "data must be a DataFrame type"
        assert self.column_text in data.columns, 'column_text specifying the column with text is not in data'

    def transform(self, data):
        """ Fit and transform self.data :
            + can apply a small cleaning on text column (self.apply_small_clean)
            + preprocess text column with nlp.pipe spacy (self.apply_spacy_preprocessing)
            + replace Named entities  (self.apply_entity_preprocessing)
        Return:
            data_copy (DataFrame) data with the column_text
            doc_spacy_data (array) documents from column_text preprocessed by spacy
        """

        data_copy = data.copy()

        if self.apply_small_clean:
            logger.info("- Apply small clean of texts...")
            data_copy[self.column_text] = data_copy[self.column_text].apply(lambda text: small_clean_text(text))

        return data_copy
