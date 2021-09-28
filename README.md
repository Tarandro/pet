# Pattern-Exploiting Training (PET)

This repository contains the code for [Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676) and [It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners](https://arxiv.org/abs/2009.07118). The papers introduce pattern-exploiting training (PET), a semi-supervised training procedure that reformulates input examples as cloze-style phrases. In low-resource settings, PET and iPET significantly outperform regular supervised training, various semi-supervised baselines and even GPT-3 despite requiring 99.9% less parameters. The iterative variant of PET (iPET) trains multiple generations of models and can even be used without any training data.

<table>
    <tr>
        <th>#Examples</th>
        <th>Training Mode</th>
        <th>Yelp (Full)</th>
        <th>AG's News</th>
        <th>Yahoo Questions</th>
        <th>MNLI</th>
    </tr>
    <tr>
        <td rowspan="2" align="center"><b>0</b></td>
        <td>unsupervised</td>
        <td align="right">33.8</td>
        <td align="right">69.5</td>
        <td align="right">44.0</td>
        <td align="right">39.1</td>
    </tr>
    <tr>
        <td>iPET</td>
        <td align="right"><b>56.7</b></td>
        <td align="right"><b>87.5</b></td>
        <td align="right"><b>70.7</b></td>
        <td align="right"><b>53.6</b></td>
    </tr>
    <tr>
        <td rowspan="3" align="center"><b>100</b></td>
        <td>supervised</td>
        <td align="right">53.0</td>
        <td align="right">86.0</td>
        <td align="right">62.9</td>
        <td align="right">47.9</td>
    </tr>
    <tr>
        <td>PET</td>
        <td align="right">61.9</td>
        <td align="right">88.3</td>
        <td align="right">69.2</td>
        <td align="right">74.7</td>
    </tr>
    <tr>
        <td>iPET</td>
        <td align="right"><b>62.9</b></td>
        <td align="right"><b>89.6</b></td>
        <td align="right"><b>71.2</b></td>
        <td align="right"><b>78.4</b></td>
    </tr>
</table>

## Requirements

Python 3.7 or later with all requirements.txt dependencies installed. To install run:
```python
$ pip install -r requirements.txt
```

## Minimum codes

```python
from main_pet import *

parameters_to_update = {
                      "method": 'pet',  # ['pet', 'ipet', 'sequence_classifier']                                 
                      "model_type": "camembert",                               
                      "model_name_or_path": "camembert-base",           
                      "path_data": "/content/train.csv",
                      "path_data_validation": "",
                      "path_data_unlabeled": "/content/unlabeled_plus.csv",       
                      "outdir": "/content/logs/",              
                      "debug": False,
                      "seed": 15, 
                      "column_text": "text_fr",
                      "target": "sentiment",  
                      "frac_trainset": 1, 
                      "nfolds": 5, 
                      "nfolds_train": 5, 
                      "cv_strategy": "KFold", 

                      "pattern_ids": [0],                             
                      "pet_repetitions": 1,                                    
                      "pet_max_seq_length": 100,                     
                      "pet_num_train_epochs": 2,                               
                      "sc_max_seq_length": 100,                        
                      "sc_num_train_epochs": 4,                                
                      "sc_max_steps": -1,                                      
                      "metrics": ["acc", "f1-macro", "f1-weighted"],
                      "reduction": "mean",                                            
                      "labels": ["negative", "positive", "neutral"],
                      "verbalizer": {"negative": ["négatif"], "positive": ["positif"], "neutral":["neutre"]},
                      "pattern": {0: "Le sentiment du texte est MASK : TEXT_A"}            # , 1: "TEXT_A . C'est MASK", 2: "TEXT_A .Le titre est MASK"
                      }
# update parameters :
args = Flags().update(parameters_to_update)
petnlp = Pet(args)
```
Preprocessing, split train/test and Training + Validation:
```python
petnlp.data_preprocessing()
petnlp.train()
# validation leaderboard :
leaderboard_val = petnlp.get_leaderboard(dataset='val')
```
Prediction on test set for all models:
```python
y_test_pred, y_test_confidence, result_dict = petnlp.prediction(on_test_data=True)
# test leaderboard :
leaderboard_test = petnlp.get_leaderboard(dataset='test')
```

## Usage examples

To find out how to work with PET:

- [Tutorial 1: PET_FinancialPhraseBank.ipynb](notebooks/PET_FinancialPhraseBank.ipynb) pipeline using Pet method

- [Tutorial 2: iPET_zero_shot_FinancialPhraseBank.ipynb](notebooks/iPET_zero_shot_FinancialPhraseBank.ipynb) pipeline using iPet method

Careful : You can use PET method in the same way as AutoNLP but for iPET the code is different

For more information about PET : https://github.com/timoschick/pet

## Code from :

    @article{schick2020exploiting,
      title={Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference},
      author={Timo Schick and Hinrich Schütze},
      journal={Computing Research Repository},
      volume={arXiv:2001.07676},
      url={http://arxiv.org/abs/2001.07676},
      year={2020}
    }

    @article{schick2020small,
      title={It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners},
      author={Timo Schick and Hinrich Schütze},
      journal={Computing Research Repository},
      volume={arXiv:2009.07118},
      url={http://arxiv.org/abs/2009.07118},
      year={2020}
    }
