"""
This script can be used to generate predictions on your own data using the pre-trained CRF model.

The input should be a pickled pandas DataFrame with the following columns. The script transforms the DataFrame to the required `python-crfsuite` format. For more information about the features, please refer to the repo's wiki.

COLUMNS:
'sentence_id', 'words', 'stem', 'pos', 'pattern_1', 'lemma_1', 'chunk_2', 'suffix_5', 'pattern_0', 'lemma_2', 'pattern_prefix', 'prefix_4', 'prefix_3', 'suffix_4', 'prefix_5', 'suffix_3', 'pos_1', 'chunk_1', 'pos_2', 'chunk_0', 'pattern_-1', 'pos_-1', 'lemma_-1', 'chunk_-1', 'pos_-2', 'lemma_-2', 'chunk_-2'

NOTE
====
My HEDGEhog repository (https://github.com/vanboefer/hedgehog) contains a transformer-based model that performs the same multi-class classification task with better performance and only tokens as features.
"""


import argparse
import pickle
import pandas as pd


def predict(model, data, predictions):
    """
    Generate predictions on `data` using the CRF `model`.
    Write the generated predictions (list of lists od strings) to a pickle file.

    Parameters
    ----------
    model: str
        path to a pickled model
    data: str
        path to a pickled pandas DataFrame with features
    predictions: str
        path to save the pickled output

    Returns
    -------
    None
    """

    # load data
    data = pd.read_pickle(data).fillna('')

    # convert features to `python-crfsuite` format
    def sent2features(df):
        return df.drop(['sentence_id', 'labels'], axis=1).to_dict(orient='records')

    X = [lst for lst in data.groupby('sentence_id').apply(sent2features).to_list()]

    # load model
    with open(model,'rb') as f:
        crf = pickle.load(f)

    # generate predictions
    y_pred = crf.predict(X)

    # save predictions
    with open(predictions,'wb') as f:
        pickle.dump(y_pred, f)

    print(f"Predictions are saved at {predictions}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', default='../model/crf.pkl')
    argparser.add_argument('--data', default='../data/train_dev_test/test.pkl')
    argparser.add_argument('--output', default='../data/output/predictions.pkl')
    args = argparser.parse_args()

    predict(
        args.model,
        args.data,
        args.output,
    )
