import argparse

import os
import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", help="path to preprocessed datasets", default="../datasets/"
    )
    parser.add_argument(
        "-g", help="path to the dataset https://github.com/ee-2/in_formal_sentences/tree/master", default="../german_dataset/"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    DATASETS_PATH = args.d
    GERMAN_DATASET_PATH = args.g

    os.makedirs(DATASETS_PATH, exist_ok=True)

    splits = ['train.csv', 'test.csv']
    pavlick_df = pd.concat([pd.read_csv("hf://datasets/osyvokon/pavlick-formality-scores/" + split) for split in splits])
    pavlick_df['score'] = (pavlick_df.avg_score + 3) / 6
    pavlick_df['text'] = pavlick_df.sentence
    pavlick_sentences = set(pavlick_df.sentence)
    pavlick_df[['score', 'text']].to_csv(DATASETS_PATH + "pavlick.csv", index=False, sep='\t')

    oishooo_df = pd.read_csv("hf://datasets/oishooo/formality_classification/formality_dataset.csv")
    oishooo_df = oishooo_df[[t not in pavlick_sentences for t in oishooo_df.text]]
    scores = []
    for formality in oishooo_df.formality_label:
        if formality == "informal":
            scores.append(0)
        if formality == "neutral":
            scores.append(0.5)
        if formality == "formal":
            scores.append(1)
    oishooo_df['score'] = scores
    oishooo_df[['score', 'text']].to_csv(DATASETS_PATH + "labeled.csv", index=False, sep='\t')

    german_df = pd.read_csv(GERMAN_DATASET_PATH + "in_formal_sentences.tsv", sep='\t')
    german_df.score = (german_df.score + 1) / 2
    german_df[['score', 'text']].to_csv(DATASETS_PATH + "german.csv", index=False, sep='\t')