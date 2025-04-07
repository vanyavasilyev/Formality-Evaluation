import argparse
from pyexpat import model

import pandas as pd
import tqdm
from sklearn.metrics import classification_report
import scipy.stats as ss
import numpy as np
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", help="path to preprocessed dataset", default="../datasets/pavlick.csv"
    )
    parser.add_argument(
        "-m", help="model name", default="s-nlp/xlmr_formality_classifier"
    )
    parser.add_argument(
        "-s", help="number of samples", default=0
    )
    parser.add_argument(
        "--method", help="evaluation metric", default="both"
    )
    return parser.parse_args()


def print_evaluation(model, tokenizer, df: pd.DataFrame, method: str = "both", samples: int = 0):
    if method not in ["classification", "spearman", "both"]:
        print("Method unknown")
    if samples == 0:
        dataset = df
    else:
        dataset = df.sample(min(samples, len(df)))
    predictions = []
    for text in tqdm.tqdm(dataset.text):
        texts = [text]
        encoding = tokenizer(
            texts,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        output = model(**encoding)
        logits = output.logits.detach().numpy()
        if logits.shape[1] < 2:
            logit = logits[0][0]
            exp = np.exp(logit)
            predictions.append(float(exp / (1 + exp)))
        else:
            predictions.append(output.logits.softmax(dim=1)[:,0].item())
    print("Evaluation of model:")
    print(predictions)
    if method in ["classification", "both"]:
        y_true = list((dataset.score > 0.5).astype(int))
        y_pred = list((np.array(predictions) > 0.5).astype(int))
        print("Classifiacation report:")
        print(classification_report(y_true, y_pred, target_names=["informal", "formal"]))
    if method in ["spearman", "both"]:
        print(f"Spearman correlation is {ss.spearmanr(dataset.score, predictions).statistic}")


def get_model(name: str):
    if name == "s-nlp/xlmr_formality_classifier":
        tokenizer = XLMRobertaTokenizerFast.from_pretrained('s-nlp/xlmr_formality_classifier')
        model = XLMRobertaForSequenceClassification.from_pretrained('s-nlp/xlmr_formality_classifier')
        return model, tokenizer
    if name == "Harshveer/autonlp-formality_scoring_2-32597818":
        tokenizer = AutoTokenizer.from_pretrained("Harshveer/autonlp-formality_scoring_2-32597818")
        model = AutoModelForSequenceClassification.from_pretrained("Harshveer/autonlp-formality_scoring_2-32597818")
        return model, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        return model, tokenizer
    except Exception:
        return None, None


if __name__ == "__main__":
    args = _parse_args()
    df = pd.read_csv(args.d, sep='\t')
    model, tokenizer = get_model(args.m)
    if model is None:
        print("Unsupported model")
    print_evaluation(model, tokenizer, df, args.method, int(args.s))
