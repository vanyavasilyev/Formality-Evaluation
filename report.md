# Formality Evaluation Report

A report for a task of creating a formality detection model.

## Datasets

While there are multiple datasets that are suitable for the task, some of them cannot be employed. One of the largest datasets is GYAFC, designed for formality transfer task. However, it is not publicly available. CoCoA-MT created for translation task. Unfortunately, it does not have formality labels for source language. The final list of datasets is the following:

- **Pavlick's** - the dataset from https://cs.brown.edu/people/epavlick/papers/formality.pdf. It is also possible to employ a part of the dataset from their https://cs.brown.edu/people/epavlick/papers/style_for_paraphrasing.pdf.

- LLM-**lableled** dataset from https://huggingface.co/datasets/oishooo/formality_classification. As it includes sentences from the previous one, it was cleared from them.

- **German** - https://aclanthology.org/2023.findings-eacl.42.pdf provides us with a German-language dataset, that may be used to evaluate some models.

## Evaluation

All the datasets have formality data in form of either score or labels. All these values may be rescaled into [0,1], where 0 correcsponds to informal sentences, 1 to fromal, and 0.5 to neutral ones. However, it was not clear from the paper whether the last property holds for the swcond Pavlick's dataset. For such type of data there are not many evaluation approaches. The first one is to interpret values as labels, e.g. >0.5 for formal texts, <0.5 for informal. It allows us to use basic classification metrics, such as accuracy and f1-score. Another way found in some papers is Spearman correlation, as it is also important for us to evaluate model in terms of scores, not predictions, letting us measure ability of a model to get exact formality level, e.g., let us maintain formality level no lower than 0.8, where 0.8 is something meaningful. 

## Models

First model was taken from https://arxiv.org/pdf/2204.08975. It is a Neural-Network-based model. The second one is also based on a NN. I found it from https://huggingface.co/. However, it is still possible to evalute non-DL models. As multiple papers (https://web.ntpu.edu.tw/~myday/doc/IRI2022/IEEE_IRI2022_Proceedings/pdfs/IRI2022-2biJIxjybiQ3DOmA6IkB2x/660300a001/660300a001.pdf, https://web.ntpu.edu.tw/~myday/doc/IRI2022/IEEE_IRI2022_Proceedings/pdfs/IRI2022-2biJIxjybiQ3DOmA6IkB2x/660300a001/660300a001.pdf) claim that DL models perform better, I decided to compare such models.

## Result

My laptop did not allow me to validate models on the whole datasets. The results on small samples are 