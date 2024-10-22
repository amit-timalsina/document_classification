import json
import re
from collections import defaultdict
from pathlib import Path

import fasttext  # type: ignore[import-untyped]

from logger import logger


def accuracy_matrix_tf(test_file_path: Path, model: fasttext.FastText._FastText) -> dict:
    """
    Calculate accuracy matrix for a FastText model.

    Args:
        test_file_path (Path): Path to the test file.
        model (fasttext.FastText): FastText model to evaluate.

    Returns:
        dict: Accuracy matrix containing the number of samples and accuracy for each label.

    """
    with test_file_path.open("r") as f:
        lines = f.readlines()

    # read file and make a predication
    actual_label = [re.search(r"__label\w+", line.strip()).group() for line in lines]  # type: ignore[union-attr]
    texts = [re.sub(r"__label\w+", "", line.strip()) for line in lines]
    predicted_labels = model.predict(texts)

    # clean get accuracy matrix
    actual_label = [j.replace("__label__", "") for j in actual_label]
    predicted_labels = [i[0].replace("__label__", "") for i in predicted_labels[0]]
    acc_label_dict = defaultdict(list)

    for i, j in zip(actual_label, predicted_labels):
        acc_label_dict[i].append(i == j)

    # get data
    return {i: {"n": len(j), "accuarcy": sum(j) / len(j)} for i, j in acc_label_dict.items()}


def evaluate(dataset_dir_path: Path, model_path: Path, save_metrices_path: Path) -> None:
    """Evaluate a fasttext model."""
    logger.info("Calculating metrics in test file...")
    model = fasttext.load_model(str(model_path))
    test_file_path = dataset_dir_path / "test.txt"
    result = model.test(str(test_file_path))
    n, p, r = result
    labels = accuracy_matrix_tf(test_file_path, model)

    model_json = {}
    model_json["test_metrices"] = {
        "summary": {"n": n, "p": p, "r": r},
        "labels": labels,
    }

    logger.info("Calculating metrics in train file...")
    train_file_path = dataset_dir_path / "train.txt"
    result = model.test(str(train_file_path))
    n, p, r = result
    labels = accuracy_matrix_tf(train_file_path, model)
    model_json["train_metrices"] = {
        "summary": {"n": n, "p": p, "r": r},
        "labels": labels,
    }

    json.dump(model_json, save_metrices_path.open("w"))
