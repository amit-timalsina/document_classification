from pathlib import Path

import fasttext  # type: ignore[import-untyped]

from document_classification.logger import logger


def train_fasttext(dataset_dir_path: Path, output_model_path: Path) -> None:
    """Train a supervised classification fasttext model."""
    logger.debug("Training fasttext model...")
    model = fasttext.train_supervised(
        input=str(dataset_dir_path / "train.txt"),
        autotuneValidationFile=str(dataset_dir_path / "validation.txt"),
        seed=42,
    )

    logger.debug("Training fasttext model completed.")
    logger.debug(
        f"Best parameters: epochs={model.epoch}, lr={model.lr}, wordNgrams={model.wordNgrams}",
        f"loss={model.loss}, dim={model.dim}, minCount={model.minCount}",
    )
    model.save_model(str(output_model_path))
