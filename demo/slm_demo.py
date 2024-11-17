from pathlib import Path

import typer

from language_model.config import ModelConfig
from language_model.create_dataset import TextWithBBoxDatasetPreparer
from language_model.finetune import (
    SLMModelTrainer,
)
from language_model.predict import SLMPredictor
from language_model.slm_model import LanguageModel
from language_model.tokenizers.text_with_layout import TextWithLayoutTokenizer
from language_model.utils import get_device
from logger import logger

app = typer.Typer()


@app.command()
def finetune(  # noqa: PLR0913
    ocr_json_path: Path = typer.Argument(..., help="Path to OCR JSON files"),  # noqa: B008
    save_path: Path = typer.Argument(..., help="Path to save fine-tuned model"),  # noqa: B008
    processed_data_path: Path = typer.Argument(..., help="Path to save/load processed data"),  # noqa: B008
    batch_size: int = typer.Option(2, help="Batch size for training"),
    num_epochs: int = typer.Option(5, help="Number of epochs for training"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate for training"),
) -> None:
    """
    Finetune the model.

    Example usuage:
    ```
    python slm_demo.py finetune ocr_jsons_tesseract fine_tuned_bert_classification \
        processed_ocr_data.pkl
    ```
    """
    # dataset_preparer = TextOnlyDatasetPreparer()
    dataset_preparer = TextWithBBoxDatasetPreparer()
    train_dataloader, val_dataloader, label_encoder = dataset_preparer.prepare_data(
        ocr_json_path,
        batch_size,
        processed_data_path,
    )
    num_labels = len(label_encoder.classes_)

    device = get_device()
    logger.info(f"Using device: {device}")

    model = LanguageModel(
        config=ModelConfig(
            model_name="microsoft/layoutlm-base-uncased",
            num_labels=num_labels,
        ),
        device=device,
    )
    # tokenizer = TextTokenizer(
    #     tokenizer=model.tokenizer,
    #     max_length=model.config.max_length,
    #     device=device,
    # )

    tokenizer = TextWithLayoutTokenizer(
        tokenizer=model.tokenizer,
        max_length=model.config.max_length,
        device=device,
    )
    model_trainer = SLMModelTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
    )

    model_trainer.train(
        train_dataloader,
        val_dataloader,
        num_epochs,
    )

    model_trainer.model.save(save_path)

    logger.info(f"Model and tokenizer saved to {save_path}")


@app.command()
def predict(
    model_path: Path = typer.Argument(..., help="Path to the fine-tuned model"),  # noqa: B008
    processed_data_path: Path = typer.Argument(..., help="Path to processed data"),  # noqa: B008
    file_to_predict: Path = typer.Argument(..., help="Path to the file to predict"),  # noqa: B008
) -> None:
    """
    Predict using the fine-tuned model.

    Example usuage:

    ```
    python slm_demo.py predict fine_tuned_bert_classification processed_ocr_data.pkl \
        ocr_jsons_tesseract/form_1040/0fe3a054ec90433480dac54da62b595e.json
    ```
    """
    device = get_device()
    model = LanguageModel(
        config=ModelConfig(
            model_name=str(model_path),
            num_labels=2,
        ),
        device=device,
    )
    predictor = SLMPredictor(model, processed_data_path)
    prediction = predictor.predict_file(
        file_to_predict,
    )
    logger.info(f"Prediction: {prediction}")


if __name__ == "__main__":
    app()
