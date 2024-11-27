from pathlib import Path

import typer

from document_classification.common.utils.get_device import get_device
from document_classification.language_model.config import ModelConfig
from document_classification.language_model.dataset_preparer.text_only import (
    TextOnlyDatasetPreparer,
)
from document_classification.language_model.dataset_preparer.text_with_bbox import (
    TextWithBBoxDatasetPreparer,
)
from document_classification.language_model.predictor import SLMPredictor
from document_classification.language_model.schemas.slm_model import LanguageModel
from document_classification.language_model.tokenizers.text import TextTokenizer
from document_classification.language_model.tokenizers.text_with_layout import (
    TextWithLayoutTokenizer,
)
from document_classification.language_model.trainer import (
    SLMModelTrainer,
)
from document_classification.logger import logger

app = typer.Typer()


@app.command()
def finetune(  # noqa: PLR0913
    ocr_json_path: Path = typer.Argument(..., help="Path to OCR JSON files"),  # noqa: B008
    save_path: Path = typer.Argument(..., help="Path to save fine-tuned model"),  # noqa: B008
    processed_data_path: Path = typer.Argument(..., help="Path to save/load processed data"),  # noqa: B008
    batch_size: int = typer.Option(2, help="Batch size for training"),
    num_epochs: int = typer.Option(5, help="Number of epochs for training"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate for training"),
    use_layout: bool = typer.Option(True, help="Use layout information for training"),  # noqa: FBT001, FBT003
    model_name: str = typer.Option(
        "microsoft/layoutlm-base-uncased",
        help="Base model to use for fine-tuning",
    ),
) -> None:
    """
    Finetune the model.

    Example usuage:

    python slm_demo.py finetune ocr_jsons_tesseract fine_tuned_bert_classification \
        processed_ocr_data.pkl --use-layout
    """
    dataset_preparer = TextWithBBoxDatasetPreparer() if use_layout else TextOnlyDatasetPreparer()
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
            model_name=model_name,
            num_labels=num_labels,
        ),
        device=device,
    )
    tokenizer_class = TextWithLayoutTokenizer if use_layout else TextTokenizer
    tokenizer = tokenizer_class(
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
