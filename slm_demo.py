from pathlib import Path

from language_model.finetune import (
    get_device,
    initialize_model,
    prepare_data,
    save_model,
    train_model,
)
from logger import logger


def main() -> None:
    """Demo for finetuning BERT on OCR data."""
    ocr_json_path = Path("ocr_jsons_tesseract")
    save_path = Path("fine_tuned_bert")
    batch_size = 2
    num_epochs = 3
    learning_rate = 2e-5

    dataloader, label_encoder = prepare_data(ocr_json_path, batch_size)
    num_labels = len(label_encoder.classes_)

    device = get_device()
    logger.info(f"Using device: {device}")

    model, tokenizer = initialize_model(num_labels)

    train_model(model, dataloader, tokenizer, device, num_epochs, learning_rate)

    save_model(model, tokenizer, save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")


if __name__ == "__main__":
    main()
