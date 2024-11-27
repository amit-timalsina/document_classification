import numpy as np
import pandas as pd
from google.cloud import vision

from document_classification.ocr.config import ocr_config

from .image_processor import GoogleVisionImageProcessor


class GoogleVisionOCRProcessor:
    """A class for processing images using Google Vision API."""

    def __init__(self, client: vision.ImageAnnotatorClient) -> None:
        """Initialize the Google Vision OCR processor."""
        self.client = client

    def perform_ocr(self, image: np.ndarray) -> pd.DataFrame:
        """Take an image and return OCR results."""
        vision_image = GoogleVisionImageProcessor.construct_vision_image(image)
        response = self.client.document_text_detection(image=vision_image)
        ocr_df = self._process_response(response)
        return self._standardize_output(ocr_df)

    def _process_response(self, response: vision.AnnotateImageResponse) -> pd.DataFrame:
        data = []
        for page_i, page in enumerate(response.full_text_annotation.pages):
            for block_i, block in enumerate(page.blocks):
                inter_line = 0
                for paragraph_i, paragraph in enumerate(block.paragraphs):
                    for word_i, word in enumerate(paragraph.words):
                        word_text = "".join([s.text for s in word.symbols])
                        detected_break = getattr(
                            word.symbols[-1].property,
                            "detected_break",
                            None,
                        )
                        space_type = detected_break.type if detected_break else None
                        confidence = np.mean([s.confidence for s in word.symbols])
                        temp_data = {
                            "page": page_i,
                            "block": block_i,
                            "paragraph": paragraph_i,
                            "line": inter_line,
                            "word": word_i,
                            "word_text": word_text,
                            "space_type": space_type,
                            "confidence": confidence,
                        }

                        temp_data.update(self.get_vertices(word, "word"))
                        data.append(temp_data)
                        if space_type is not None and space_type > 1:
                            inter_line += 1

        if not data:
            return pd.DataFrame({})

        ocr_df = pd.DataFrame(data)
        ocr_df["index_sort"] = ocr_df.index

        # compatibility issues
        ocr_df["Text"] = ocr_df["word_text"]
        ocr_df["text"] = ocr_df["word_text"]
        ocr_df["x0"] = ocr_df["word_x"]
        ocr_df["y0"] = ocr_df["word_y"]
        ocr_df["x1"] = ocr_df["word_x"]
        ocr_df["y1"] = ocr_df["word_y"]
        ocr_df["x2"] = ocr_df["word_x2"]
        ocr_df["y2"] = ocr_df["word_y2"]

        if not ocr_df.empty:
            ocr_df = ocr_df[(ocr_df.x0 < ocr_df.x2) & (ocr_df.y0 < ocr_df.y2)]

        return ocr_df

    @staticmethod
    def get_vertices(text: vision.Word, prefix: str = "word") -> dict:
        """Get vertices."""
        try:
            box = text.bounding_poly
        except AttributeError:
            box = text.bounding_box

        xs = [p.x for p in box.vertices]
        ys = [p.y for p in box.vertices]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        data = {
            prefix + "_x": x1,
            prefix + "_y": y1,
            prefix + "_x2": x2,
            prefix + "_y2": y2,
            prefix + "_h": y2 - y1,
            prefix + "_w": x2 - x1,
        }
        data.update({f"point_x{i}": x for i, x in enumerate(xs)})
        data.update({f"point_y{i}": y for i, y in enumerate(ys)})
        return data

    def _standardize_output(self, ocr_df: pd.DataFrame) -> pd.DataFrame:
        for col in ocr_config.output_columns:
            if col not in ocr_df.columns:
                ocr_df[col] = None
        return ocr_df
