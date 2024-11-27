from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from document_classification.common.schemas.bounding_box import BoundingBox


def custom_collate_fn(batch: list) -> dict:
    """
    Collates a batch of data samples for a PyTorch dataset.

    This function takes a batch of data samples, which are tuples containing text, labels, and
    bounding boxes, and returns a dictionary containing the collated data. The bounding boxes are
    padded to the maximum length in the batch, and the bounding box tensors are created and padded.
    """
    texts, labels, bboxes = zip(*batch)

    # Pad the bounding boxes to the same length
    max_bbox_len = max(len(bbox) for bbox in bboxes)
    padded_bboxes = [
        bbox + [BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0)] * (max_bbox_len - len(bbox))
        for bbox in bboxes
    ]

    # Convert bounding boxes to tensor
    bbox_tensors = [
        torch.tensor([[b.x_min, b.y_min, b.x_max, b.y_max] for b in bbox]) for bbox in padded_bboxes
    ]

    bbox_tensor = pad_sequence(bbox_tensors, batch_first=True)

    return {
        "texts": texts,
        "labels": torch.tensor(labels),
        "bboxes": bbox_tensor,
    }
