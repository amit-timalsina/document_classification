from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from language_model.ocr_dataset import BoundingBox


def custom_collate_fn(batch):
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
    bbox_tensors = pad_sequence(bbox_tensors, batch_first=True)

    return {
        "texts": texts,
        "labels": torch.tensor(labels),
        "bboxes": bbox_tensors,
    }
