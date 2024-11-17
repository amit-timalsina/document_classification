from abc import ABC, abstractmethod

import torch


class BaseTokenizer(ABC):
    @abstractmethod
    def process_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        pass
