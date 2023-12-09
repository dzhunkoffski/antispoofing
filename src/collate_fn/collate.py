import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[Tuple]):
    """
    Collate and pad fields in dataset items
    """

    audio_batch = []
    label_batch= []

    for item in dataset_items:
        audio, label = item
        audio_batch.append(audio)
        label_batch.append(label)

    return {
        'audio': audio_batch,
        'label': label_batch
    }
