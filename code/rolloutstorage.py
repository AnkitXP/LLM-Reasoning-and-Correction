from torchtyping import TensorType
from dataclasses import dataclass
from typing import Iterable

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch

@dataclass
class SCoRERLElement:
    first_query_tensors: TensorType["query_size"]
    first_response_logits: TensorType["response_size", "vocab_size"]
    first_response_kl_divs: TensorType["response_size"]
    first_attempt_rewards: TensorType

    second_query_tensors: TensorType["query_size" + "response_size"]
    second_response_logits: TensorType["response_size", "vocab_size"]
    second_response_kl_divs: TensorType["response_size"]
    second_attempt_rewards: TensorType

@dataclass
class SCoRERLBatch:
    first_query_tensors: TensorType["batch_size", "query_size"]
    first_response_logits: TensorType["batch_size", "response_size", "vocab_size"]
    first_response_kl_divs: TensorType["batch_size", "response_size"]
    first_attempt_rewards: TensorType["batch_size"]
    
    second_query_tensors: TensorType["batch_size", "query_size" + "response_size"]
    second_response_logits: TensorType["batch_size", "response_size", "vocab_size"]
    second_response_kl_divs: TensorType["batch_size", "response_size"]
    second_attempt_rewards: TensorType["batch_size"]

class SCoRERolloutStorage:
    def __init__(self, tokenizer):
        super().__init__()
        self.pad_token_id = tokenizer.pad_token_id
        self.history: Iterable[SCoRERLElement] = []

    def push(self, exps: Iterable[SCoRERLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def __getitem__(self, index: int) -> SCoRERLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(self, batch_size: int, shuffle: bool) -> DataLoader:
        
        def collate_fn(elems: Iterable[SCoRERLElement]):
            
            return SCoRERLBatch(
                pad_sequence(
                    [elem.first_query_tensors for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.first_response_logits for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.first_response_kl_divs for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                torch.tensor([elem.first_attempt_rewards for elem in elems]),
                pad_sequence(
                    [elem.second_query_tensors for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.second_response_logits for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.second_response_kl_divs for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                torch.tensor([elem.second_attempt_rewards for elem in elems]),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=True)