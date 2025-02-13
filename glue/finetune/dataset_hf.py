from __future__ import annotations

import torch
import json
from typing import TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    import pathlib


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file: pathlib.Path, task: str) -> None:
        load = partial(self.load_file, input_file)

        match task:
            case "boolq":
                load("question", "passage")
            case "cola":
                load("sentence")
            case "mnli":
                load("premise", "hypothesis")
            case "mrpc":
                load("sentence1", "sentence2")
            case "multirc":
                load("question", "answer", "paragraph", "Question: {} Answer: {}")
            case "qnli":
                load("question", "sentence")
            case "qqp":
                load("question1", "question2")
            case "rte":
                load("sentence1", "sentence2")
            case "sst2":
                load("sentence")
            case "wsc":
                load("span2_text", "span1_text", "text", "Does \"{}\" refer to \"{}\" in this passage?")
            case _:
                raise ValueError("This is not an implemented task! Please implement it!")

    def load_file(self, input_file: pathlib.Path, key1: str, key2: str | None = None, key3: str | None = None, template: str | None = None) -> None:
        self.texts = []
        self.labels = []

        with input_file.open("r") as file:
            for line in file:
                data = json.loads(line)

                if key2 is not None:
                    if template is not None:
                        assert key3 is not None
                        self.texts.append((template.format(data[key1], data[key2]), data[key3]))
                    else:
                        self.texts.append((data[key1], data[key2]))
                else:
                    self.texts.append(data[key1])

                self.labels.append(data["label"])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        text = self.texts[index]
        label = self.labels[index]

        return text, label


def collate_function(tokenizer: PreTrainedTokenizerBase, causal: bool, max_length: int, data: list[tuple[str, str] | int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    texts = []
    labels = []

    for text, label in data:
        texts.append(text)
        labels.append(label)

    labels = torch.tensor(labels, dtype=torch.long)
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    if causal:
        attention_mask = encodings.attention_mask.unsqueeze(1).repeat(1, encodings.attention_mask.size(-1), 1).tril(diagonal=0)
    else:
        attention_mask = encodings.attention_mask

    return encodings.input_ids, attention_mask, labels


class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: pathlib.Path, task: str) -> None:
        load = partial(self.load_file, input_file)

        match task:
            case "boolq":
                load("question", "passage")
            case "cola":
                load("sentence")
            case "mnli":
                load("premise", "hypothesis")
            case "mrpc":
                load("sentence1", "sentence2")
            case "multirc":
                load("question", "answer", "paragraph", "Question: {} Answer: {}")
            case "qnli":
                load("question", "sentence")
            case "qqp":
                load("question1", "question2")
            case "rte":
                load("sentence1", "sentence2")
            case "sst2":
                load("sentence")
            case "wsc":
                load("span2_text", "span1_text", "text", "Does \"{}\" refer to \"{}\" in this passage?")
            case _:
                raise ValueError("This is not an implemented task! Please implement it!")

    def load_file(self, input_file: pathlib.Path, key1: str, key2: str | None = None, key3: str | None = None, template: str | None = None) -> None:
        self.texts = []

        with input_file.open("r") as file:
            for line in file:
                data = json.loads(line)

                if key2 is not None:
                    if template is not None:
                        assert key3 is not None
                        self.texts.append((template.format(data[key1], data[key2]), data[key3]))
                    else:
                        self.texts.append((data[key1], data[key2]))
                else:
                    self.texts.append(data[key1])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        text = self.texts[index]

        return text


def predict_collate_function(tokenizer: PreTrainedTokenizerBase, causal: bool, max_length: int, data: list[tuple[str, str] | int]) -> tuple[torch.Tensor, torch.Tensor]:

    texts = []

    for text, label in data:
        texts.append(text)

    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    if causal:
        attention_mask = encodings.attention_mask.unsqueeze(1).repeat(1, encodings.attention_mask.size(-1), 1).tril(diagonal=0)
    else:
        attention_mask = encodings.attention_mask

    return encodings.input_ids, attention_mask
