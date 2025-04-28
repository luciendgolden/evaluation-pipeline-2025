from __future__ import annotations

import json
import pathlib
from typing import Any


def decode(line: str, file_name: pathlib.Path, task: str) -> dict[str, str]:
    """This function takes a line of a JSONL file and returns a dictionary of terms to be used by the evaluation.

    Args:
        line(str): A JSONL line from a datafile.
        file_name(pathlib.Path): The file name the line comes from.
        task(str): The task we are evaluating, this tells us what needs to be imported.

    Returns:
        dict[str, str]: A dictionary with values used for evaluation
    """
    raw_dict = json.loads(line.strip())

    if task == "blimp":
        pair = decode_blimp(raw_dict, file_name)
    elif task == "ewok":
        pair = decode_ewok(raw_dict)
    else:
        raise NotImplementedError(f"The task {task} is not implemented! Please implement it or choose one of the implemented tasks.")

    return pair


def decode_blimp(raw_dict: dict[str, Any], file_name: pathlib.Path) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint of a BLiMP datafile and returns a dictionary of terms to be used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single datapoint of a BLiMP datafile.
        file_name(pathlib.Path): When no UID is mentioned, we take the file name.

    Returns:
        dict[str, str]: A dictionary with values used for evaluation
    """
    if "field" in raw_dict:
        pair = {
            "sentences": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "label": 0,
            "field": raw_dict["field"],
            "UID": raw_dict["UID"],
            "linguistics_term": raw_dict["linguistics_term"],
        }
        if pair["field"] == "syntax_semantics":  # Standardizing the style of this field
            pair["field"] = "syntax/semantics"
    else:  # For the supplemetal tasks, there is no field or UID
        pair = {
            "sentences": [raw_dict["sentence_good"], raw_dict["sentence_bad"]],
            "label": 0,
            "field": "supplemental",
            "UID": file_name.stem,
            "linguistics_term": "supplemental",
        }

    return pair


def decode_ewok(raw_dict: dict[str, Any]) -> dict[str, str]:
    """This function takes a dictionary of a single datapoint of a EWoK datafile
    and returns a dictionary of terms to be used by the evaluation.

    Args:
        raw_dict(dict[str, Any]): A dictionary from a single datapoint of a
            EWoK datafile.

    Returns:
        dict[str, str]: A dictionary with values used for evaluation
    """
    pair = {
        "sentences": [" ".join([raw_dict["Context1"], raw_dict["Target1"]]), " ".join([raw_dict["Context1"], raw_dict["Target2"]])],
        "label": 0,
        "UID": raw_dict["Domain"],
        "context_type": raw_dict["ContextType"],
        "context_contrast": raw_dict["ContextDiff"],
        "target_contrast": raw_dict["TargetDiff"],
    }

    return pair


def read_file(data_path: pathlib.Path, task: str) -> list[dict[str, str]]:
    """Takes the path to a data directory and a task, reads the JSONL datafiles
    in the directory and returns a list of dictionaries containing all the
    information used by the evaluation.

    Args:
        data_path(pathlib.Path): The path to a data directory containing JSONL
            files.
        task(str): The task of the data (for example blimp).

    Returns:
        list[dict[str, str]]: A list of dictionaries containing the information
            to evaluate the given task.
    """
    pairs = []
    for filename in data_path.iterdir():
        if filename.suffix != ".jsonl":
            continue

        with filename.open("r") as data_file:
            for line in data_file:
                pairs.append(decode(line, filename, task))

    return pairs
