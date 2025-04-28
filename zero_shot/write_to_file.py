from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from _io import TextIOWrapper


def create_evaluation_report(temperature: float, temperature_pos: int, avg_accuracy: torch.Tensor, accuracies: dict[str, list[dict[str, float]]], file: TextIOWrapper | None = None) -> None:
    """This function creates a report and either saves it to a file or prints it to the terminal.

    Args:
        temperature(float): The temperature at which the model is evaluated.
        temperature_pos(int): The position of the evaluated temperature.
        avg_accuracy(torch.Tensor): The average accuracy of the model at the given temperature.
        avg_accuracy(dict[str, list[dict[str, float]]]): The finegrained accuracies of the model
            at the given temperature.
        file(TextIOWrapper | None): The file to write to results to. (If None, it will printed
            printed to the terminal)
    """
    print(f"TEMPERATURE: {temperature:.2f}", file=file)
    print(file=file)

    for domain, accuracy in accuracies.items():
        print(f"### {domain.upper()} ACCURACY", file=file)
        for subdomain, acc in accuracy[temperature_pos].items():
            print(f"{subdomain}: {acc:.2f}", file=file)
        print(file=file)

    print("### AVERAGE ACCURACY", file=file)
    print(f"{avg_accuracy:.2f}", file=file)
    print(file=file)
