from __future__ import annotations

import torch
from torch.nn import functional as F
from tqdm import tqdm
from typing import TYPE_CHECKING
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import copy

if TYPE_CHECKING:
    from argparse import Namespace
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


class Trainer():

    def __init__(
        self: Trainer,
        model: nn.Module,
        train_dataloader: DataLoader,
        args: Namespace,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: LRScheduler | None = None,
        ema_model: nn.Module | None = None,
        valid_dataloader: DataLoader | None = None,
        predict_dataloader: DataLoader | None = None
    ) -> None:
        """The Trainer class handles all the fine tuning,
        evaluation, and prediction of a given task for a
        given model.

        Args:
            model(nn.Module): The model to finetune.
            train_dataloader(DataLoader): The dataloader
                of the train datasets.
            args(Namespace): The config information such
                as hyperparameters, directories, verbose,
                etc.
            optimizer(Optimizer): The optimizer used
                during training.
            device(torch.device): The device to use for
                finetuning.
            scheduler(LRScheduler | None): The learning
                rate scheduler to use during finetuning.
            ema_model(nn.Module | None): The exponential
                moving average model, if used.
            valid_dataloader(DataLoader | None): The
                dataloader of the the dataset to validate
                on, without it, no validation will be done.
            predict_dataloader(DataLoader | None): The
                dataloader of the dataset to predict on,
                without it, no prediction will be done.
        """
        self.model: DataLoader = model
        self.train_dataloader: DataLoader = train_dataloader
        self.args: Namespace = args
        self.optimizer: Optimizer = optimizer
        self.device: torch.device = device
        self.scheduler: LRScheduler = scheduler
        self.ema_model: nn.Module = ema_model
        self.valid_dataloader: DataLoader = valid_dataloader
        self.predict_dataloader: DataLoader = predict_dataloader

    def train_epoch(self: Trainer, total_steps: int, global_step: int = 0) -> int:
        """This function does a single epoch of the training.

        Args:
            total_steps(int): The total number of finetuning
                steps to do. Used for the progress bar and in
                case the finetuning needs to be stoped mid
                epoch.
            global_step(int): The current step the finetuning
                is on.

        Returns:
            int: The current step the model is on at the end of
                the epoch.
        """
        self.model.train()

        progress_bar = tqdm(initial=global_step, total=total_steps)

        for input_data, attention_mask, labels in self.train_dataloader:
            input_data = input_data.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            labels = labels.to(device=self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_data, attention_mask)

            loss = F.cross_entropy(logits, labels)
            loss.backward()

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.ema_model is not None:
                with torch.no_grad():
                    for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
                        param_k.data.mul_(self.args.ema_decay).add_((1.0 - self.args.ema_decay) * param_q.detach().data)

            metrics = self.calculate_metrics(logits, labels, self.args.metrics)

            metrics_string = ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

            progress_bar.update()

            if self.args.verbose:
                progress_bar.set_postfix_str(metrics_string)

            global_step += 1

        progress_bar.close()

        return global_step

    @torch.no_grad()
    def evaluate(self: Trainer) -> dict[str, float]:
        """This function does an evaluation pass on the
        validation dataset.

        Returns:
            dict[str, float]: A dictionary of scores of the
                model on the validation dataset, based on
                the metrics to evaluate on.
        """
        assert self.valid_dataloader is not None, "No valid dataset to run evaluation on!"

        if self.ema_model is not None:
            model = self.ema_model
        else:
            model = self.model
        model.eval()

        progress_bar = tqdm(total=len(self.valid_dataloader))

        labels = []
        logits = []

        for input_data, attention_mask, label in self.valid_dataloader:
            input_data = input_data.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            label = label.to(device=self.device)

            logit = model(input_data, attention_mask)

            logits.append(logit)
            labels.append(label)

            progress_bar.update()

        labels = torch.cat(labels, dim=0)
        logits = torch.cat(logits, dim=0)

        metrics = self.calculate_metrics(logits, labels, self.args.metrics)

        progress_bar.close()

        if self.args.verbose:
            metrics_string = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
            print(metrics_string)

        return metrics

    def save_model(self: Trainer, model: nn.Module) -> None:
        """This function saves the passed model to a file. The
        directory is specified inside the arguments passed to
        the constructor of the class.

        Args:
            model(nn.Module): The model to save.
        """
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), self.args.save_path / "model.pt")

    def _compare_scores(self: Trainer, best: float, current: float, bigger_better: bool) -> bool:
        if best is None:
            return True
        else:
            if current > best and bigger_better:
                return True
            elif current < best and not bigger_better:
                return True
            return False

    @staticmethod
    def calculate_metrics(logits: torch.Tensor, labels: torch.Tensor, metrics_to_calculate: list[str]) -> dict[str, float]:
        """This function calculates the metrics specified by
        the user. This is a static method and can be used
        without initializing a Trainer.

        Args:
            logits(torch.Tensor): A tensor of logits per class
                calculated by a model.
            labels(torch.Tensor): A tensor of correct labels
                for each element of the batch
            metrics_to_calculate(list[str]): A list of metrics
                to evaluate.

        Returns:
        dict[str, float]: a dictionary containing the scores of
            the model on the specified metrics.

        Shapes:
            - logits: :math:`(B, N)`
            - labels: :math:`(B)`, where each element is in
                :math:`[0, N-1]`
        """
        predictions = logits.argmax(dim=-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        metrics = dict()

        for metric in metrics_to_calculate:
            if metric == "f1":
                metrics["f1"] = f1_score(labels, predictions)
            elif metric == "accuracy":
                metrics["accuracy"] = accuracy_score(labels, predictions)
            elif metric == "mcc":
                metrics["mcc"] = matthews_corrcoef(labels, predictions)
            else:
                print(f"Metric {metric} is unknown / not implemented. It will be skipped!")

        return metrics

    def train(self: Trainer) -> None:
        """This function does the training based on the
        hyperparameters, model, optimizer, scheduler specified
        to the constructor of the class.
        """
        total_steps: int = self.args.num_epochs * len(self.train_dataloader)
        step: int = 0
        best_score: float | None = None
        self.best_model: nn.Module | None = None
        update_best: bool = False

        for epoch in range(self.args.num_epochs):
            step = self.train_epoch(total_steps, step)

            if self.valid_dataloader is not None:
                metrics: dict[str, float] = self.evaluate()
                if self.args.keep_best_model:
                    score: float = metrics[self.args.metric_for_valid]
                    if self._compare_scores(best_score, score, self.args.higher_is_better):
                        if self.ema_model is not None:
                            self.best_model = copy.deepcopy(self.ema_model)
                        else:
                            self.best_model = copy.deepcopy(self.model)
                        best_score = score
                        update_best = True

            if self.args.save:
                if self.args.keep_best_model and update_best:
                    self.save_model(self.best_model)
                    update_best = False
                elif self.ema_model is not None:
                    self.save_model(self.ema_model)
                else:
                    self.save_model(self.model)

    @torch.no_grad()
    def predict_classification(self: Trainer) -> torch.Tensor:
        """This function creates predictions for the prediction
        dataset.

        Returns:
            dict[str, float]: A dictionary of scores of the
                model on the validation dataset, based on
                the metrics to evaluate on.
        """
        assert self.predict_dataloader is not None, "No predict dataset to predict on!"

        if hasattr(self, "best_model"):
            model: nn.Module = self.best_model
        elif self.ema_model is not None:
            model = self.ema_model
        else:
            model = self.model
        model.eval()

        progress_bar = tqdm(total=len(self.predict_dataloader))

        logits = []

        for input_data, attention_mask in self.predict_dataloader:
            input_data = input_data.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)

            logit = model(input_data, attention_mask)

            logits.append(logit)

            progress_bar.update()

        logits = torch.cat(logits, dim=0)
        preds = logits.argmax(dim=-1)

        progress_bar.close()

        return preds
