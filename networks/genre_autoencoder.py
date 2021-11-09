from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import BatchEncoding, BertConfig, BertModel

from constants import GPU_AVAILABLE
from utils import logger


class GenreDecoder(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        """
        Genre classification head. Takes the genre embedding as input and tries
        to classify the true genres. The.

        Args:
            input_dim: Latent size dimension
            num_labels: Total number of genres
        """
        super(GenreDecoder, self).__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction head that takes the genre embeddings as input and predicts
        the genre outputs.

        Args:
            x: Genre embeddings

        Returns:
            Predicted genres
        """
        x = self.classifier(x)
        return nn.Sigmoid()(x)


class GenreAutoencoder(nn.Module):
    def __init__(
        self,
        num_labels: int,
        encoding_dim: int,
        scheduler,
        optimizer_params: Dict,
        vocab_size: int,
    ):
        """
        A model which can be seen as an autoencoder (AE) for the genre strings.
        Takes a Bert-like tokenized concatenated string of genre strings as
        input and aims to predict the input genres from the pooled output of a
        small Bert-like model.

        Args:
            num_labels: Total number of genres
            encoding_dim: Latent size of the AE
            scheduler: Learning rate scheduler
            optimizer_params: Params for the Adam optimizer
            vocab_size: Total size of genre vocab.
        """
        super(GenreAutoencoder, self).__init__()
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=encoding_dim,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=64,
        )
        self.encoder = BertModel(bert_config)
        self.decoder = GenreDecoder(
            input_dim=encoding_dim, num_labels=num_labels
        )
        self.optimizer, self.scheduler = self.init_optimizer(
            scheduler, optimizer_params
        )
        self.metrics = {"train_loss": []}
        if GPU_AVAILABLE:
            self.encoder.cuda()
            self.decoder.cuda()

    def forward(self, x: BatchEncoding) -> torch.Tensor:
        """
        Runs a forward pass for a concatenated list of genre strings.

        Args:
            x: Bert-like batch encooding

        Returns:
            Genre embedding
        """
        encoding = self.encoder(**x).pooler_output
        return self.decoder(encoding)

    def init_optimizer(self, scheduler, optimizer_params):
        """
        Assign both the discriminator and the generator optimizer.

        Args:
            scheduler: Learning rate scheduler
            optimizer_params: PyTorch optimizer parameters
        """
        encoders = (self.encoder, self.decoder)
        optimizer_grouped_parameters = []
        for encoder in encoders:
            params = [
                {
                    "params": [
                        p for p in encoder.parameters() if p.requires_grad
                    ],
                    "weight_decay": 0,
                },
            ]
            optimizer_grouped_parameters.extend(params)
        optimizer = AdamW(
            params=optimizer_grouped_parameters, **optimizer_params
        )
        scheduler = scheduler(optimizer)
        return optimizer, scheduler

    def train(
        self,
        data_loader,
        steps: int,
        batch_size: int,
        **kwargs,
    ):
        """
        Trains the Genre Autoencoder.

        Args:
            data_loader: Data Loader used for training
            steps: Absolute number of training steps
            batch_size: Batch size for training
            **kwargs: Keyword arguments. See below.

        Keyword Args:
            path: Path to which model training graphs will be written
            write_model_to: Path that can be passed to write the model to during
                training
            grad_acc_steps: Gradient accumulation steps. Ideally a factor of the
                batch size. Otherwise not the entire batch will be used for
                training
        """
        kwargs.get("path", ".")
        kwargs.get("write_model_to", None)
        print_output_every_n = kwargs.get("print_output_every_n", steps // 100)
        for step in range(steps):
            batch, labels = data_loader.__getitem__(step)
            self.metrics["train_loss"].append(
                self.train_on_batch(batch, labels)
            )
            if step % print_output_every_n == 0:
                self._print_output(n=print_output_every_n)
            #     for _k, v in self.metrics.items():
            #         plot_metric(
            #             path,
            #             steps=self.images_shown,
            #             metric=v.get("values"),
            #             y_label=v.get("label"),
            #             file_name=v.get("file_name"),
            #         )
            #     if model_dump_path:
            #         self.save(model_dump_path)

    def train_on_batch(self, batch: BatchEncoding, labels: torch.Tensor) -> np.ndarray:
        """
        Trains the AE on the given batch.

        Args:
            batch: Tokenized genre sequence
            labels: True genres

        Returns:
            Scalar loss of this batch
        """
        if GPU_AVAILABLE:
            batch = {k: v.cuda() for k, v in batch.items()}
            labels = labels.cuda()
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        x = self(batch)
        loss = nn.BCELoss()(x, labels)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.scheduler.step()
        return loss.cpu().detach().numpy()

    def _print_output(self, n):
        steps = (len(self.metrics["train_loss"]) * n) - n
        train_loss = np.round(
            np.mean(self.metrics["train_loss"][-n:]), decimals=3
        )
        # val_loss = np.round(
        #     np.mean(self.metrics["val_loss"][-n:]), decimals=3
        # )
        logger.info(
            f"Steps: {steps}"
            + f" Train Loss: {train_loss} -"
            # + f" Discriminator Loss: {d_loss}"
        )
