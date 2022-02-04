from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import hamming_loss
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BatchEncoding, BertConfig, BertModel
from transformers.optimization import get_linear_schedule_with_warmup

from constants import GPU_AVAILABLE
from networks.utils import plot_metric
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


class GenreEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        """
        Genre encoderEncoder that uses BERT to encode a list of genres into a
        dense space.

        Args:
             config: Huggingface Bert Model config.
        """
        super(GenreEncoder, self).__init__()
        self.encoder = BertModel(config)

    def forward(self, x: Dict[str, torch.Tensor]):
        """
        Encoding method.

        Args:
            x: Tokenized genre input
        """
        return self.encoder(**x).pooler_output


class GenreAutoencoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        """
        A model which can be seen as an autoencoder (AE) for the genre strings.
        Takes a Bert-like tokenized concatenated string of genre strings as
        input and aims to predict the input genres from the pooled output of a
        small Bert-like model.

        Args:
            encoder: Genre encoder
            decoder: Genre decoder
        """
        super(GenreAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loss = nn.BCELoss()

        self.metrics = {
            "train_loss": {
                "file_name": "train_loss.png",
                "label": "Train Loss",
                "values": [],
            },
            "val_loss": {
                "file_name": "val_loss.png",
                "label": "Val. Loss",
                "values": [],
            },
            "val_accuracy": {
                "file_name": "val_accuracy.png",
                "label": "Val. Accuracy",
                "values": [],
            },
            "val_em_ratio": {
                "file_name": "val_em_ratio.png",
                "label": "Val. Exact Match Ratio",
                "values": [],
            },
            "val_hamming_loss": {
                "file_name": "val_hamming_loss.png",
                "label": "Val. Hamming Loss",
                "values": [],
            },
            "samples_seen": 0,
        }

    def forward(self, x: BatchEncoding) -> torch.Tensor:
        """
        Runs a forward pass for a concatenated list of genre strings.

        Args:
            x: Bert-like batch encooding

        Returns:
            Genre embedding
        """
        encoding = self.encoder(x)
        return self.decoder(encoding)

    def configure_optimizers(self):
        """
        Assign both the discriminator and the generator optimizer.
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
            params=optimizer_grouped_parameters, lr=1e-6, betas=(0.0, 0.99)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.max_steps * 0.1),
            num_training_steps=self.trainer.max_steps,
        )
        return [optimizer], [scheduler]

    def train2(
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
        path = kwargs.get("path", ".")
        print_output_every_n = kwargs.get("print_output_every_n", steps // 100)
        for step in range(steps):
            batch, labels = data_loader.train_generator[step]
            self.metrics["train_loss"]["values"].append(
                self.train_on_batch(batch, labels)
            )
            self.samples_seen += len(batch)
            if step % print_output_every_n == 0:
                self.validate(data_loader)
                self._print_output(n=print_output_every_n)
                for _k, v in self.metrics.items():
                    plot_metric(
                        path,
                        steps=self.samples_seen,
                        metric=v.get("values"),
                        y_label=v.get("label"),
                        file_name=v.get("file_name"),
                    )

    def training_step(self, batch: BatchEncoding):
        """
        Trains the AE on the given batch.

        Args:
            batch: Tokenized genre sequence

        Returns:
            Scalar loss of this batch
        """
        x, labels = batch
        x = self(x)
        return self.loss(x, labels)

    def _print_output(self, n):
        steps = (len(self.metrics["train_loss"]["values"]) * n) - n
        train_loss = np.round(
            np.mean(self.metrics["train_loss"]["values"][-n:]), decimals=3
        )
        val_loss = np.round(self.metrics["val_loss"]["values"][-1], decimals=3)
        val_accuracy = np.round(
            self.metrics["val_accuracy"]["values"][-1], decimals=3
        )
        val_em_ratio = np.round(
            self.metrics["val_em_ratio"]["values"][-1], decimals=3
        )
        val_hamming_loss = np.round(
            self.metrics["val_hamming_loss"]["values"][-1],
            decimals=3,
        )
        logger.info(
            f"Steps: {steps}"
            + f" Train Loss: {train_loss} -"
            + f" Val Loss: {val_loss} -"
            + f" Val Acc.: {val_accuracy} -"
            + f" Val EM Ratio: {val_em_ratio} -"
            + f" Val Hamming Loss: {val_hamming_loss}"
        )

    def validate(self, data_loader):
        self.encoder.eval()
        self.decoder.eval()
        data, labels = data_loader.return_dataset("val_set")
        input_ids = torch.split(data["input_ids"], 512)
        attention_mask = torch.split(data["attention_mask"], 512)
        token_type_ids = torch.split(data["token_type_ids"], 512)
        prediction = []
        for ii, am, tti in tqdm(
            zip(input_ids, attention_mask, token_type_ids),
            total=len(input_ids),
        ):
            batch = {
                "input_ids": ii,
                "attention_mask": am,
                "token_type_ids": tti,
            }
            if GPU_AVAILABLE:
                batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                prediction.append(self(batch))
        prediction = torch.cat(prediction).cpu()
        loss = nn.BCELoss()(prediction, labels)
        self.metrics["val_loss"]["values"].append(loss.numpy().tolist())
        correct = labels == prediction.round()
        self.metrics["val_accuracy"]["values"].append(
            np.mean(correct.numpy() != 0).tolist()
        )
        self.metrics["val_em_ratio"]["values"].append(
            np.mean(correct.numpy().all(1)).tolist()
        )
        self.metrics["val_hamming_loss"]["values"].append(
            hamming_loss(labels, prediction.round())
        )
