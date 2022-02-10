from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import hamming_loss, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from transformers import BatchEncoding
from transformers.optimization import get_linear_schedule_with_warmup


class PosLoss(nn.Module):
    def forward(self, pred, true) -> torch.Tensor:
        """
        Calculates the loss.

        Args:
            pred: Predicted Logits
            true: True values
        """
        pos_loss = true * torch.log(pred)
        return -1 * torch.masked_select(pos_loss, true == 1).mean()


class NegLoss(nn.Module):
    def forward(self, pred, true) -> torch.Tensor:
        """
        Calculates the loss.

        Args:
            pred: Predicted Logits
            true: True values
        """
        neg_loss = (1 - true) * -torch.log(1 - pred)
        return torch.masked_select(neg_loss, true == 0).mean()


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
        self.classifier_head = nn.Linear(input_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction head that takes the genre embeddings as input and predicts
        the genre outputs.

        Args:
            x: Genre embeddings

        Returns:
            Predicted genres
        """
        x = self.classifier_head(x)
        return torch.sigmoid(x)


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

        self.pos_loss = PosLoss()
        self.neg_loss = NegLoss()

    def forward(
        self,
        x: BatchEncoding = None,
        masked_x: BatchEncoding = None,
        masked_labels: BatchEncoding = None,
    ) -> Tuple:
        """
        Runs a forward pass for a concatenated list of genre strings.

        Args:
            x: Bert-like batch encoding

        Returns:
            Genre embedding
        """
        decoding = None
        mlm_outputs = None
        if x:
            encoding = self.encoder.bert(**x).pooler_output
            decoding = self.decoder(encoding)
        if masked_x is not None and masked_labels is not None:
            mlm_outputs = self.encoder(**masked_x, labels=masked_labels)
        return decoding, mlm_outputs

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
            params=optimizer_grouped_parameters, lr=1e-6, betas=(0, 0.99)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.max_steps * 0.1),
            num_training_steps=self.trainer.max_steps,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch: BatchEncoding):
        """
        Trains the AE on the given batch.

        Args:
            batch: Tokenized genre sequence

        Returns:
            Scalar loss of this batch
        """
        sch = self.lr_schedulers()
        sch.step()
        x, masked_x, masked_labels, labels = batch
        decoding, mlm_output = self(x, masked_x, masked_labels)
        pos_loss = self.pos_loss(decoding, labels)
        neg_loss = self.neg_loss(decoding, labels)
        mlm_loss = mlm_output.loss
        total_loss = pos_loss + neg_loss + mlm_loss
        self.log(
            "train/pos_loss",
            pos_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/neg_loss",
            neg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/mlm_loss",
            mlm_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return total_loss

    def validation_step(self, batch: Tuple, _batch_idx) -> Dict:
        """
        Takes a batch from the validation generator and predicts it.

        Args:
            batch: Batch. A Tuple of data and labels
            _batch_idx: Batch index
        """
        data, labels = batch
        output = self(data)
        return {"pred": output[0], "true": labels}

    def validation_epoch_end(self, x: List[Dict]):
        """
        Takes the predictions from the validation steps and calculates the
        validation metrics on it.

        Args:
            x: List of validation_step outputs
        """
        prediction = torch.cat([xi["pred"].cpu() for xi in x])
        true = torch.cat([xi["true"].cpu() for xi in x])
        pos_loss = self.pos_loss(prediction, true)
        neg_loss = self.neg_loss(prediction, true)
        class_prediction = prediction.round()
        correct = true == class_prediction
        accuracy = correct.numpy().mean()
        em_ratio = np.mean(correct.numpy().all(1)).tolist()
        h_loss = hamming_loss(true.cpu(), class_prediction)
        recall = recall_score(true, class_prediction, average="micro")
        precision = precision_score(true, class_prediction, average="micro")
        self.log_dict(
            {
                "val/pos_loss": pos_loss,
                "val/neg_loss": neg_loss,
                "val/accuracy": accuracy,
                "val/exact-match-ratio": em_ratio,
                "val/hamming-loss": h_loss,
                "val/recall": recall,
                "val/precision": precision,
            }
        )

    def on_fit_start(self):
        """
        This method gets executed before a Trainer trains this model.

        It tells the W&B logger to watch the model in order to check the
        gradients report the gradients if W&B is online.
        """
        if hasattr(self, "logger"):
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.watch(self)
