import os, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import training.speaker_verification.model as models
from training.speaker_verification.criterion import ContrastiveLoss, SimpleContrastiveLoss, CosineLoss
import training.speaker_verification.common as U
import torch.optim as optim
import numpy as np

class VerificationTrainer:

    def __init__(self,
                 batch_size,
                 learning_rate,
                 num_speakers=1000,
                 resume=None,
                 pad='wrap'):

        self.num_speakers = num_speakers

        # padding mode
        self.pad_mode = pad

        self.clip = 0.25

        cpd = None
        if resume:
            cpd = torch.load(resume, map_location=lambda storage, loc: storage)
            self.num_speakers = cpd["num_speakers"]

        self.model = models.IdentifyAndEmbed(self.num_speakers).cuda()
        self.batch_size = batch_size
        self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=5e-4,
                                    momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 20, 30])

        if resume:
            self.model.load_state_dict(cpd["model"])
            self.optimizer.load_state_dict(cpd["optimizer"])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

        # Loss that adaptively searches for the optimal margin
        # self.verification_criterion = ContrastiveLoss(num_search_steps=5, search_freq=100)
        self.verification_criterion = SimpleContrastiveLoss(margin=1.0)


    def checkpoint(self, path):
        cpd = {
            "num_speakers": self.num_speakers,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(cpd, path)

    def train_epoch_classification(self, data_loader):
        for idx, (data_batch, label_batch) in enumerate(data_loader):
            seq_batch, seq_lens = U.process_data_batch(data_batch, mode=self.pad_mode)
            self.optimizer.zero_grad()
            preds, _ = self.model([seq_batch, seq_lens])
            loss = F.nll_loss(preds, label_batch.cuda())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            correct = (torch.argmax(preds, dim=1) == label_batch.cuda()).sum()
            yield loss.item(), correct.item()
        self.lr_scheduler.step()

    def _verification_objective(self, embeddings1, embeddings2, labels):
        return self.verification_criterion(embeddings1, embeddings2, labels)

    def train_epoch_verification(self, data_loader, alpha=1.0):
        for idx, (data_batch1, data_batch2, label_batch1, label_batch2) in enumerate(data_loader):
            seq_batch1, seq_lens1 = U.process_data_batch(data_batch1, mode=self.pad_mode)
            seq_batch2, seq_lens2 = U.process_data_batch(data_batch2, mode=self.pad_mode)
            self.optimizer.zero_grad()
            preds1, em1 = self.model([seq_batch1, seq_lens1])
            preds2, em2 = self.model([seq_batch2, seq_lens2])
            embed_labels = (label_batch1 == label_batch2).float().cuda()
            verification_loss = self._verification_objective(preds1, preds2, embed_labels)
            classification_loss = F.nll_loss(preds1, label_batch1.cuda()) + F.nll_loss(preds2, label_batch2.cuda())
            loss = (1 - alpha) * classification_loss + alpha * verification_loss
            loss.backward()
            self.optimizer.step()
            # best_margin = self.verification_criterion.margin_searcher.best_margin
            yield verification_loss, classification_loss

        # Reset the search at each epoch
        # self.verification_criterion.reset()

    def validation(self, data_loader):
        num_correct = 0
        for idx, (data_batch, label_batch) in enumerate(data_loader):
            seq_batch, seq_lens = U.process_data_batch(data_batch, mode=self.pad_mode)
            preds, _ = self.model([seq_batch, seq_lens])
            correct = (torch.argmax(preds, dim=1) == label_batch.cuda()).sum()
            num_correct += correct.item()

        return 1.0 - (num_correct / float(len(data_loader.dataset)))

    def compute_verification_eer(self, validator):
        return validator.evaluate(self.model)
