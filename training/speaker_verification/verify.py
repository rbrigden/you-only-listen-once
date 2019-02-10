import torch
import numpy as np
import torch.nn.functional as F
import data.voxceleb.voxceleb as voxceleb
import training.speaker_verification.eer as eer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class VerificationEvaluator:

    def __init__(self, processed_test_root):
        self.processed_test_root = processed_test_root
        # TODO: Make this work for NIST-SRE
        self.enrol_set, self.test_set, self.labels = self._prepare_voxceleb()

    def _prepare_voxceleb(self):
        veri_file_path = "data/voxceleb/veri_test.txt"
        return voxceleb.VoxcelebVerification.build(veri_file_path, self.processed_test_root)

    def _process_data_batch(self, data_batch):
        # pad the sequences, each seq must be (L, *)
        seq_lens = [len(x) for x in data_batch]
        seq_batch = pad_sequence(data_batch, batch_first=True)
        return seq_batch.unsqueeze(1).cuda(), seq_lens

    def evaluate(self, model):
        batch_size = 20
        enrol_loader = DataLoader(self.enrol_set, shuffle=False, num_workers=8, batch_size=batch_size,
                                  collate_fn=voxceleb.voxceleb_veri_collate)
        test_loader = DataLoader(self.test_set, shuffle=False, num_workers=8, batch_size=batch_size,
                                 collate_fn=voxceleb.voxceleb_veri_collate)

        embedding_size = model.embedding_size
        enrol_embeddings = torch.zeros((len(self.enrol_set), embedding_size)).cuda()
        test_embeddings = torch.zeros((len(self.test_set), embedding_size)).cuda()

        for idx, (enrol_batch,) in enumerate(enrol_loader):
            enrol_batch, enrol_seq_lens = self._process_data_batch(enrol_batch)

            bsize = enrol_batch.size()[0]
            bidx = idx * bsize
            with torch.no_grad():
                enrol_embeddings[bidx:bidx + bsize, :] = model.forward([enrol_batch, enrol_seq_lens], em=True)

        for idx, (test_batch,) in enumerate(test_loader):
            test_batch, test_seq_lens = self._process_data_batch(test_batch)

            bsize = test_batch.size()[0]
            bidx = idx * bsize
            with torch.no_grad():
                test_embeddings[bidx:bidx + bsize, :] = model.forward([test_batch, test_seq_lens], em=True)

        scores = []
        for enrol_idx, test_idx in zip(self.enrol_set.sample_idxs, self.test_set.sample_idxs):
            score = F.cosine_similarity(enrol_embeddings[enrol_idx].unsqueeze(0), test_embeddings[test_idx].unsqueeze(0), dim=1)
            scores.append(score.item())

        return eer.EER(self.labels, scores)[0]
