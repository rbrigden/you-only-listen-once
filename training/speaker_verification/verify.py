import torch
import numpy as np
import torch.nn.functional as F
import data.voxceleb.voxceleb as voxceleb
import training.speaker_verification.eer as eer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import training.speaker_verification.common as U
import training.speaker_verification.model as models
import argparse


def euclidean_similarity(e1, e2):
    return - torch.dist(e1, e2, p=2)

def cosine_similarity(e1, e2):
    return F.cosine_similarity(e1, e2)


class VerificationEvaluator:

    def __init__(self, processed_test_root, pad='wrap'):
        self.processed_test_root = processed_test_root
        # TODO: Make this work for NIST-SRE
        self.enrol_set, self.test_set, self.labels = self._prepare_voxceleb()
        self.similarity_fn = cosine_similarity
        self.pad_mode = pad

    def _prepare_voxceleb(self):
        veri_file_path = "data/voxceleb/veri_test.txt"
        return voxceleb.VoxcelebVerification.build(veri_file_path, self.processed_test_root)

    def evaluate(self, model, num_workers=8):
        batch_size = 16
        enrol_loader = DataLoader(self.enrol_set, shuffle=False, num_workers=num_workers//2, batch_size=batch_size,
                                  collate_fn=voxceleb.voxceleb_veri_collate)
        test_loader = DataLoader(self.test_set, shuffle=False, num_workers=num_workers//2, batch_size=batch_size,
                                 collate_fn=voxceleb.voxceleb_veri_collate)

        embedding_size = model.embedding_size
        enrol_embeddings = torch.zeros((len(self.enrol_set), embedding_size))
        test_embeddings = torch.zeros((len(self.test_set), embedding_size))

        for idx, (enrol_batch,) in enumerate(enrol_loader):
            enrol_batch, enrol_seq_lens = U.process_data_batch(enrol_batch, mode=self.pad_mode)

            bsize = enrol_batch.size()[0]
            bidx = idx * bsize
            with torch.no_grad():
                enrol_embeddings[bidx:bidx + bsize, :] = model.forward([enrol_batch, enrol_seq_lens], em=True).cpu()

        for idx, (test_batch,) in enumerate(test_loader):
            test_batch, test_seq_lens = U.process_data_batch(test_batch, mode=self.pad_mode)

            bsize = test_batch.size()[0]
            bidx = idx * bsize
            with torch.no_grad():
                test_embeddings[bidx:bidx + bsize, :] = model.forward([test_batch, test_seq_lens], em=True).cpu()

        scores = []
        for enrol_idx, test_idx in zip(self.enrol_set.sample_idxs, self.test_set.sample_idxs):
            score = self.similarity_fn(enrol_embeddings[enrol_idx].unsqueeze(0), test_embeddings[test_idx].unsqueeze(0))
            scores.append(score.item())

        return eer.EER(self.labels, scores)[0]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb-test-path', type=str, default="/home/rbrigden/voxceleb/test/processed1")
    parser.add_argument('--model-path', type=str, default="/home/rbrigden/capstone/models/verification/base_13eer.pt")
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)


    args = parser.parse_args()
    with torch.cuda.device(args.device):
        cpd = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        num_speakers = cpd["num_speakers"]

        model = models.IdentifyAndEmbed(num_speakers).cuda()
        model.load_state_dict(cpd["model"])
        model.eval()
        evaluator = VerificationEvaluator(args.voxceleb_test_path)
        eer = evaluator.evaluate(model)
        print(eer)