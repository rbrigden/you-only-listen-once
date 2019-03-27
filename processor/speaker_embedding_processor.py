import torch
import numpy as np
import inference
import training.speaker_verification.model as models


class SpeakerEmbeddingProcessor:

    def __init__(self, model_cls, checkpoint_path):
        self.embedding_model = model_cls()
        self.inference_engine = SpeakerEmbeddingInference(self.embedding_model)
        self.inference_engine.load_params(checkpoint_path)

    def forward(self, spect_batch):
        return self.inference_engine.forward(spect_batch)

    def __call__(self, spect_batch):
        return self.forward(spect_batch)



class SpeakerEmbeddingInference:

    def __init__(self, model):
        self.model = model

    def _process_data_batch(self, data_batch):
        # pad the sequences, each seq must be (L, *)
        seq_lens = [len(x) for x in data_batch]
        seq_batch = pad_sequence(data_batch, batch_first=True)
        return seq_batch.unsqueeze(1).cuda(), seq_lens

    def forward(self, x):
        self.model.eval()
        seq_batch, seq_lens = self._process_data_batch(x)
        with torch.no_grad():
            embeddings = self.model([seq_batch, seq_lens], em=True)
        return embeddings

    def load_params(self, checkpoint_path):
        cpd = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(cpd["model"])
        self.model.load_state_dict(cpd["model"])

