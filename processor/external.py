from processor.speaker_embedding_processor import SpeakerEmbeddingProcessor
from processor.audio_processor import AudioProcessor
import torch.utils.data
import numpy as np
import os
import gin
import tqdm

@gin.configurable
def load_voxceleb_embeddings(voxceleb_wav_path, n=1024):
    paths = []
    for root, _, filenames in os.walk(voxceleb_wav_path):
        for filename in filenames:
            processed_file_path = os.path.join(root, filename)
            paths.append(processed_file_path)

    idxs = np.arange(len(paths))
    np.random.shuffle(idxs)
    paths = [paths[i] for i in idxs[:n]]
    return embeddings_from_wav_set(paths)

def collate(batch):
    return [x[0] for x in batch]


def embeddings_from_wav_set(wav_file_paths):
    batch_size = 64

    get_embeddings = SpeakerEmbeddingProcessor(use_gpu=True)
    audio_processor = AudioProcessor()

    dset = WavInferenceDataSet(wav_file_paths, audio_processor)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=16)
    embeddings = []

    pbar = tqdm.tqdm(total=len(wav_file_paths))
    for idx, utterance_batch in enumerate(loader):
        embeddings_batch = get_embeddings(utterance_batch)
        embeddings.append(embeddings_batch.numpy())
        pbar.update(batch_size)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


class WavInferenceDataSet(torch.utils.data.Dataset):

    def __init__(self, file_paths, audio_processor):
        super(WavInferenceDataSet, self).__init__()
        self.audio_processor = audio_processor
        self.utterance_paths = file_paths

    def __len__(self):
        return len(self.utterance_paths)

    def __getitem__(self, item):
        return self.audio_processor.from_file(self.utterance_paths[item])
