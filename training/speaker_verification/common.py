import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def process_data_batch(data_batch, mode='zeros'):
    # pad the sequences, each seq must be (L, *)
    seq_lens = [len(x) for x in data_batch]
    max_len = max(seq_lens)
    pad_widths = [max_len - l for l in seq_lens]

    if mode == 'zeros':
        seq_batch = pad_sequence(data_batch, batch_first=True)
    elif mode == 'wrap':
        seq_batch = torch.stack(
            [torch.FloatTensor(np.pad(x.numpy(), pad_width=((0, w), (0, 0)), mode='wrap')) for w, x in
             zip(pad_widths, data_batch)])
    else:
        raise ValueError("Invalid mode specified")
    return seq_batch.unsqueeze(1).cuda(), seq_lens
