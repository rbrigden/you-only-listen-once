import numpy as np
from scipy.io import wavfile as wav
import torch

class PresenceScore:

    def __init__(self):
        pass

    def _build_trellis(self, log_probs, trellis_order):
        trellis = []
        for idx in trellis_order:
            trellis.append(log_probs[:, idx].clone())
        trellis = torch.stack(trellis)
        return trellis.permute(1, 0)

    def _sum_paths(self, trellis):
        num_frames, num_states = trellis.size()

        alpha_0 = trellis[0].clone()
        for t in range(1, num_frames):
            alpha_1 = trellis[t].clone()
            for s in range(1, num_states+1):
                alpha_1[s-1] = torch.logsumexp(alpha_0[max(s-2, 0):s], dim=0) + alpha_1[s-1]
            alpha_0 = alpha_1
        return alpha_0[-1]

    def _sum_paths_blank(self, trellis):
        num_frames, num_states = trellis.size()

        alpha_0 = trellis[0].clone()
        for t in range(1, num_frames):
            alpha_1 = trellis[t].clone()
            for s in range(0, num_states):
                if s % 2 == 0:
                    fan_in = alpha_0[max(s-1, 0):s+1]
                else:
                    fan_in = alpha_0[max(s-2, 1):s+1]
                alpha_1[s] = torch.logsumexp(fan_in, dim=0) + alpha_1[s]
            alpha_0 = alpha_1
        return torch.logsumexp(alpha_0[-2:], dim=0)

    def forward(self, target, log_probs, chars):
        trellis_order_chars = ["-"] + [target[i // 2] if i % 2 == 0 else "-" for i in range(len(target * 2))]
        trellis_order = [chars.index(c) for c in trellis_order_chars]
        trellis = self._build_trellis(log_probs, trellis_order)
        return self._sum_paths_blank(trellis)






if __name__ == '__main__':
    from presence_detection.speech_rec import SpeechRec
    sr = SpeechRec("/home/rbrigden/deepspeech-models/output_graph.pb","/home/rbrigden/deepspeech-models/alphabet.txt")

    fs, audio = wav.read("/home/rbrigden/demo.wav")

    seqs = [
        "maybe this moment works better cause some stuff on the page doesnt work as well as it does in real life",
        "maybe this moment works better cause some stuff on the page doesnt work as well as it does in real wild",
        "maybe this moment works better cause some stuff on the page doesnt work as well as it does in bad wild",
        "maybe this moment works worse cause some stuff on the page doesnt work as well as it does in bad wild",
        "hello this moment works worse cause some stuff on the page doesnt work as well as it does in bad wild",
        "hello this moment works worse cause some stuff on the book doesnt work as well as it does in bad wild",
        "hello this moment works worse goodness some stuff on the book doesnt work as well as it does in bad wild",
        "hello this blocked works worse goodness some stuff on the book doesnt work as well as it does in bad wild",
        "hello this blocked works worse glam some stuff on the book doesnt work as well as it does in bad wild",
        "hello this blocked fairs worse glam some stuff on the book doesnt work as well as it does in bad wild",
        "hello its blocked fairs worse glam some stuff on the book doesnt work as well as it does in bad wild",
        "hello its blocked fairs worse glam some stuff on the grain doesnt work as well as it does in bad wild",
        "hello its blocked fairs worse glam some stuff on the grain doesnt work as poorly as it does in bad wild",
        "hello its blocked fairs worse glam some food on the grain doesnt work as poorly as it does in bad wild",
        "hello its blocked fairs worse glam some food on the grain doesnt pond as poorly as it does in bad wild",
        "hello its blocked fairs worse glam some food off the grain doesnt pond as poorly as it does in bad wild",
        "hello its blocked fairs worse glam little food off the grain doesnt pond as poorly as it does in bad wild",
        "hello its blocked fairs worse glam little food off a grain doesnt pond as poorly as it does in bad wild",
        "hello its blocked fairs worse glam little food off a grain doesnt pond list poorly as it does in bad wild",
        "hello its blocked fairs worse glam little food off a grain doesnt pond list poorly as it came in bad wild",
        "hello its blocked fairs worse glam little food off a grain doesnt pond list poorly as she came in bad wild",
        "hello its blocked fairs worse glam little food off a grain doesnt pond list poorly as she came can bad wild",
        "hello its blocked fairs worse glam",

    ]



    log_probs = sr.forward(audio, fs, display_chars=True)
    chars = sr.alphabet._label_to_str + ["-"]
    pscore = PresenceScore()
    for i, seq in enumerate(seqs):
        print(seq)
        print(pscore.forward(seq, log_probs, chars))

