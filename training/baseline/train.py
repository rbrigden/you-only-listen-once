import torch
import numpy as np
import ray
import hw2.model as model
import torch.optim as optim
import torch.nn as nn
import os
import shutil
from torch.utils.data.dataset import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader
from ray.tune import Trainable, Experiment
from ray.tune.util import get_pinned_object, pin_in_object_store
import ray.tune as tune
import torch.nn.functional as F
from hw2.util.utils import EER
import hw2.losses as losses
from hw2.process import ContrastiveBatchSampler

FEATS = 64

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def init_cpd(lr, nspeakers):
    sc = model.SpeakerClassifier2d(nspeakers)
    optimizer = optim.Adam(lr)

    cpd = {'nspeakers': nspeakers,
           'lr': lr,
           'state_dict': sc.state_dict(),
           'optimizer': optimizer.state_dict()
           }
    return cpd

def classification_loss(pred, labels):
    criterion = nn.NLLLoss()
    return criterion(pred, labels)

def train(sen, optimizer, train_set, nspeakers, batch_size, alpha=0.5):
    criterion = nn.CosineEmbeddingLoss()
    cuda = torch.cuda.is_available()
    print("CUDA",cuda)

    sampler = ContrastiveBatchSampler(train_set.labels, nspeakers, batch_size)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=20, pin_memory=True)

    epoch_loss = 0
    for utterance_batch1, utterance_batch2, label_batch1, label_batch2 in loader:
        optimizer.zero_grad()
        if cuda:
            utterance_batch1, utterance_batch2, label_batch1, label_batch2 = utterance_batch1.cuda(), utterance_batch2.cuda(), label_batch1.cuda(), label_batch2.cuda()
        out1, out2 = sen(utterance_batch1, utterance_batch2)
        pred1, embed1 = out1
        pred2, embed2 = out2
        embed_labels = 2 * (label_batch1 == label_batch2).type(torch.cuda.FloatTensor) - 1
        embed_loss = criterion(embed1, embed2, embed_labels)
        closs = classification_loss(pred1, label_batch1) + classification_loss(pred2, label_batch2)
        loss = alpha * closs + (1-alpha) * embed_loss
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    return epoch_loss


def eval(sc, dev_enrol_set, dev_test_set, trials, labels):
    cuda = torch.cuda.is_available()
    batch_size = 6
    enrol_loader = DataLoader(dev_enrol_set, shuffle=False, num_workers=20, batch_size=batch_size)
    test_loader = DataLoader(dev_test_set, shuffle=False, num_workers=20, batch_size=batch_size)
    trials_set = TensorDataset(trials)
    trials_loader = DataLoader(trials_set, shuffle=False, num_workers=20, batch_size=batch_size)

    scores = []
    
    embedding_size = sc.embedding_size
    enrol_embeddings = torch.zeros((len(dev_enrol_set), embedding_size))
    test_embeddings = torch.zeros((len(dev_test_set), embedding_size))

    if cuda:
        enrol_embeddings = enrol_embeddings.cuda()
        test_embeddings = test_embeddings.cuda()


    for idx, batch in enumerate(enrol_loader):
        enrol_batch = batch
        if cuda:
            enrol_batch = enrol_batch.cuda()
        bsize = enrol_batch.size()[0]
        bidx = idx * bsize
        with torch.no_grad():
            enrol_embeddings[bidx:bidx + bsize, :] = sc(enrol_batch, em=True)

    for idx, batch in enumerate(test_loader):
        test_batch = batch
        if cuda:
            test_batch = test_batch.cuda()
        bsize = test_batch.size()[0]
        bidx = idx * bsize
        with torch.no_grad():
            test_embeddings[bidx:bidx + bsize, :] = sc(test_batch, em=True)

    for batch in trials_loader:
        idx_batch = batch[0]
        enrol_idx = idx_batch[:, 0]
        test_idx = idx_batch[:, 1]
        scores_batch = F.cosine_similarity(enrol_embeddings[enrol_idx], test_embeddings[test_idx], dim=1)
        scores.append(scores_batch.cpu())
    scores = torch.cat(scores).numpy()

    return EER(labels, scores)


class Trainer(tune.Trainable):
    def _setup(self):
        self.sen = model.SpeakerEmbedding2d(self.config["nspeakers"])
        self.siamese_sen = model.SiameseNet(self.sen)
        self.optimizer = optim.SGD(params=self.sen.parameters(), lr=self.config["lr"], weight_decay=0.0001)
        self.iteration = 1
        td = self.config['train_set_id']
        self.train_set = get_pinned_object(td)
        print("Loaded train")
        dd = self.config['dev_set_id']
        self.dev_enrol_set, self.dev_test_set, self.dev_trials, self.dev_labels = get_pinned_object(dd)
        print("Loaded dev")
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 30, 40, 50, 60, 80, 100, 120], gamma=0.5)
        cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cuda:
            self.sen.to(device)

    def _train(self):
        epoch_loss = train(self.siamese_sen, self.optimizer, self.train_set, self.config["nspeakers"], self.config["batch_size"], alpha=self.config["alpha"])
        self.lr_scheduler.step()

        # do eval every 1 epochs
        if self.iteration % 1 == 0:
            self.sen.eval()
            eer = eval(self.sen, self.dev_enrol_set, self.dev_test_set, self.dev_trials, self.dev_labels)[0]
            self.sen.train()
        else:
            eer = 1.0

        report = {"epoch_mean_loss": epoch_loss / (len(self.train_set) / self.config["batch_size"]), 'eer':eer, "training_iteration":self.iteration}

        self.iteration += 1
        return report

    def _save(self, checkpoint_dir):
        cpd = {'iteration': self.iteration,
               'state_dict': self.sc.state_dict(),
               'optimizer': self.optimizer.state_dict()
               }
        torch.save(cpd, checkpoint_dir + "/save")

    def _restore(self, path):
        cpd = torch.load(path)
        self.iteration = cpd['iteration']
        self.sc.load_state_dict(cpd['state_dict'])
        self.optimizer.load_state_dict(cpd['optimizer'])


if __name__ == "__main__":
    ray.init()
    dset = TensorDataset(torch.randn(100, 64, 1024), torch.randn(100, 1024),
                         torch.randint(100, size=(100,)).type(torch.LongTensor))

    dset_id = pin_in_object_store(dset)
    tune.register_trainable('train_sc', Trainer)
    exp = Experiment(
        name="speaker classification",
        run='train_sc',
        stop={"timesteps_total": 1},
        config={
            "lr": 1e-3,
            "dset_id": dset_id,
            "nspeakers": 100,
            "batch_size": 1,
        })

    tune.run_experiments(exp)
