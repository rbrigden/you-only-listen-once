import torch
import torch.nn as nn
import torch.nn.functional as F

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

import time
class MarginSearch:
    """ Use Bayesian hyperparameter search to find an optimal margin """

    def __init__(self, starting_point=None):
        self.hyperopt = BayesianOptimization(
            f=None,
            pbounds={'m': (0, 100)},
            verbose=2,
            random_state=1,
        )

        if starting_point:
            self.hyperopt.probe({'m': starting_point})

        # Prefer exploitation. We should experiment with some exploration tho
        self.util_func = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

        self._next_probe = None

    def next_probe(self):
        if self._next_probe is None:
            self._next_probe = self.hyperopt.suggest(self.util_func)
        else:
            raise ValueError("Need to update before getting next search probe")
        return self._next_probe["m"]

    def update(self, value):
        if self._next_probe is not None:
            self.hyperopt.register(params=self._next_probe, target=value)
            self._next_probe = None
        else:
            raise ValueError("Need to get the next probe before updating")

    @property
    def best_margin(self):
        if "params" in self.hyperopt.max:
            return self.hyperopt.max["params"]["m"]
        else:
            return None


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, num_search_steps, search_freq):
        """
        :param num_search_steps: Number of hyperopt steps per batch
        """
        super(ContrastiveLoss, self).__init__()
        self.init_margin = 0
        self.num_search_steps = num_search_steps
        self.search_freq = search_freq
        self.margin_searcher = MarginSearch()
        self.t = 0

    def _loss(self, embedding1, embedding2, target, margin, size_average=True):
        distances = (embedding2 - embedding1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(margin - distances.sqrt()).pow(2))
        return losses.mean()


    def forward(self, embedding1, embedding2, target, size_average=True):
        current_best_margin = self.margin_searcher.best_margin
        current_best_margin = self.init_margin if current_best_margin is None else current_best_margin

        loss_to_backward = self._loss(embedding1, embedding2, target, current_best_margin, size_average=size_average)

        e1, e2 = embedding1.data, embedding2.data

        # start = time.time()

        if self.t % self.search_freq == 0:
            for t in range(self.num_search_steps):
                margin = self.margin_searcher.next_probe()

                # BayesianOpt is maximizer
                score = - self._loss(e1, e2, target, margin, size_average=size_average)
                self.margin_searcher.update(score)
        self.t += 1

        # end = time.time()
        # print("dt: {}".format(round(end - start)))
        return loss_to_backward


    def reset(self):
        last_point = self.margin_searcher.best_margin
        self.margin_searcher = MarginSearch(starting_point=last_point)



class SimpleContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        """
        :param num_search_steps: Number of hyperopt steps per batch
        """
        super(SimpleContrastiveLoss, self).__init__()
        self.margin = margin

    def _loss(self, embedding1, embedding2, target, margin, size_average=True):
        distances = (embedding2 - embedding1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(margin - distances.sqrt()).pow(2))
        return losses.mean()


    def forward(self, embedding1, embedding2, target, size_average=True):

        loss = self._loss(embedding1, embedding2, target, self.margin, size_average=size_average)
        return loss




class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

