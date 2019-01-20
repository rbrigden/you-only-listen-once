import numpy as np
import os
import argparse
import ray
import hw2.util as U
import hw2.train as train
import hw2.process as processor
import torch
import ray
import ray.tune as tune
from ray.tune.schedulers import HyperBandScheduler
from torch.utils.data import TensorDataset
from ray.tune.util import pin_in_object_store
from ray.tune import Experiment


TRAINING_CHUNK_SIZE = 8000
EVAL_CHUNK_SIZE = 16000


def serializer(obj):
    return obj.cpu().detach().numpy()


def deserializer(serialized_obj):
    return torch.from_numpy(serialized_obj)


def load_dev_set(args):
    dev_path = os.path.join(os.getcwd(), "data/dev/dev.preprocessed.npz")
    trials, labels, enrol, test = U.utils.dev_load(dev_path)
    trials = np.array([[t[0], t[1]] for t in trials]).reshape(-1, 2).astype(np.int)
    dev_enrol_set = processor.FixedWidthUtteranceSample(enrol, EVAL_CHUNK_SIZE)
    dev_test_set = processor.FixedWidthUtteranceSample(test, EVAL_CHUNK_SIZE)
    return (dev_enrol_set, dev_test_set, torch.LongTensor(trials), labels)

def load_train_set(args):
    chunk_path = os.path.join(os.getcwd(), "data/chunks/")
    train_data = U.utils.train_load(chunk_path, range(1, 1 + args.chunks))
    print("Loading {} chunks".format(args.chunks))
    features, speakers, nspeakers = train_data
    train_set = processor.SiameseFixedWidthUtteranceSampleWithSpeakers(features,
            speakers, TRAINING_CHUNK_SIZE)
    print("training length: {}".format(train_set))
    return nspeakers, train_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Speech Verification")
    parser.add_argument("--ray", action='store_true', default=False)
    parser.add_argument("--data-parallel", action='store_true', default=False)
    parser.add_argument("--chunks", type=int, default=1)

    args = parser.parse_args()
    print(args)
    ray.init(num_gpus=1)
    ray.register_custom_serializer(torch.Tensor, serializer=serializer, deserializer=deserializer)

    # Load train set
    nspeakers, train_set = load_train_set(args)
    train_set_id = pin_in_object_store(train_set)

    print("Loaded train. pinned={}".format(True))

    # Load dev set
    dev_set = load_dev_set(args)
    dev_set_id = pin_in_object_store(dev_set)
    print("Loaded dev. pinned={}".format(True))

    tune.register_trainable('train_sc', train.Trainer)
    exp = Experiment(
        name="speaker classification",
        run='train_sc',
        config={
            "stop": {'training_iteration': 500},
            # "lr": ray.tune.grid_search([1e-1, 1e-2]),
            "lr": 1e-2,
            "alpha": 0.9,
            "train_set_id": train_set_id,
            "dev_set_id": dev_set_id,
            "nspeakers": nspeakers,
            # "batch_size": ray.tune.grid_search([128, 64, 32]),
            "batch_size" : 32,
            "data_parallel": args.data_parallel
        },
        trial_resources={
            "cpu": 20,
            "gpu": 1
        },

    )

    tune.run_experiments(exp, with_server=True, server_port=4321)
