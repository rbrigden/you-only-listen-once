import time
import argparse
import torch
import torch.utils.data as dutils
import training.speaker_classification.train as train
import training.speaker_verification.verify as verify

import data.voxceleb.voxceleb as voxceleb


def train_speaker_classifier(args):
    speakers = list(range(1200))

    trainer = train.SpeakerClassifierTrainer(batch_size=args.batch_size, learning_rate=args.lr,
                                             num_speakers=len(speakers))

    if args.resume is not None:
        print("Loading classifier params from {}".format(args.resume))
        trainer.resume(args.resume)
    else:
        print("Initialize classifier params from scratch")

    train_set, val_set = voxceleb.VoxcelebID.create_split(args.voxceleb_path, speakers, split=0.8, shuffle=True)

    # Voxceleb length stats: (mean = 356, min = 171, max = 6242, std = 230)
    train_collate_fn = voxceleb.voxceleb_clip_collate(512, sample=True)
    val_collate_fn = voxceleb.voxceleb_clip_collate(1024, sample=False)



    train_loader = dutils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=8)
    val_loader = dutils.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=val_collate_fn, num_workers=8)

    # Verification EER computation
    veri_evaluator = verify.VerificationEvaluator(args.voxceleb_test_path)

    total_samples_processed = 0

    for e in range(args.epochs):
        total_loss = 0
        nbatches = 0
        runner = trainer.train_epoch(train_loader)

        # Train for an epoch
        start = time.time()
        num_correct = 0
        for loss, correct in runner:
            num_correct += correct
            total_samples_processed += args.batch_size
            nbatches += 1
            total_loss += loss
        end = time.time()

        # Compute validation error
        print("saving checkpoint")
        trainer.checkpoint("models/speaker_classification/simple.pt")

        epoch_time = end - start
        train_error = 1 - (num_correct / float(len(train_set)))
        val_error = trainer.validation(val_loader)
        veri_eer = trainer.compute_verification_eer(veri_evaluator)

        print("Epoch {} of {} took {} seconds \n"
              "\tEpoch loss: {}, \n"
              "\tTraining error: {}\n"
              "\tValidation error: {}\n"
              "\tEER: {}\n".format(e+1, args.epochs, epoch_time, total_loss / nbatches, train_error, val_error, veri_eer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb-path', type=str, default="/home/rbrigden/voxceleb/processed")
    parser.add_argument('--voxceleb-test-path', type=str, default="/home/rbrigden/voxceleb/test/processed")

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    train_speaker_classifier(parser.parse_args())
