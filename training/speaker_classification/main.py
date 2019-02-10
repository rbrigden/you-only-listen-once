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

    train_loader = dutils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=voxceleb.voxceleb_collate, num_workers=8)
    val_loader = dutils.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=voxceleb.voxceleb_collate, num_workers=8)

    # Verification EER computation
    veri_evaluator = verify.VerificationEvaluator(args.voxceleb_test_path)

    total_samples_processed = 0
    start = time.time()

    for e in range(args.epochs):
        total_loss = 0
        nbatches = 0
        print("Epoch {}/{}".format(e + 1, args.epochs))
        runner = trainer.train_epoch(train_loader)

        # Train for an epoch
        for loss in runner:
            end = time.time()
            total_samples_processed += args.batch_size
            nbatches += 1
            total_loss += loss

        # Compute validation error
        print("saving checkpoint")
        trainer.checkpoint("models/speaker_classification/simple.pt")
        val_error = trainer.validation(val_loader)
        veri_eer = trainer.compute_verification_eer(veri_evaluator)

        print("Epoch loss: {}, Validation error: {}, EER: {}".format(total_loss / nbatches, val_error, veri_eer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb-path', type=str, default="/home/rbrigden/voxceleb/processed")
    parser.add_argument('--voxceleb-test-path', type=str, default="/home/rbrigden/voxceleb/test/processed")

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    train_speaker_classifier(parser.parse_args())
