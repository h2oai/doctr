# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from cmath import inf
from distutils.command.build import build
import os

os.environ['USE_TORCH'] = '1'

import datetime
import hashlib
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from fastprogress.fastprogress import master_bar, progress_bar
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import ColorJitter, Compose, Normalize
import torchvision.transforms as TT

from doctr import transforms as T
from doctr.datasets import VOCABS, RecognitionDataset, WordGenerator, FUNSD, SynthText, CORD, IMGUR5K
from doctr.models import login_to_hub, push_to_hf_hub, recognition
from doctr.utils.metrics import TextMatch
from utils import plot_recorder, plot_samples
import cv2
import subprocess
import shutil
from torch import nn
from torch.nn import functional as F
from doctr.models.recognition.core import build_targets, calc_loss
from doctr.models.recognition.crnn import CTCPostProcessor

def record_lr(
    model: torch.nn.Module,
    train_loader: DataLoader,
    batch_transforms,
    optimizer,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_it: int = 100,
    amp: bool = False
):
    """Gridsearch the optimal learning rate for the training.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py
    """

    if num_it > len(train_loader):
        raise ValueError("the value of `num_it` needs to be lower than the number of available batches")

    model = model.train()
    # Update param groups & LR
    optimizer.defaults['lr'] = start_lr
    for pgroup in optimizer.param_groups:
        pgroup['lr'] = start_lr

    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    scheduler = MultiplicativeLR(optimizer, lambda step: gamma)

    lr_recorder = [start_lr * gamma ** idx for idx in range(num_it)]
    loss_recorder = []

    if amp:
        scaler = torch.cuda.amp.GradScaler()

    for batch_idx, (images, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()

        images = batch_transforms(images)
        targets, seq_len = build_targets(targets, model.module.vocab)
        # Forward, Backward & update
        optimizer.zero_grad()
        if amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
            scaler.scale(train_loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, targets)['loss']
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        # Update LR
        scheduler.step()

        # Record
        if not torch.isfinite(train_loss):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            else:
                break
        loss_recorder.append(train_loss.item())
        # Stop after the number of iterations
        if batch_idx + 1 == num_it:
            break

    return lr_recorder[:len(loss_recorder)], loss_recorder


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, mb,
                  amp=False, val_loader = None, val_metric = None, eval_steps = None, min_loss = inf, exp_name = None, epoch = None, args = None):

    if amp:
        scaler = torch.cuda.amp.GradScaler()

    # Iterate over the batches of the dataset
    train_log = []
    for steps, (images, targets) in enumerate(progress_bar(train_loader, parent=mb)):
        model.train()

        if torch.cuda.is_available():
            images = images.cuda()
        targets, seq_len = build_targets(targets, model.module.vocab)
        images = images.expand(images.shape[0], 3, images.shape[2], images.shape[3])
        images = batch_transforms(images)

        # train_loss = model(images, targets)['loss']

        optimizer.zero_grad()
        if amp:
            with torch.cuda.amp.autocast():
                train_loss = model(images)['loss']
            scaler.scale(train_loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            train_loss = calc_loss(logits, targets, seq_len, model.module.vocab)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if args.sched == "reducelr":
            pass
        else:
            scheduler.step()
        train_log.append(train_loss.item())
        mb.child.comment = f'Training loss: {np.mean(train_log):.6}'
        if eval_steps is not None:
            if steps % eval_steps == 0:
                train_logs = []
                val_loss, exact_match, partial_match, levenshtein_distance = evaluate(model, val_loader, batch_transforms, val_metric, amp=amp, eval_errors = args.eval_errors)
                if args.sched == "reducelr":
                    scheduler.step(val_loss)
                if args.wb:
                    wandb.log({
                        'training_loss':np.mean(train_log),
                        'learning_rate': scheduler._last_lr[0],
                        'val_loss': val_loss,
                        'exact_match': exact_match,
                        'partial_match': partial_match,
                        'levenshtein_distance': levenshtein_distance,
                    })
                if val_loss < min_loss:
                    print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
                    torch.save(model.state_dict(), f"./{exp_name}.pt")
                    min_loss = val_loss
                    mb.write(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                             f"(Exact: {exact_match:.2%} | Partial: {partial_match:.2%} | Levenshtein: {levenshtein_distance:.2%})")




@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False, eval_errors = False):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    if eval_errors:
        try:
            shutil.rmtree("error_samples")
        except:
            pass
        os.makedirs("error_samples", exist_ok = True)
    for images, targets in val_loader:
        orig_targets = targets
        if torch.cuda.is_available():
            images = images.cuda()
        targets, seq_len = build_targets(targets, model.module.vocab)
        images = images.expand(images.shape[0], 3, images.shape[2], images.shape[3])
        images = batch_transforms(images)
        if amp:
            with torch.cuda.amp.autocast():
                out = model(images, return_preds=True)
        else:
            out = {}
            logits = model(images)
            out["loss"] = calc_loss(logits, targets, seq_len, model.module.vocab)
            postprocessor = CTCPostProcessor(vocab=model.module.vocab)
            out["preds"] = postprocessor(logits)

        # Compute metric
        if len(out['preds']):
            words, _ = zip(*out['preds'])
        else:
            words = []
        val_metric.update(orig_targets, words)
        if eval_errors:
            for i, (word, target) in enumerate(zip(words, orig_targets)):
                if word != target:
                    img = images[i].permute((1, 2, 0)).detach().cpu().numpy()
                    cv2.imwrite(f"./error_samples/{word}_{target}.jpg", img*255)
        val_loss += out['loss'].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result['raw'], result['unicase'], result['levenshtein_distance']


def main(args):

    print(args)

    if args.push_to_hub:
        login_to_hub()

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    torch.backends.cudnn.benchmark = True

    vocab = VOCABS[args.vocab]
    if args.add_space:
        vocab += " "
    fonts = args.font.split(",")
    if args.all_fonts:
        fonts = []
        font_str = subprocess.check_output("fc-list ':lang=fr'", shell = True).decode().split("\n")
        for font in font_str:
            fonts.append(font.split(":")[0])
        print(len(fonts))
    # Load val data generator
    st = time.time()
    if isinstance(args.val_path, str):
        with open(os.path.join(args.val_path, 'labels.json'), 'rb') as f:
            val_hash = hashlib.sha256(f.read()).hexdigest()

        val_set = RecognitionDataset(
            img_folder=os.path.join(args.val_path, 'images'),
            labels_path=os.path.join(args.val_path, 'labels.json'),
            img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
        )
    elif args.funsd_val:
        val_set = FUNSD(train=False, download=True, recognition_task = True,
                img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
            ]),
        )
    elif args.cord_val:
        val_set = CORD(train=False, download=True, recognition_task = True,
                img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
            ]),
        )
    else:
        val_hash = None
        # Load synthetic data generator
        val_set = WordGenerator(
            vocab=vocab,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            num_samples=args.val_samples * len(vocab),
            font_family=fonts,
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # Ensure we have a 90% split of white-background images
                T.RandomApply(T.ColorInversion(), 0.9),
            ]),
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(val_set),
        pin_memory=torch.cuda.is_available(),
        collate_fn=val_set.collate_fn,
    )
    print(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in "
          f"{len(val_loader)} batches)")

    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

    # Load doctr model
    model = recognition.__dict__[args.arch](pretrained=args.pretrained, vert_stride = args.vert_stride, vocab=vocab)
    model = nn.DataParallel(model)

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint)

    # GPU
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    # Silent default switch to GPU if available
    elif torch.cuda.is_available():
        args.device = 0
    else:
        logging.warning("No accessible GPU, targe device set to CPU.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()

    # Metrics
    val_metric = TextMatch()

    if args.test_only:
        print("Running evaluation")
        val_loss, exact_match, partial_match, levenshtein_distance = evaluate(model, val_loader, batch_transforms, val_metric, amp=args.amp, eval_errors = args.eval_errors)
        print(f"Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Partial: {partial_match:.2%} | Levenshtein: {levenshtein_distance:.2%} )")
        return

    st = time.time()

    if isinstance(args.train_path, str):
        train_paths = args.train_path.split(",")
        # Load train data generator
        base_path = Path(train_paths[0])
        parts = [base_path] if base_path.joinpath('labels.json').is_file() else [
            base_path.joinpath(sub) for sub in os.listdir(base_path)
        ]
        with open(parts[0].joinpath('labels.json'), 'rb') as f:
            train_hash = hashlib.sha256(f.read()).hexdigest()
        train_set = RecognitionDataset(
            "./" + str(parts[0].joinpath('images')),
            "./" + str(parts[0].joinpath('labels.json')),
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                T.RandomApply(ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02), 0.2),
                T.RandomApply(TT.GaussianBlur(kernel_size=(3, 3), sigma=(.1, 2)), .2),
                T.RandomApply(TT.RandomPerspective(distortion_scale=.3), .2),
                T.RandomApply(TT.RandomRotation(5), .2),
            ]),
        )
        if len(parts) > 1:
            for subfolder in parts[1:]:
                train_set.merge_dataset(RecognitionDataset(
                    subfolder.joinpath('images'), subfolder.joinpath('labels.json')))
        if len(train_paths) > 1:
            for path in train_paths[1:]:
                train_set.merge_dataset(RecognitionDataset(
                        "./" + str(Path(path).joinpath('images')), "./" + str(Path(path).joinpath('labels.json'))))
    elif args.train_synthtext:
        train_hash = None
        # Load synthtext dataset
        train_set = SynthText(train = True, predownload_path = args.train_synthtext, recognition_task=True,
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # Ensure we have a 90% split of white-background images
                T.RandomApply(T.ColorInversion(), 0.1),
                T.RandomApply(ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02), 0.2),
                T.RandomApply(TT.GaussianBlur(kernel_size=(3, 3), sigma=(.1, 2)), .2),
                T.RandomApply(TT.RandomPerspective(distortion_scale=.8), .5),
                T.RandomApply(TT.RandomRotation(10), .2),
            ]),
        )
    elif args.funsd_train:
        train_hash = None
        train_set = FUNSD(train=True, download=True, recognition_task = True,
                img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # T.RandomApply(T.ColorInversion(), 0.9),
                # T.RandomApply(ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02), .5),
                # T.RandomApply(TT.GaussianBlur(kernel_size=(3, 3), sigma=(.1, .5)), .1),
                # T.RandomApply(TT.RandomPerspective(distortion_scale=.8), .5),
                T.RandomApply(TT.RandomRotation(5), .2),
                # Ensure we have a 90% split of white-background images

            ]),
        )
    elif args.cord_train:
        train_hash = None
        train_set = CORD(train=True, download=True, recognition_task = True,
                img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # T.RandomApply(T.ColorInversion(), 0.9),
                # T.RandomApply(ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02), .5),
                # T.RandomApply(TT.GaussianBlur(kernel_size=(3, 3), sigma=(.1, .5)), .1),
                # T.RandomApply(TT.RandomPerspective(distortion_scale=.8), .5),
                T.RandomApply(TT.RandomRotation(5), .2),
                # Ensure we have a 90% split of white-background images

            ]),
        )
    elif args.imgur_train:
        train_hash = None
        train_set = IMGUR5K(train=True, download=True, recognition_task = True,
                img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # T.RandomApply(T.ColorInversion(), 0.9),
                # T.RandomApply(ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02), .5),
                # T.RandomApply(TT.GaussianBlur(kernel_size=(3, 3), sigma=(.1, .5)), .1),
                # T.RandomApply(TT.RandomPerspective(distortion_scale=.8), .5),
                T.RandomApply(TT.RandomRotation(5), .2),
                # Ensure we have a 90% split of white-background images

            ]),
        )
    else:
        train_hash = None
        # Load synthetic data generator
        train_set = WordGenerator(
            vocab=vocab,
            word_list = args.external_vocab,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            num_samples=args.train_samples * len(vocab),
            font_family=fonts,
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                T.RandomApply(T.ColorInversion(), 0.9),
                T.RandomApply(ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02), .5),
                T.RandomApply(TT.GaussianBlur(kernel_size=(3, 3), sigma=(.1, .5)), .1),
                # T.RandomApply(TT.RandomPerspective(distortion_scale=.8), .5),
                T.RandomApply(TT.RandomRotation(5), .2),
                # Ensure we have a 90% split of white-background images

            ]),
        )
    if args.train_samples is not None:
        sampler = RandomSampler(train_set, num_samples = args.train_samples * args.batch_size, replacement = True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=sampler,
        shuffle = shuffle,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_set.collate_fn,
    )
    print(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in "
          f"{len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    # Optimizer
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], args.lr,
                                 betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    # LR Finder
    if args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
        plot_recorder(lrs, losses)
        return
    # Scheduler
    if args.sched == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4)
    elif args.sched == 'onecycle':
        scheduler = OneCycleLR(optimizer, args.lr, args.epochs * len(train_loader))
    elif args.sched == "reducelr":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, cooldown=10, min_lr=1e-8, eps=1e-08, verbose = True)

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}_{current_time}" if args.name is None else args.name

    # W&B
    if args.wb:

        run = wandb.init(
            name=exp_name,
            project="text-recognition",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "architecture": args.arch,
                "input_size": args.input_size,
                "optimizer": "adam",
                "framework": "pytorch",
                "scheduler": args.sched,
                "vocab": args.vocab,
                "train_hash": train_hash,
                # "val_hash": val_hash,
                "pretrained": args.pretrained,
            }
        )

    # Create loss queue
    min_loss = np.inf
    # Training loop
    mb = master_bar(range(args.epochs))
    for epoch in mb:
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, mb, amp=args.amp,
                      val_loader = val_loader, val_metric = val_metric, eval_steps=args.eval_steps, 
                      min_loss = min_loss, exp_name = exp_name, epoch = epoch, args = args)
        # Validation loop at the end of each epoch
        val_loss, exact_match, partial_match, levenshtein_distance = evaluate(model, val_loader, batch_transforms, val_metric, amp=args.amp, eval_errors = args.eval_errors)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            if args.eval_steps is None:
                torch.save(model.state_dict(), f"./{exp_name}.pt")
            min_loss = val_loss
        mb.write(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                 f"(Exact: {exact_match:.2%} | Partial: {partial_match:.2%} | Levenshtein: {levenshtein_distance:.2%} )")
        # W&B
        if args.wb:
            wandb.log({
                'learning_rate': scheduler._last_lr[0],
                'val_loss': val_loss,
                'exact_match': exact_match,
                'partial_match': partial_match,
                'levenshtein_distance': levenshtein_distance,
            })
    torch.save(model.state_dict(), f"./{exp_name}_final.pt")
    if args.wb:
        run.finish()

    if args.push_to_hub:
        push_to_hf_hub(model, exp_name, task='recognition', run_config=args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR training script for text recognition (PyTorch)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('arch', type=str, help='text-recognition model to train')
    parser.add_argument('--train_path', type=str, default=None, help='path to train data folder(s)')
    parser.add_argument('--val_path', type=str, default=None, help='path to val data folder')
    parser.add_argument(
        '--train-samples',
        type=int,
        default=None,
        help='Multiplied by the vocab length gets you the number of synthetic training samples that will be used.'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=None,
        help='Number of steps to take before model evaluation'
    )
    parser.add_argument(
        '--val-samples',
        type=int,
        default=20,
        help='Multiplied by the vocab length gets you the number of synthetic validation samples that will be used.'
    )
    parser.add_argument(
        '--font',
        type=str,
        default="FreeMono.ttf,FreeSans.ttf,FreeSerif.ttf",
        help='Font family to be used'
    )
    parser.add_argument('--min-chars', type=int, default=1, help='Minimum number of characters per synthetic sample')
    parser.add_argument('--max-chars', type=int, default=32, help='Maximum number of characters per synthetic sample')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--device', default=None, type=int, help='device')
    parser.add_argument('--input_size', type=int, default=32, help='input size H for the model, W = 4*H')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument('-j', '--workers', type=int, default=None, help='number of workers used for dataloading')
    parser.add_argument('--resume', type=str, default=None, help='Path to your checkpoint')
    parser.add_argument('--vocab', type=str, default="french", help='Vocab to be used for training')
    parser.add_argument("--test-only", dest='test_only', action='store_true', help="Run the validation loop")
    parser.add_argument('--show-samples', dest='show_samples', action='store_true',
                        help='Display unormalized training samples')
    parser.add_argument('--wb', dest='wb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--push-to-hub', dest='push_to_hub', action='store_true', help='Push to Huggingface Hub')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Load pretrained parameters before starting the training')
    parser.add_argument('--sched', type=str, default='cosine', help='scheduler to use')
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument('--find-lr', action='store_true', help='Gridsearch the optimal LR')
    parser.add_argument('--funsd_val', action='store_true', help='use funsd for validation instead of generated words')
    parser.add_argument('--funsd_train', action='store_true', help='use funsd for validation instead of generated words')
    parser.add_argument('--cord_train', action='store_true', help='use CORD for training')
    parser.add_argument('--cord_val', action='store_true', help='use CORD for validation')
    parser.add_argument('--imgur_train', action='store_true', help='use IMGUR5k for training')
    parser.add_argument('--all_fonts', action='store_true', help='pull all installed fonts from the system for generation')
    parser.add_argument('--external_vocab', type=str, default=None, help='Use external vocab for word generation')
    parser.add_argument('--train_synthtext', type=str, default=None, help='train against synthtext dataset')
    parser.add_argument('--eval_errors', action='store_true', help='write out images the model fails on')
    parser.add_argument('--add_space', action='store_true', help='add spaces to the vocabulary')
    parser.add_argument('--vert_stride', action='store_true', help='changes vertical stride to 1, giving model some vertical resolution in the LSTM')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
