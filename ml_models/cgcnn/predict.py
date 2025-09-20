# Crystal Graph Convolutional Neural Network (CGCNN) prediction script
# Refactored to avoid global variables. Provides both a CLI and a callable
# function `predict_cgcnn` to run inference on CIF files with a pretrained CGCNN.

import argparse
import os
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet


def predict_cgcnn(
    modelpath: str,
    cifpath: str,
    batch_size: int = 256,
    workers: int = 0,
    disable_cuda: bool = False,
    print_freq: int = 10,
    chunk_id: int = 1,
    output_csv: Optional[str] = None,
) -> str:
    """
    Callable wrapper that prepares config and runs inference; returns CSV path.
    """
    if output_csv is None:
        output_csv = f"test_results_{chunk_id}.csv"

    args = SimpleNamespace(
        modelpath=modelpath,
        cifpath=cifpath,
        batch_size=batch_size,
        workers=workers,
        disable_cuda=disable_cuda,
        print_freq=print_freq,
        chunk_id=chunk_id,
        output_csv=output_csv,
    )

    model_args = _load_model_args(args.modelpath)

    # compute cuda flag once, store on args for convenience
    args.cuda = (not args.disable_cuda) and torch.cuda.is_available()

    _ = _run(args, model_args)  # returns (metric_value, csv_path)
    return os.path.abspath(args.output_csv)

def _load_model_args(modelpath: str) -> SimpleNamespace:
    """Load model hyperparameters from checkpoint; fall back to defaults."""
    if os.path.isfile(modelpath):
        print(f"=> loading model params '{modelpath}'")
        model_checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
        # Checkpoint is expected to have an "args" dict-like payload
        model_args = SimpleNamespace(**model_checkpoint["args"])
        print(f"=> loaded model params '{modelpath}'")
    else:
        print(f"=> no model params found at '{modelpath}'")
        model_args = SimpleNamespace(task="regression")  # minimal default
    return model_args

def _build_model(dataset: CIFData, model_args: SimpleNamespace, use_cuda: bool) -> Tuple[nn.Module, int, int]:
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=model_args.atom_fea_len,
        n_conv=model_args.n_conv,
        h_fea_len=model_args.h_fea_len,
        n_h=model_args.n_h,
        classification=(model_args.task == "classification"),
    )
    if use_cuda:
        model.cuda()
    return model, orig_atom_fea_len, nbr_fea_len


def _run(args: SimpleNamespace, model_args: SimpleNamespace):
    """Main evaluation entry. Returns (metric_value, csv_path)."""
    dataset = CIFData(args.cifpath)
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pool,
        pin_memory=args.cuda,
    )

    model, _, _ = _build_model(dataset, model_args, args.cuda)

    criterion = nn.NLLLoss() if model_args.task == "classification" else nn.MSELoss()
    normalizer = Normalizer(torch.zeros(3))

    if os.path.isfile(args.modelpath):
        print(f"=> loading model '{args.modelpath}'")
        checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        print(
            f"=> loaded model '{args.modelpath}' "
            f"(epoch {checkpoint['epoch']}, validation {checkpoint['best_mae_error']})"
        )
    else:
        print(f"=> no model found at '{args.modelpath}'")

    metric = _validate(args, model_args, test_loader, model, criterion, normalizer, test=True)
    return metric, args.output_csv


def _validate(
    args: SimpleNamespace,
    model_args: SimpleNamespace,
    val_loader,
    model,
    criterion,
    normalizer,
    test: bool = False,
):
    """Run evaluation loop; write predictions if test=True. Returns main metric."""
    batch_time = AverageMeter()
    losses = AverageMeter()

    if model_args.task == "regression":
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    model.eval()
    end = time.time()

    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (
                    Variable(input[0].cuda(non_blocking=True)),
                    Variable(input[1].cuda(non_blocking=True)),
                    input[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                )
            else:
                input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])

        if model_args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()

        with torch.no_grad():
            target_var = target_normed.cuda(non_blocking=True) if args.cuda else target_normed

        output = model(*input_var)
        loss = criterion(output, target_var)

        if model_args.task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_targets += target.view(-1).tolist()
                test_preds += test_pred.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                assert test_pred.shape[1] == 2
                test_targets += target.view(-1).tolist()
                test_preds += test_pred[:, 1].tolist()
                test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if model_args.task == "regression":
                print(
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})"
                )
            else:
                print(
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Accu {accuracies.val:.3f} ({accuracies.avg:.3f})\t"
                    f"Precision {precisions.val:.3f} ({precisions.avg:.3f})\t"
                    f"Recall {recalls.val:.3f} ({recalls.avg:.3f})\t"
                    f"F1 {fscores.val:.3f} ({fscores.avg:.3f})\t"
                    f"AUC {auc_scores.val:.3f} ({auc_scores.avg:.3f})"
                )

    if test:
        import csv
        with open(args.output_csv, "w") as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))

    if model_args.task == "regression":
        print(f" ** MAE {mae_errors.avg:.3f}")
        return mae_errors.avg
    else:
        print(f" ** AUC {auc_scores.avg:.3f}")
        return auc_scores.avg

class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def mae(prediction, target):
    """Compute mean absolute error between prediction and target tensors."""
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    """Evaluate classification predictions with accuracy, precision, recall, F1, and AUC."""
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average="binary"
        )
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Track and compute running averages of metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Save model checkpoint to file."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

def _build_argparser():
    parser = argparse.ArgumentParser(description="Crystal gated neural networks")
    parser.add_argument("modelpath", help="path to the trained model.")
    parser.add_argument("cifpath", help="path to the directory of CIF files.")
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--print-freq", "-p", default=10, type=int, metavar="N", help="print frequency")
    parser.add_argument("--chunk_id", type=int, default=1, help="Chunk index (1-based)")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path")
    return parser

if __name__ == "__main__":
    cli = _build_argparser().parse_args(sys.argv[1:])
    csv_path = predict_cgcnn(
        modelpath=cli.modelpath,
        cifpath=cli.cifpath,
        batch_size=cli.batch_size,
        workers=cli.workers,
        disable_cuda=cli.disable_cuda,
        print_freq=cli.print_freq,
        chunk_id=cli.chunk_id,
        output_csv=cli.output_csv,
    )
    print(f"Wrote predictions to: {csv_path}")
