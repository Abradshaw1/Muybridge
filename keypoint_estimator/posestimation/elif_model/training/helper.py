"""
Training/Evaluation Helper Functions for ElifPose
================================================

This module provides utility classes and functions for model training, evaluation,
benchmarking, and statistics tracking for the ElifPose pose estimation pipeline.

Contents:
- AverageMeter: Tracks and formats running statistics (e.g., accuracy, loss).
- evaluate: Runs model evaluation over a dataloader.
- print_size_of_model: Prints the size of a model checkpoint.
- train_one_epoch: Runs a single epoch of training.
- load_model: Loads model weights from a checkpoint.
- run_benchmark: Benchmarks inference speed of a TorchScript model.

Author: ETH Zurich Digital Circuits and Systems Group
Date: 2025-06-30
"""

import torch
import time
import os

class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric (e.g., loss, accuracy).
    Useful for tracking statistics during training or evaluation.
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the meter with a new value.
        Args:
            val (float): New value to add.
            n (int): Number of occurrences (default=1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Return a formatted string for printing."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def evaluate(model, data_loader, device, num_eval_batches=100):
    """
    Evaluate the model over a dataloader for a limited number of batches.
    Args:
        model: Model to evaluate (must have .loss method returning a dict with 'acc_pose').
        data_loader: DataLoader for evaluation data.
        device: Device to run evaluation on.
        num_eval_batches (int): Max number of batches to evaluate (default=100).
    Returns:
        AverageMeter: Accuracy statistics for the run.
    """
    model.eval()
    Sim_PCK = AverageMeter('Acc', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for inputs, samples in data_loader:
            inputs = inputs.to(device)
            # Move all tensors in samples to device if possible
            if isinstance(samples, list):
                samples = [s.to(device) if hasattr(s, 'to') else s for s in samples]
            elif hasattr(samples, 'to'):
                samples = samples.to(device)
            loss = model.loss(inputs, samples)
            acc = loss['acc_pose']
            Sim_PCK.update(acc.item(), inputs.size(0))
            if cnt >= num_eval_batches:
                return Sim_PCK
            cnt += 1
    return Sim_PCK

def print_size_of_model(model, name):
    """
    Print the size of a model's state_dict in megabytes.
    Args:
        model: Model whose size to print.
        name (str): Name label for the model.
    """
    torch.save(model.state_dict(), "temp.p")
    print(name, ' size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, device, optim_wrapper, train_dataset_length):
    """
    Run a single training epoch over the dataloader.
    Args:
        model: Model to train (must have .loss method returning a dict with 'loss_kpt' and 'acc_pose').
        dataloader: DataLoader for training data.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        epoch (int): Current epoch number.
        device: Device to run training on.
        optim_wrapper (dict): Contains 'clip_grad' for gradient clipping.
        train_dataset_length (int): Length of the training dataset.
    Returns:
        model: The trained model after this epoch.
    """
    model.train()
    i = 1
    for inputs, samples in dataloader:
        inputs = inputs.to(device)
        if isinstance(samples, list) and hasattr(samples[0], 'to'):
            samples = [s.to(device) for s in samples]
        optimizer.zero_grad()
        loss = model.loss(inputs, samples)
        loss['loss_kpt'].backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), **optim_wrapper['clip_grad'])
        optimizer.step()
        scheduler.step()
        if i % 50 == 0:
            print(f"Epoch: {epoch} Iteration: {i}/{train_dataset_length} Loss: {loss['loss_kpt'].item()}, Accuracy: {loss['acc_pose'].item()}")
        i += 1
    return model

def load_model(model, model_path):
    """
    Load model weights from a checkpoint file.
    Args:
        model: Model instance to load weights into.
        model_path (str): Path to the checkpoint file.
    Returns:
        model: The model with loaded weights.
    """
    model.load_state_dict(torch.load(model_path))
    return model

def run_benchmark(model_file, dataloader, num_batches=5):
    """
    Benchmark the inference time of a TorchScript model over several batches.
    Args:
        model_file (str): Path to the TorchScript model file.
        dataloader: DataLoader for input data.
        num_batches (int): Number of batches to benchmark (default=5).
    Returns:
        float: Total elapsed time in seconds for all batches.
    """
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    # Run the scripted model on a few batches of images
    for i, (inputs, samples) in enumerate(dataloader):
        if i < num_batches:
            start = time.time()
            output = model.loss(inputs, samples)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = inputs.size()[0] * num_batches
    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed