import os

import random
from datetime import timedelta

import numpy as np
import torch
import numpy as np

from omegaconf import OmegaConf

from src import datasets
from src.model import BERT, BERTForClassification
import wandb




def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt


def get_model_and_optimizer(opt):

    if opt.pretraining:
        model = BERT(opt)


        if "cuda" in opt.device or "mps" in opt.device:
            model = model.to(opt.device)
        print(model, "\n")

        # Create optimizer with different hyper-parameters for the main model
        # and the downstream classification model.
        main_model_params = [
            p
            for p in model.parameters()
            if all(p is not x for x in model.criterion.parameters())
        ]

        if opt.training.optimizer == "SGD":
            
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": main_model_params,
                        "lr": opt.training.learning_rate,
                        "weight_decay": opt.training.weight_decay,
                        "momentum": opt.training.momentum,
                    },
                    {
                        "params": model.criterion.parameters(),
                        "lr": opt.training.downstream_learning_rate,
                        "weight_decay": opt.training.downstream_weight_decay,
                        "momentum": opt.training.momentum,
                    },
                ]
            )
        elif opt.training.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": main_model_params,
                        "lr": opt.training.learning_rate,
                        "weight_decay": opt.training.weight_decay,
                        "betas": (opt.training.betas[0] , opt.training.betas[1]), 
                    },
                    {
                        "params": model.criterion.parameters(),
                        "lr": opt.training.downstream_learning_rate,
                        "weight_decay": opt.training.downstream_weight_decay,
                        "betas": (opt.training.betas[0] , opt.training.betas[1]), 
                    },
                ]
            )
        return model, optimizer
    elif opt.fine_tune.task == "classification":
        model = BERTForClassification(opt)
        if "cuda" in opt.device or "mps" in opt.device:
            model = model.to(opt.device)
        if opt.fine_tune.freeze_bert:
            for param in model.bert.parameters():
                param.requires_grad = False

        if opt.training.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=opt.training.downstream_learning_rate,
                weight_decay=opt.training.downstream_weight_decay,
                momentum=opt.training.momentum,
            )
        elif opt.training.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=opt.training.downstream_learning_rate,
                weight_decay=opt.training.downstream_weight_decay,
                betas=(opt.training.betas[0], opt.training.betas[1]),
            )
        return model, optimizer
    else:
        raise ValueError(f"Unsupported fine-tune task: {opt.fine_tune.task}")
    

# 784, 2000, 2000, 2000 # main params
# 6000, 10 # classification_loss params

## INSERIRE L'IDEA DI PARTITIONING
def get_data(opt, partition, vocab):
    if partition == "train":
        path = os.path.join(opt.input.dataset_path, f"{opt.input.dataset_name}_train.tsv")
        corpus_lines = opt.input.train_corpus_lines
    elif partition == "val" or partition == "test":
        path = os.path.join(opt.input.dataset_path, f"{opt.input.dataset_name}_test.tsv")
        corpus_lines = opt.input.test_corpus_lines

    else:
        raise ValueError("Unknown partition.")
    
    

    if opt.input.dataset_name == "wikipedia":
        dataset = datasets.BERTDataset(corpus_path= path , vocab = vocab, seq_len= opt.input.seq_len,
                                        corpus_lines= corpus_lines, on_memory= opt.input.on_memory)
    elif opt.input.dataset_name == "quora":
        dataset = datasets.QuoraDataset(vocab=vocab, seq_len=opt.input.seq_len,
                                        corpus_lines=corpus_lines)
    else:
        raise ValueError("Unknown dataset.")

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=opt.training.num_workers,
        persistent_workers=True
    )



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dict_to_cuda(opt, obj):
    if isinstance(obj, dict):
        obj = {key: value.to(opt.device) for key, value in obj.items()}
    return obj


def preprocess_inputs(opt, data):
    if "cuda" in opt.device or "mps" in opt.device:
        data = dict_to_cuda(opt, data)
    return data
 
# cools down after the first half of the epochs
def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    if opt.pretraining:
        optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
            opt, epoch, opt.training.learning_rate
        )
        optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
            opt, epoch, opt.training.downstream_learning_rate
        )

    else:
        optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
            opt, epoch, opt.training.downstream_learning_rate
        )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value
    wandb.log(partition_scalar_outputs, step=epoch)

# create save_model function
def save_model(model):
    torch.save(model.state_dict(), f"{wandb.run.name}-model.pt")
    # log model to wandb
    wandb.save(f"{wandb.run.name}-model.pt")


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict