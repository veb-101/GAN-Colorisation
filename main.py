import random
import numpy as np
from tqdm.notebook import tqdm

import torch
from data_loader import make_dataloaders
from utils import create_loss_meters, log_results, update_losses, visualize
from trainer import MainModel


seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, train_dl, epochs, display_every=200):
    data = next(
        iter(valid_dl)
    )  # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = (
            create_loss_meters()
        )  # function returing a dictionary of objects to
        i = 0  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            update_losses(
                model, loss_meter_dict, count=data["L"].size(0)
            )  # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict)  # function to print out the losses
                visualize(
                    model, data, save=False
                )  # function displaying the model's outputs


if __name__ == "__main__":
    train_ = r"images\train"
    valid_ = r"images\valid"

    device = get_default_device()

    train_dl = make_dataloaders(path=train_, batch_size=1)
    valid_dl = make_dataloaders(
        path=valid_, batch_size=1, is_training=False, shuffle=False
    )

    # initialize once per epoch
    training_loader_iter = iter(train_dl)
    length = len(training_loader_iter)
    # get batches
    for i in range(length):
        data = next(training_loader_iter)
        Ls, abs_ = data["L"], data["ab"]
        print(Ls.shape, abs_.shape)
        break

    print(f"Number of batches ::Train:: {len(train_dl)}, ::Valid:: {len(valid_dl)}")

