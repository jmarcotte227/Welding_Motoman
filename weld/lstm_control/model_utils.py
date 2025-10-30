import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def save_model(epoch, 
                    model_state_dict,
                    val_loss,
                    config,
                    data_mean,
                    data_std,
                    train_losses,
                    valid_losses,
                    save_dir):

    checkpoint = {
        'epoch' : epoch,
        'model_state_dict' : model_state_dict,
        'val_loss' : val_loss,

        # model architecture
        'input_dim' : config.INPUT_SIZE,
        'hidden_dim' : config.HIDDEN_SIZE,
        'output_dim' : config.OUTPUT_SIZE,
        'num_layers' : config.NUM_LAYERS,
        'dropout' : config.DROPOUT,

        # data regularization
        'data_mean' : data_mean,
        'data_std' : data_std,
    }

    filename = (f"model_h-{config.HIDDEN_SIZE}_part-{config.SET_NO}_"
                f"loss-{val_loss:.4f}")
    filepath = os.path.join(save_dir,filename)
    os.makedirs(filepath, exist_ok=True)
    # save checkpoint
    torch.save(checkpoint, f"{filepath}/{filename}.pt")

    # save loss data
    np.savetxt(f"{save_dir}/{filename}/train_loss.csv", train_losses, delimiter=',')
    np.savetxt(f"{save_dir}/{filename}/valid_loss.csv", valid_losses, delimiter=',')

class DataReg():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def reg(self, data, idx=None):
        if idx is None:
            return (data-self.mean)/self.std
        return (data-self.mean[idx])/self.std[idx]

    def unreg(self, data, idx=None):
        if idx is None:
            return (data*self.std)+self.mean
        return (data*self.std[idx])+self.mean[idx]
    def scale(self, data, idx=None):
        if idx is None:
            return (data/self.std)
        return (data/self.std[idx])

    def unscale(self, data, idx=None):
        if idx is None:
            return (data*self.std)
        return (data*self.std[idx])
