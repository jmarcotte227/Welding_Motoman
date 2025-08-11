import sys
import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from lstm_model_next_step import WeldLSTM

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train(train_dataset, valid_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model parameters
    INPUT_SIZE = 3
    HIDDEN_SIZE = 8
    OUTPUT_SIZE = 2
    NUM_LAYERS = 1

    # hyper parameters
    BATCH_SIZE = 20
    MAX_EPOCH = 2000
    LR = 0.005
    WD = 0.0001
    DROPOUT = 0

    # model
    model = WeldLSTM(INPUT_SIZE,
                     HIDDEN_SIZE,
                     OUTPUT_SIZE,
                     NUM_LAYERS,
                     DROPOUT)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR,
                                 weight_decay=WD)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                              lr=LR)
    # loss
    loss_fn = nn.MSELoss()


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle = True,
                                  drop_last = True
                                  )

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=1,
                                  shuffle = False
                                  )
    
    # initialize storage lists
    train_losses = []
    valid_losses = []
    best_loss = 1e6

    # setup progress bar
    pbar = tqdm(range(MAX_EPOCH))
    # Training Loop
    for epoch in pbar:
        train_loss = []
        model.train()
        for src,trg in train_dataloader:
            src = src.to(device)
            trg = trg.to(device)

            pred = model(src[:,:,[0,2,3]])

            optimizer.zero_grad()
            loss = loss_fn(pred, trg)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # validation
        valid_loss = []
        model.eval()
        for src,trg in valid_dataloader:
            src = src.to(device)
            trg = trg.to(device)

            # vset, avg temp, dh
            pred = model(src[:,:,[0,2,3]])

            # pred = torch.squeeze(pred,0)
            loss = loss_fn(pred, torch.squeeze(trg))

            valid_loss.append(loss.item())

        # compute averages
        train_loss = sum(train_loss)/len(train_loss)
        valid_loss = sum(valid_loss)/len(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # check if valid loss is better than best and save
        if valid_loss<best_loss:
            torch.save(model, f'saved_model_8_next_step.pt')
            best_loss = valid_loss

        # print()
        # print(f'-----Epoch {epoch} Stats-----')
        # print(f'Training Loss: {train_loss}')
        # print(f'Valid Loss:    {valid_loss}')
        pbar.set_description(f"T/V: {train_loss:.4f}/{valid_loss:.4f}")

    print("----- Final Results -----")
    print(f'Min Valid Loss: {min(valid_losses)}')
    print(f'Epoch: {valid_losses.index(min(valid_losses))}')
    print(f'Epoch: {valid_losses.index(min(valid_losses))}')

    fig,ax = plt.subplots()
    ax.plot(train_losses)
    ax.plot(valid_losses)
    ax.grid()
    plt.show()

class WeldDataset(Dataset):
    '''
    To be used with the processed files from a welding run.
    '''
    def __init__(self,
                 filepath,
                 mean=None,
                 std=None,
                 norm=True):

        # load arguments
        self.mean = mean
        self.std = std
        
        data = np.load(filepath)

        # standardize data
        temp_data = np.reshape(data, (-1,4))
        # print(np.nanmax(temp_data[:,2]))
        # print(np.nanmin(temp_data[:,2]))
        
        if self.mean is None:
            self.mean = np.nanmean(temp_data,axis=0)
            self.std = np.nanstd(temp_data,axis=0)
        if norm:
            for i in range(self.mean.shape[0]):
                data[:,:,i] = (data[:,:,i]-self.mean[i])/self.std[i]
        
        # target has temp and height
        self.trg = torch.Tensor(data[:,1:,-2:])
        self.src = np.zeros((data.shape[0], data.shape[1]-1,data.shape[2]))

        # right shift the set point
        self.src[:,:,0] = data[:,1:,0]
        # trim the last measurement
        self.src[:,:,1:] = data[:,:-1,1:]
        self.src = torch.Tensor(self.src)
        
        # fliter nan values
        self.trg = torch.nan_to_num(self.trg)
        self.src = torch.nan_to_num(self.src)

    def __len__(self):
        return self.trg.shape[0]

    def __getitem__(self, idx):
        return self.src[idx,:,:], self.trg[idx,:,:]

def rmse(e):
    sum = 0
    for err in e:
        sum += err**2

    return np.sqrt(sum/len(e))
    # return sum/len(e)

def test(valid_dataset, nonorm_dataset):
    # load model
    model = torch.load('saved_model_8_next_step.pt')
    model.eval()
    # model.train()
    # load data
    VALID_DATA_DIR = '../data/processed/CL_hot.npy'


    # initialize log-log model
    llmodel = SpeedHeightModel(a=-0.36997977, b=1.21532975)
    # llmodel = SpeedHeightModel()

    seq_len = valid_dataset[0][1].shape[0]
    pred_err_array = np.empty((seq_len,seq_len))
    pred_err_array.fill(np.nan)
    pred_err_array_mod = np.empty((seq_len,seq_len))
    pred_err_array_mod.fill(np.nan)
    ll_pred_err_array = np.zeros(seq_len)
    for start_seg in range(25,seq_len):
        errors = np.zeros((len(valid_dataset),seq_len, 2))
        errors_z = np.zeros((len(valid_dataset)*seq_len, 2))
        errors_ll = np.zeros((len(valid_dataset), seq_len))
        for idx, (src,trg) in enumerate(valid_dataset):
            src_nonorm, trg_nonorm = nonorm_dataset[idx] 
            # lstm error
            # pred_lstm = model(src[:,[0,2]])
            # pred_lstm = model(src[:,:3])
            pred_lstm = model(torch.unsqueeze(src[:,[0,2,3]], dim=0), start_seg)
            # pred_lstm = model(torch.unsqueeze(src[:,0], dim=1))
            # pred_lstm = pred_lstm.squeeze()
            pred_std =pred_lstm.detach()*valid_dataset.std[2:]+valid_dataset.mean[2:]

            error = trg_nonorm-pred_std
            # error = np.reshape(error, (-1,2))
            errors[(idx),:,:] = error

            #z errors
            error_z = trg-(pred_lstm.detach())
            error_z = np.reshape(error_z, (-1,2))
            errors_z[(idx)*seq_len:(idx+1)*seq_len,:] = error_z

            # static model error
            vel = src_nonorm[:,0]
            pred_ll = llmodel.v2dh(vel)

            error_ll = trg_nonorm[:,1]-pred_ll
            errors_ll[idx,:] = error_ll
            # print(f"e{rmse(trg_nonorm[:,1]-pred_std.detach()[:,1])}")

            if idx in [15,50,95]:
                print(trg_nonorm.shape)
                print(pred_std.shape)
                fig,ax = plt.subplots(2,1)
                ax[0].plot(trg_nonorm[:,1])
                ax[0].plot(pred_std[:,1])
                # ax[0].plot(pred_ll)
                # ax[0].plot(trg_nonorm[0,1]-pred_std[0,:,1])
                ax[1].plot(trg_nonorm[:,0])
                ax[1].plot(pred_std[:,0])
                ax[0].set_title(f"Layer {idx}")
                ax[0].set_ylim([-0.5,4])
                ax[0].plot([25,25], [-2,4])
                # ax[1].set_ylim([14000,15750])
                ax[1].set_xlabel("Segment Index")
                ax[0].set_ylabel("dh (mm)")
                ax[1].set_ylabel("Brightness")
                ax[0].legend(["Measured", "Predicted"])
                plt.show()
            print(f'lstm error: {sum(error)/len(error)}')
            print(f'll error:   {sum(error_ll)/len(error_ll)}')

        # print('--------Height Error---------')
        # print(f'LSTM MAE:     {mae(errors[:,1])}')
        # print(f'LSTM RMSE:    {rmse(errors[:,1])}')
        # print(f'LSTM max:     {np.max(errors[:,1])}')
        # print(f'Log-Log RMSE: {rmse(errors_ll)}')
        # print('--------Height Error Z---------')
        # print(f'mean:         {np.mean(errors_z[:,1])}')
        # print(f'std-dev:      {np.std(errors_z[:,1])}')

        # print()
        # print('--------Temp Error---------')
        # print(f'LSTM MAE:     {mae(errors[:,0])}')
        # print(f'LSTM RMSE:    {rmse(errors[:,0])}')
        # print(f'LSTM max:     {np.max(errors[:,0])}')
        # print('--------Temp Error Z---------')
        # print(f'mean:         {np.mean(errors_z[:,0])}')
        # print(f'std-dev:      {np.std(errors_z[:,0])}')


        num_layers = 105
        error_at_pred = np.zeros(seq_len-start_seg)
        for i in range(seq_len-start_seg):
            temp_errors = np.zeros(num_layers)
            for j in range(num_layers):
                temp_errors[j] = errors[j,start_seg+i, 1]
            error_at_pred[i] = rmse(temp_errors)
        # pred_err_array[start_seg, :seq_len-start_seg]= error_at_pred
        pred_err_array[start_seg, start_seg:]= error_at_pred
        pred_err_array_mod[start_seg, :seq_len-start_seg] = error_at_pred
    # calculate log-log error
    for i in range(seq_len):
        ll_pred_err_array[i] = rmse(errors_ll[:,i])

    # calculate minimum
    min_val = np.nanmin(pred_err_array)
    print(min_val)
    # min_val = min(min_val, np.min(ll_pred_err_array))
    max_val = np.nanmax(pred_err_array)
    print(max_val)
    # max_val = max(max_val, np.max(ll_pred_err_array))


    # fig,ax = plt.subplots(2,1, sharex=True)
    # im = ax[0].imshow(pred_err_array, 
    #                   vmin = min_val,
    #                   vmax = max_val,
    #                   aspect='auto')
    # im2 = ax[1].imshow(np.expand_dims(ll_pred_err_array,axis=0),
    #                    vmin = min_val,
    #                    vmax = max_val,
    #                    aspect='auto')
    # ax[1].set_xlabel("Steps Ahead")
    # ax[0].set_ylabel("Prediction Starting Step")
    # ax[0].set_title("Prediction Error, Spaitial")
    # cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    # cbar.set_label("RMSE Error (mm)")
    # plt.show()
    fig,ax = plt.subplots()
    ax.plot(np.nanmean(pred_err_array_mod, axis=0))
    plt.show()
    fig,ax = plt.subplots()
    ax.plot(np.nanmean(pred_err_array_mod, axis=1))
    plt.show()

    fig,ax = plt.subplots()
    im = ax.imshow(pred_err_array, 
                      vmin = min_val,
                      vmax = max_val,
                      )
    ax.set_xlabel("Steps Ahead")
    ax.set_ylabel("Prediction Starting Step")
    ax.set_title("Prediction Error, spatial")
    cbar = fig.colorbar(im)
    cbar.set_label("RMSE Error (mm)")
    plt.show()

    fig,ax = plt.subplots()
    im = ax.imshow(pred_err_array_mod, 
                      vmin = min_val,
                      vmax = max_val,
                      )
    ax.set_xlabel("Steps Ahead")
    ax.set_ylabel("Prediction Starting Step")
    ax.set_title("Prediction Error, steps_ahead")
    cbar = fig.colorbar(im)
    cbar.set_label("RMSE Error (mm)")
    plt.show()

    # analyzing trend of errors

    # plt.hist(errors_z[:,1],bins=40)
    # plt.show()

class SpeedHeightModel:
    """
    Model relating dh to torch speed according to the equation
    ln(h) = a ln(v) + b
    """

    def __init__(self, lam=0.05, beta=0.99, a=-0.4619, b=1.647, p = None):
        # Beta == 1 for non-exponentail updates
        self.coeff_mat = np.array([a, b])
        self.nom_a = a
        self.nom_b = b
        self.lam = lam
        if p is None:
            self.p = np.diag(np.ones(self.coeff_mat.shape[0]) * self.lam)
        else: self.p = p
        self.beta = beta

    def v2dh(self, v):
        """outputs the height for a velocity or array of velocities"""
        logdh = self.coeff_mat[0] * np.log(v) + self.coeff_mat[1]

        dh = np.exp(logdh)
        return dh

    def dh2v(self, dh):
        """outputs the velocity for a height or set of heights"""
        logdh = np.log(dh)
        logv = (logdh - self.coeff_mat[1]) / self.coeff_mat[0]

        v = np.exp(logv)
        return v

    def model_update_rls(self, vels, dhs):
        """updates the model coefficients using the recursive
        least-squares algorithm"""
        # Algorithm from
        # https://osquant.com/papers/recursive-least-squares-linear-regression/
        vels = np.reshape(vels,-1)
        for idx, vel in enumerate(vels):
            x = np.array([[np.log(np.array(vel))], [1]])
            y = np.log(dhs[idx])
            if not np.isnan(y):
                r = 1 + (x.T @ self.p @ x) / self.beta
                k = self.p @ x / (r * self.beta)
                e = y - x.T @ self.coeff_mat
                self.coeff_mat = self.coeff_mat + k @ e
                self.p = self.p / self.beta - k @ k.T * r
def mae(e):
    sum = 0
    for err in e:
        sum+=np.abs(err)
    return sum/len(e)

if __name__== '__main__':
    # load data
    TRAIN_DATA_DIR = '../data/lstm_processed/CL_cold.npy'
    VALID_DATA_DIR = '../data/lstm_processed/CL_hot.npy'

    train_dataset = WeldDataset(TRAIN_DATA_DIR)
    torch.save(train_dataset.mean, 'mean.pt')
    torch.save(train_dataset.std, 'std.pt')
    valid_dataset = WeldDataset(VALID_DATA_DIR, train_dataset.mean, train_dataset.std)
    nonorm_dataset = WeldDataset(VALID_DATA_DIR, norm=False)

    # train model
    train(train_dataset, valid_dataset)
    # test model
    test(valid_dataset, nonorm_dataset)
