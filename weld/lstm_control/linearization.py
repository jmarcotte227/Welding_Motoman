import sys
import torch
from torch import tanh, sigmoid, diag, square
import matplotlib.pyplot as plt

# load internal packages
from lstm_model_next_step import WeldLSTM
from lstm_train_next_step import WeldDataset


def tanh_p(x):
    # use for vector x
    return diag(1-square(tanh(x)))

def sigmoid_p(x):
    return diag(sigmoid(x)*(1-sigmoid(x)))

def lstm_linearization(model, h_0, c_0, u_0):
    # takes the model, hidden state opperating points and input operating
    # point and produces the linearized system matrices. 
    h_dim = h_0.shape[0]
    
    # model weights
    # TODO: needed to add '0' to the end of the variable name. not in the documentation
    W_hi = model.lstm.weight_hh_l0[0*h_dim:1*h_dim,:].T
    W_hf = model.lstm.weight_hh_l0[1*h_dim:2*h_dim,:].T
    W_hc = model.lstm.weight_hh_l0[2*h_dim:3*h_dim,:].T
    W_ho = model.lstm.weight_hh_l0[3*h_dim:4*h_dim,:].T

    W_ui = model.lstm.weight_ih_l0[0*h_dim:1*h_dim,:].T
    W_uf = model.lstm.weight_ih_l0[1*h_dim:2*h_dim,:].T
    W_uc = model.lstm.weight_ih_l0[2*h_dim:3*h_dim,:].T
    W_uo = model.lstm.weight_ih_l0[3*h_dim:4*h_dim,:].T

    W_y = model.linear.weight

    # model biases
    b = model.lstm.bias_ih_l0+model.lstm.bias_hh_l0
    b_i = b[0*h_dim:1*h_dim]
    b_f = b[1*h_dim:2*h_dim]
    b_c = b[2*h_dim:3*h_dim]
    b_o = b[3*h_dim:4*h_dim]

    f = sigmoid(W_hf.T@h_0+W_uf.T@u_0+b_f)
    i = sigmoid(W_hi.T@h_0+W_ui.T@u_0+b_i)
    tc = tanh(W_hc.T@h_0+W_uc.T@u_0+b_c)
    o = sigmoid(W_ho.T@h_0+W_uo.T@u_0+b_o)
    c = f*c_0+i*tc

    ##### Compute A_h #####
    # df_dh = W_hf@sigmoid_p(W_hf.T@h_0+W_uf.T@u_0+b_f)
    # di_dh = W_hi@sigmoid_p(W_hi.T@h_0+W_ui.T@u_0+b_i)
    # dtc_dh = W_hc@tanh_p(W_hc.T@h_0+W_uc.T@u_0+b_c)

    # dc_dh = df_dh@diag(c_0)+di_dh@diag(tc)+dtc_dh@diag(i)
    # do_dh = dc_dh@tanh_p(c)

    # dtanhc_dh = dc_dh@tanh_p(c)

    # dh_dh = do_dh@diag(tanh(c))+dtanhc_dh@diag(o)
    # A_h = dh_dh.T
    A_h = None

    # ##### Compute B_h #####
    # df_du = W_uf@sigmoid_p(W_hf.T@h_0+W_uf.T@u_0+b_f)
    # di_du = W_ui@sigmoid_p(W_hi.T@h_0+W_ui.T@u_0+b_i)
    # dtc_du = W_uc@tanh_p(W_hc.T@h_0+W_uc.T@u_0+b_c)

    # dc_du = df_du@diag(c_0)+di_du@diag(tc)+dtc_du@diag(i)
    # do_du = W_uo@sigmoid_p(W_ho.T@h_0+W_uo.T@u_0+b_o)

    # dtanhc_du = dc_du@tanh_p(c)

    # dh_du = do_du@diag(tanh(c))+dtanhc_du@diag(o)
    # B_h = dh_du.T

    ##### Compute B_h fast #####
    # df_du = W_uf@sigmoid_p(W_hf.T@h_0+W_uf.T@u_0+b_f)
    # di_du = W_ui@sigmoid_p(W_hi.T@h_0+W_ui.T@u_0+b_i)
    # dtc_du = W_uc@tanh_p(W_hc.T@h_0+W_uc.T@u_0+b_c)

    # dc_du = df_du@diag(c_0)+di_du@diag(tc)+dtc_du@diag(i)
    # do_du = W_uo@sigmoid_p(W_ho.T@h_0+W_uo.T@u_0+b_o)

    # dtanhc_du = dc_du@tanh_p(c)

    # dh_du = do_du@diag(tanh(c))+dtanhc_du@diag(o)
    # B_h = dh_du.T

    ##### Compute B_h one line #####
    dh_du = (W_uo@sigmoid_p(W_ho.T@h_0+W_uo.T@u_0+b_o))@diag(tanh(c))+(((W_uf@sigmoid_p(W_hf.T@h_0+W_uf.T@u_0+b_f))@diag(c_0)+(W_ui@sigmoid_p(W_hi.T@h_0+W_ui.T@u_0+b_i))@diag(tc)+(W_uc@tanh_p(W_hc.T@h_0+W_uc.T@u_0+b_c))@diag(i))@tanh_p(c))@diag(o)
    B_h = dh_du.T

    # print("Separated: ", B_h)
    # B_h = (W_uo@sigmoid_p(W_ho.T@h_0+W_uo.T@u_0+b_o)@diag(tanh(c))+(W_uf@sigmoid_p(W_hf.T@h_0+W_uf.T@u_0+b_f)@diag(c_0)+W_ui@sigmoid_p(W_hi.T@h_0+W_ui.T@u_0+b_i)@diag(tc)+W_uc@tanh_p(W_hc.T@h_0+W_uc.T@u_0+b_c)@diag(i))@tanh_p(c)@diag(o)).T
    # print("Together: ", B_h)

    C = W_y

    return A_h, B_h, C

if __name__=="__main__":
    # x = torch.Tensor([0,1,2,3])

    # print("Tanh(x): ", tanh(x))
    # print("Tanh_p(x): ", tanh_p(x))
    # print("sigmoid(x): ", sigmoid(x))
    # print("sigmoid_p(x): ", sigmoid_p(x))

    # load data
    TRAIN_DATA_DIR = '../data/lstm_processed/CL_cold.npy'
    VALID_DATA_DIR = '../data/lstm_processed/CL_hot.npy'

    train_dataset = WeldDataset(TRAIN_DATA_DIR)
    valid_dataset = WeldDataset(VALID_DATA_DIR, train_dataset.mean, train_dataset.std)
    nonorm_dataset = WeldDataset(VALID_DATA_DIR, norm=False)

    # load model
    model = torch.load('../multi_output/saved_model_8_next_step.pt')
    model.eval()

    seq_num = 45
    v_min = 3.0
    v_max = 17.0
    # convert limits
    v_min = torch.tensor([(v_min-train_dataset.mean[0])/train_dataset.std[0]], dtype=torch.float32)
    v_max = torch.tensor([(v_max-train_dataset.mean[0])/train_dataset.std[0]], dtype=torch.float32)
    while True:
        seq_num = int(input("Enter layer number: "))

        src, trg = valid_dataset[seq_num]
        src_nonorm, trg_nonorm = nonorm_dataset[seq_num]

        h_pred = []
        h_pred_lin = []
        h_act = []
        v_set = []
        v_set_coeff = []
        v_plan_lin = []

        # warm up model with first 10 steps
        for start_seg in range(1,45):
            pred_lstm, state = model(torch.unsqueeze(src[:,[0,2,3]], dim=0),
                                     start_seg,
                                     stop_seq = True)

            # define operating point
            h_0 = torch.squeeze(state[0])
            c_0 = torch.squeeze(state[1])
            u_0 = torch.squeeze(src[start_seg, [0,2,3]])
            u_0[0] = src[start_seg-1,0]
            y_0, _ = model(torch.unsqueeze(u_0, dim=0), 
                                hidden_state=state)
            y_0 = torch.squeeze(y_0)

            A_h, B_h, C = lstm_linearization(model, h_0, c_0, u_0)

            # dv_set = 0.5

            # du = torch.tensor([(dv_set)/valid_dataset.std[0],
            #                    trg[start_seg-1,0]-u_0[1],
            #                    trg[start_seg-1, 1]-u_0[2]])
            du = torch.tensor([src[start_seg,0]-u_0[0],
                               trg[start_seg-1,0]-u_0[1],
                               trg[start_seg-1, 1]-u_0[2]])

            # linearized output
            # TODO: make sure I don't need the A matrix. Since I'm linearizing about the previous hidden state, 
            #       dh_k-1 goes to 0?
            dy = C@B_h@du
            print(dy.shape)

            out_est = y_0+dy
            out_est_reg = out_est.detach()*valid_dataset.std[[2,3]]+valid_dataset.mean[[2,3]]
            print(out_est_reg)
            h_pred_lin.append(out_est_reg[1].detach())

            # compute next prediction of LSTM
            out,_ = model.lstm(torch.unsqueeze(src[start_seg, [0,2,3]], dim=0), state)
            out = model.linear(out)
            out = out.detach()*valid_dataset.std[[2,3]]+valid_dataset.mean[[2,3]]
            h_pred.append(out[:,1])
            h_act.append(trg_nonorm[start_seg,1])

            v_set.append(src[start_seg, 0].detach()*valid_dataset.std[0]+valid_dataset.mean[0])

            v_set_coeff.append((C@B_h)[1,0].detach())

            dh_des = trg[start_seg,1]

            
            # isolate the effect of the velocity on the height input
            B_h = B_h[:,0]
            C = C[1,:]
            print(B_h.shape)
            print(C.shape)

            v_plan = (dh_des-y_0[1])/(C@B_h)+u_0[0]
            print(v_plan.shape)
            v_plan = min(max(v_plan, v_min), v_max)
            print(v_plan)
            v_plan_lin.append(v_plan.detach()*valid_dataset.std[0]+valid_dataset.mean[0])

        print(type(h_pred))
        fig,ax = plt.subplots(2,1, sharex = True)
        ax[0].plot(h_act)
        ax[0].plot(h_pred_lin)
        ax[0].plot(h_pred)
        fig.suptitle(f"Layer {seq_num}")
        ax[0].legend([
                "Measured",
                "Linearized LSTM Prediction",
                "LSTM Prediction",
            ])
        ax[0].set_ylabel("dh (mm)")
        ax[1].plot(v_set)
        ax[1].plot(v_plan_lin)
        ax[1].legend(["v_set", "v_plan linear"])
        ax[1].set_ylabel("V_set (mm/s)")
        ax[1].set_xlabel("Segment Index")
        plt.show()

        # v prediction
        fig, ax = plt.subplots()
        ax.set_title(f"Layer {seq_num} Velocity Calculation")
        ax.set_ylabel("V_set (mm/s)")
        ax.set_xlabel("Segment Index")
        ax.plot(v_set)
        ax.plot(v_plan_lin)
        ax.legend([
            "v_set actual",
            "v_set linear prediction"
            ])

        fig,ax = plt.subplots()
        fig.suptitle("v_set Coefficient")
        ax.plot(v_set_coeff)
        ax.set_ylabel("Coefficient Value")
        ax.set_xlabel("Prediction Step")
        plt.show()
        print("dy: ", dy.detach()*valid_dataset.std[[2,3]])
        print(out_est.detach()*valid_dataset.std[[2,3]]+valid_dataset.mean[[2,3]])
