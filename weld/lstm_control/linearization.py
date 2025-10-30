import sys
import torch
from torch import tanh, sigmoid, diag, square
import matplotlib.pyplot as plt

# load internal packages
sys.path.append("../multi_output/")

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
