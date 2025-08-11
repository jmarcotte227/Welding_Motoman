from torch import nn
import torch

class WeldLSTM(nn.Module):
    def __init__(self,
                 input_size = 6,
                 hidden_size = 1024,
                 output_size = 1,
                 num_layers = 1,
dropout = 0
                 ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout = dropout
                          )
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=output_size)

    def forward(self, src, start_step = 0, stop_seq = False, hidden_state = None):
        if self.training:
            output, state = self.lstm(src)
            output = self.linear(output)

        elif hidden_state is not None:
            output, state = self.lstm(src, hidden_state)
            output = self.linear(output)

            return output, state

        else:
            # should be testing
            # one batch at a time
            # loop through sequence
            output = torch.zeros((src.shape[-2], self.output_size))
            # print("Working Output")
            # print(output.shape)
            state = None
            # check if iteration starts after the firs step
            # allows lookahead while maintaining previous use
            if start_step > 0:
                temp_out, state = self.lstm(src[:,:start_step])
                # print(state)
                state = (state[0][0,:,:], state[1][0,:,:])
                # print(state)
                # print("temp_out: ",temp_out.shape)
                output[:start_step,:] = self.linear(temp_out)

            # exits the sequence and returns the state of the filter
            if stop_seq:
                output = output[:start_step,:]
                return output, state

            for idx, val in enumerate(src[0][start_step:]):
                # print("val: ", val.shape)
                val_temp = val.unsqueeze(0)
                if state is None:
                    out, state = self.lstm(val_temp)
                    output[idx,:] = self.linear(out)
                else:
                    # construct input from previous output
                    _input = torch.Tensor([[val_temp[:,0],
                                            output[idx+start_step-1,0],
                                            output[idx+start_step-1,1]]])
                    out, state = self.lstm(_input, state)
                    output[idx+start_step,:] = self.linear(out)

        return output

