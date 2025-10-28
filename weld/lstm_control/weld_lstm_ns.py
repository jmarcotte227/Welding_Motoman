from torch import nn
import torch

class WeldLSTMNextStep(nn.Module):
    """
    An LSTM to be used with welding data. This model is used for next
    step prediction, only taking a single un-batched input and producing
    the output and hidden state.

    Args:
        input_size (int): The number of elements in the input.
        hidden_size (int): The size of the LSTM hidden state.
        output_size (int): The number of elements output from the 
            linear layer.
        num_layers (int): The number of LSTM layers before the 
            output is measured.
        dropout (float): The dropout probability for connections
            when more than 1 LSTM layer is used.

    Shape:
        - Input:  (input_size)
        - Output: (output_size)
    """

    def __init__(self,
                 input_size = 3,
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

    def forward(self, src, hidden_state = None):
        output, state = self.lstm(src, hidden_state)
        output = torch.squeeze(self.linear(output))

        return output, state
