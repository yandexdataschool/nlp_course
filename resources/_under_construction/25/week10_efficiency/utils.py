import torch
from torch import nn, Tensor


class Catcher(nn.Module):
    def __init__(self):
        super().__init__()

        # We know that LLaMA layers take a Tensor of hidden states,
        # and some kwargs of which attention_mask and position_ids
        # are required. attention_mask and position_ids are also the
        # same for the entire dataset, so we only have to register the
        # last ones
        self.hidden_states = []
        self.attention_mask = None
        self.position_ids = None


    def forward(self, hidden_states, **kwargs):
        assert hidden_states.shape[0] == 1 # only one element from dataset
        self.hidden_states.append(hidden_states[0])
        self.attention_mask = kwargs['attention_mask']
        self.position_ids = kwargs['position_ids']
        raise ValueError

    def get_the_catch(self):
        return torch.stack(self.hidden_states), self.attention_mask, self.position_ids


def get_first_layer_inputs(model: nn.Module, model_inputs: Tensor):
    catcher = Catcher()
    original_layers = model.model.layers

    model.model.layers = nn.ModuleList((catcher,))
    for sample in model_inputs:
        try:
            model(sample.unsqueeze(0))
        except ValueError:
            pass
    model.model.layers = original_layers

    return catcher.get_the_catch()
