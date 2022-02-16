import torch

def avg_pooling(hidden_state, attention_mask):
    attention_hidden_state = hidden_state * attention_mask.unsqueeze(dim=-1)
    sum_hidden_state = torch.sum(attention_hidden_state, dim=1)
    n = torch.sum(attention_mask.unsqueeze(dim=-1), dim=1)
    avg_hidden_state = sum_hidden_state / n
    return avg_hidden_state



def max_pooling(hidden_state, attention_mask):
    attention_mask = attention_mask.bool()

    x = torch.where(~attention_mask, torch.inf, (~attention_mask).double(), )

    #Add inf to the masked states
    attention_hidden_state = hidden_state - x.unsqueeze(dim=-1)

    max_hidden_state = torch.max(attention_hidden_state, dim=1)

    return max_hidden_state.values.float()

