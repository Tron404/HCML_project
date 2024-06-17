import torch
import torch.nn as nn

from transformers import BertModel
from transformers import BertConfig
from typing import Literal, Callable

class RegressorBERT(torch.nn.Module):
    def __init__(self, 
                 model_name: str, 
                 loss_func: Callable, 
                 aggregation_method: Literal["sum", "mean"] = "mean", 
                 hidden_layers:list = [-1], 
                 device: torch.device=torch.device("cpu"), 
                 freeze_bert: bool|list = True, 
                 dense_layers:list = []) -> None:
        
        super(RegressorBERT, self).__init__()

        # set up internal attributes
        self.aggregation_method = aggregation_method
        self.hidden_layers = hidden_layers
        self.device = device
        self.loss_func = loss_func

        # load up given BERT model and freeze all or specific layers
        self._load_bert(model_name)
        if freeze_bert:
            self._freeze_bert_layers(freeze_bert)

        # creaete chain of dense and ReLU layers based on given list
        dense_layers = [layer for layer_size in dense_layers for layer in (nn.Linear(*layer_size, device=self.device), nn.ReLU())]

        # 1 guaranteed dense layer between BERT and output layer 
        self.regressor_layers = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 128, device=self.device),
            torch.nn.ReLU(),
            *dense_layers,
            torch.nn.Linear(128, 1, device=self.device)
        )

    # TODO freeze specific layers
    def _freeze_bert_layers(self, freeze_bert: bool|list) -> None:
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

    def _load_bert(self, model_name: str) -> None:
        configuration = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(model_name, config=configuration).to(self.device)

    def _pool_tensors(self, x: torch.Tensor) -> torch.Tensor:
        # collapse layer dimension
        match self.aggregation_method:
            case "sum":
                x = torch.sum(x, dim=0)
            case "mean":
                x = torch.mean(x, dim=0)

        return torch.mean(x, dim=1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, **kwargs):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).hidden_states
        # resulting shape -> (num_layers, input_size, seq_len, emb_dim)
        output = torch.stack([output[layer] for layer in self.hidden_layers], dim=0)
        # resulting shape -> (input_size, emb_dim)
        output = self._pool_tensors(output)
        output = self.regressor_layers(output)

        # compute loss
        price = kwargs["price"]
        if isinstance(price, torch.Tensor):
            price = price.unsqueeze(1).type(torch.float32)
        else:
            price = torch.as_tensor(price, dtype=torch.float32).unsqueeze(1)
            
        loss = self.loss_func(output, price)

        return {"loss": loss, "logits": output}