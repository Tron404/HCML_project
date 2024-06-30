import torch

from transformers import BertModel, PreTrainedModel, BertConfig

class RegressorBERT(PreTrainedModel):
    config_class = BertConfig
    def __init__(self,
                 config,
                 **kwargs 
                 ) -> None:
        
        super().__init__(config)

        if kwargs["load_finetuned"]:
            self._load_bert(kwargs["model_name"])

        # set up internal attributes
        self.aggregation_method = kwargs["aggregation_method"]
        self.loss_func = kwargs["loss_func"]
        self.hidden_layers = kwargs["hidden"]
        self.return_hidden = kwargs["return_hidden"]

        # load up given BERT model and freeze all or specific layers
        self._load_bert(kwargs["model_name"])
        self._freeze_bert_layers(kwargs["freeze_bert"])

        self.regression_head = kwargs["regression_head"]

    def _freeze_bert_layers(self, freeze_bert: str) -> None:
        match freeze_bert:
            case "all":
                bert_layers = self.bert.parameters()
            case "all_but_last_1":
                bert_layers = list(self.bert.parameters())[:-1]
            case "all_but_last_2":
                bert_layers = list(self.bert.parameters())[:-2]
            case "none":
                bert_layers = []
                
        for parameter in bert_layers:
            parameter.requires_grad = False

    def _load_bert(self, model_name: str) -> None:
        configuration = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(model_name, config=configuration)

    def _pool_tensors(self, x: torch.Tensor) -> torch.Tensor:
        # collapse layer dimension
        match self.aggregation_method:
            case "sum":
                x = torch.sum(x, dim=0)
            case "mean":
                x = torch.mean(x, dim=0)

        return torch.mean(x, dim=1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, **kwargs):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = bert_output.hidden_states
        # resulting shape -> (num_layers, input_size, seq_len, emb_dim)
        output = torch.stack([output[layer] for layer in self.hidden_layers], dim=0)
        # resulting shape -> (input_size, emb_dim)
        output = self._pool_tensors(output)
        output = self.regression_head(output)

        # compute loss
        price = kwargs["price"]
        if isinstance(price, torch.Tensor):
            price = price.unsqueeze(1).type(torch.float32)
        else:
            price = torch.as_tensor(price, dtype=torch.float32).unsqueeze(1)
            
        loss = self.loss_func(output, price)

        if self.return_hidden:
            return_items = {"loss": loss, "logits": output.flatten(), "bert_raw_output": bert_output}
        else:
            del bert_output
            return_items = {"loss": loss, "logits": output.flatten()}

        return return_items