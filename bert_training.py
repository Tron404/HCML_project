import pandas as pd
import pickle
import torch

# import custom modules
from arg_parser_training import argParser
from bert_model import RegressorBERT

# import modules for defining the transformer model's components and training
from math import ceil
from transformers import BertTokenizer, BertConfig
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup, set_seed
from torch.optim import AdamW
from torch.nn.functional import mse_loss, l1_loss
from auxiliary_functions import rmse_loss
from torch import nn

# import modules for data processing and storage
from auxiliary_functions import *

# custom trainer to keep track of learning rate
class customTrainer(Trainer):
    def log(self, logs: dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)

loss_name_to_func = {
    "rmse": rmse_loss,
    "mse": mse_loss,
    "l1": l1_loss
}

optimizer_name_to_func = {
    "adamw": AdamW
}

if __name__ == "__main__":
    arg_parser = argParser()
    args = arg_parser.arg_parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model_name = "google-bert/bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name, return_tensors="pt", truncation=True, max_length=512, padding="max_length")

    # set seeds for consistent results using the same base model
    torch.manual_seed(42)
    set_seed(42)

    # model obtained after hyperparameter search
    regression_head = nn.Sequential(
        nn.Linear(768, 256, bias=True), nn.BatchNorm1d(256), nn.GELU(),
        nn.Linear(256, 256, bias=True), nn.BatchNorm1d(256), nn.GELU(),
        nn.Linear(256, 1024, bias=True), nn.BatchNorm1d(1024), nn.GELU(),
        nn.Linear(1024,1,bias=True)
    )

    model = RegressorBERT(model_name=model_name,
                          freeze_bert=args.freeze,
                          hidden=args.hidden,
                          loss_func=loss_name_to_func[args.loss_func],
                          aggregation_method=args.agg,
                          regression_head=regression_head,
                          config=BertConfig.from_pretrained(model_name),
                          load_finetuned=False,
                          return_hidden=False
                          ).to(device)
    
    df = pd.read_csv("data/funda_data_11_06_2024.csv")
    df = df[["descrip", "price"]]
    dataset = create_dataset(df, args.datasize, tokenizer, 3)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(output_dir="regressor_bert_best", 
                                    num_train_epochs=args.epochs, 
                                    eval_strategy="epoch", 
                                    save_strategy="epoch", 
                                    push_to_hub=False,
                                    remove_unused_columns=True,
                                    label_names=["price"],
                                    report_to="tensorboard",
                                    per_device_train_batch_size=args.batches,
                                    per_device_eval_batch_size=args.batches,
                                    metric_for_best_model="eval_loss",
                                    bf16=True,
                                    disable_tqdm=True,
                                    load_best_model_at_end=True,
                                    save_total_limit=1
                                    )

    num_training_steps = ceil(len(dataset["train"])/args.batches * args.epochs)
    optimizer = optimizer_name_to_func[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    optimizer_scheduler = (optimizer,
                        get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps))
    trainer = customTrainer(model=model,
                    tokenizer=tokenizer,
                    optimizers=optimizer_scheduler,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["test"],
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
                    )

    final_eval_results = trainer.evaluate(dataset["test"])
    
    trainer.train()

    model.save_pretrained("regressor_bert_final")
    pickle.dump(model, open("model.pickle", "wb"))