import os
import pandas as pd
import torch

os.environ["CUDA_VISIBLE_DEVICES"]=""
from arg_parser_training import argParser
from bert_model import RegressorBERT
from datasets import Dataset
from transformers import BertTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from torch.nn.functional import mse_loss
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
def tokenize(x: str) -> dict:
    return tokenizer(x["descrip"], max_length=512, truncation=True)

def rmse_loss(input, target):
    loss = mse_loss(input, target)
    return torch.sqrt(loss)

def test_model(sample_text: str, sample_price: int) -> dict:
    tokens = tokenizer(sample_text, truncation=True, max_length=512, padding="longest", return_tensors="pt")
    outputs = model(**tokens, price=[sample_price])

    return outputs

def create_dataset(pandas_df: pd.DataFrame, limit_size: int) -> Dataset:
    dataset = Dataset.from_pandas(pandas_df[:limit_size])
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(0.2, seed=42)
    dataset = dataset.remove_columns("descrip")

    return dataset

loss_name_to_func = {
    "rmse": rmse_loss
}

optimizer_name_to_func = {
    "adamw": AdamW
}

if __name__ == "__main__":
    torch.manual_seed(42)
    arg_parser = argParser()
    args = arg_parser.arg_parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model_name = "google-bert/bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    model = RegressorBERT(model_name=model_name, hidden_layers=args.hidden, device=device, loss_func=loss_name_to_func[args.loss_func], aggregation_method=args.agg, dense_layers=args.dense)
    
    # print(model)

    df = pd.read_csv("funda_data_11_06_2024.csv")
    df = df[["descrip", "price"]]
    dataset = create_dataset(df, args.datasize)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)

    training_args = TrainingArguments(output_dir="regressor_bert", 
                                    # learning_rate=args.lr, 
                                    num_train_epochs=args.epochs, 
                                    # weight_decay=args.w_decay, 
                                    eval_strategy="epoch", 
                                    save_strategy="no", 
                                    # load_best_model_at_end=True, 
                                    push_to_hub=False,
                                    remove_unused_columns=True,
                                    label_names=["price"]
                                    )

    optimizer = optimizer_name_to_func[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    optimizer_scheduler = (optimizer,
                           get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epochs*args.batches))
    trainer = Trainer(model=model,
                    tokenizer=tokenizer,
                    optimizers=optimizer_scheduler,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["test"],
                    #   compute_metrics=compute_metrics
                    )
    trainer.train()
    trainer.save_model()
    
    print(test_model(df.iloc[0]["descrip"], df.iloc[0]["price"])  )