import json
import numpy as np
import os
import optuna
import pandas as pd
import torch

# import custom modules
from arg_parser_training import argParser
from bert_model import RegressorBERT

# import modules for defining the transformer model's components and training
from functools import partial
from math import ceil
from transformers import BertTokenizer, PreTrainedTokenizer, BertConfig
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup, set_seed
from torch.optim import AdamW
from torch.nn.functional import mse_loss, l1_loss
from auxiliary_functions import rmse_loss

# import modules for data processing and storage
from auxiliary_functions import StandardScaler, Dataset, tokenize, compute_metrics, customTrainer
from datasets import DatasetDict

def create_dataset_valid_split(pandas_df: pd.DataFrame, limit_size: int, tokenizer: PreTrainedTokenizer, z_threhsold: float) -> Dataset:
    # eliminate outliers with z-scores and scale
    scaler = StandardScaler()
    pandas_df = pandas_df.iloc[:limit_size]
    prices = pandas_df["price"].to_numpy()
    z_scores = (prices-np.mean(prices))/np.std(prices)
    pandas_df = pandas_df.iloc[z_scores < z_threhsold]
    pandas_df["price"] = np.asarray(scaler.fit_transform(pandas_df["price"].to_numpy().reshape(-1,1)).tolist())

    dataset = Dataset.from_pandas(pandas_df)
    dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.remove_columns("descrip")

    dataset = dataset.train_test_split(test_size=0.4, seed=42)
    dataset_test = dataset["test"].train_test_split(test_size=0.5, seed=42)
    dataset = DatasetDict({
        "train": dataset["train"],
        "test": dataset_test["train"],
        "validation": dataset_test["test"],
    })

    print(dataset.shape)

    return dataset

loss_name_to_func = {
    "rmse": rmse_loss,
    "mse": mse_loss,
    "l1": l1_loss
}

optimizer_name_to_func = {
    "adamw": AdamW
}

def objective(trial: optuna.Trial, tokenizer, args, dataset):
    #### create custom layers on top of BERT
    torch.manual_seed(42)
    set_seed(42)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []
    embd_size = 768
    in_features = embd_size
    for idx_num_layers in range(n_layers):
        units = trial.suggest_categorical(f"size_layers_{idx_num_layers}", [128, 256, 512, 1024])
        layers += [torch.nn.Linear(in_features, units)]
        layers += [torch.nn.BatchNorm1d(units)]
        layers += [torch.nn.GELU()]
        in_features = units

    layers += [torch.nn.Linear(in_features, 1)]
    model_reg = torch.nn.Sequential(*layers).to(device)
    ##########################################

    #### search for freezing parts/all of the parameters, and how many hidden layers to use
    freeze_paras = trial.suggest_categorical("freezing_parameters", ["all", "all_but_last_1", "all_but_last_2"])
    hidden = trial.suggest_int("hidden_layers", low=1, high=3)
    hidden_list = [-hidden_id for hidden_id in range(1,hidden+1)]
    ##########################################

    model_name = "google-bert/bert-base-multilingual-cased"
    model = RegressorBERT(model_name=model_name,
                          freeze_bert=freeze_paras, 
                          hidden=hidden_list,
                          loss_func=loss_name_to_func[args.loss_func],
                          aggregation_method=args.agg,
                          regression_head=model_reg,
                          config=BertConfig.from_pretrained(model_name),
                          load_finetuned=False,
                          return_hidden=False
                          ).to(device)

    training_args = TrainingArguments(output_dir="regressor_bert_hyper", 
                    num_train_epochs=args.epochs, 
                    eval_strategy="epoch", 
                    save_strategy="epoch", 
                    push_to_hub=False,
                    remove_unused_columns=True,
                    label_names=["price"],
                    report_to="tensorboard",
                    per_device_train_batch_size=args.batches,
                    per_device_eval_batch_size=args.batches,
                    bf16=True,
                    disable_tqdm=True,
                    load_best_model_at_end=True,
                    save_total_limit=1
                    )
    
    #### search for best optimizer parameters
    learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=5e-2)
    w_decay = trial.suggest_float("weight_decay", low=1e-4, high=1e-1)
    num_training_steps = ceil(len(dataset["train"])/args.batches * args.epochs)
    optimizer = optimizer_name_to_func[args.optimizer](model.parameters(), lr=learning_rate, weight_decay=w_decay)
    optimizer_scheduler = (optimizer,
                        get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps))
    ##########################################
    
    trainer = customTrainer(tokenizer=tokenizer,
                    model=model,
                    optimizers=optimizer_scheduler,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["validation"],
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
                    )
    
    results = trainer.train()
    path = os.path.join("trial_runs", trial.study.study_name, str(trial.number))

    # save all parameters/results for manual validation
    eval_results = trainer.evaluate(dataset["validation"])
    os.makedirs(path, exist_ok=True)
    json.dump({**results.metrics, **eval_results, "trial_values": trial.params}, open(path + "/" + "metrics_architecture.json", "w"), indent=2)

    del model, optimizer # delete objects for better memory usage

    return eval_results["eval_loss"]

if __name__ == "__main__":
    arg_parser = argParser()
    args = arg_parser.arg_parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model_name = "google-bert/bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    
    df = pd.read_csv("data/funda_data_11_06_2024.csv")
    df = df[["descrip", "price"]]
    dataset = create_dataset_valid_split(df, args.datasize, tokenizer, 3)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #### hyperparameter search with Optuna
    study_name = args.study_name
    study = optuna.create_study(study_name=study_name, direction="minimize", storage="sqlite:///HPO.sqlite3", load_if_exists=True)
    obj = partial(objective, tokenizer=tokenizer, args=args, dataset=dataset)
    study.optimize(func=obj, n_trials=args.num_trials)