import torch
import numpy as np
import pandas as pd

from transformers import Trainer
from transformers import PreTrainedTokenizer
from torch.nn.functional import mse_loss, l1_loss
from datasets import Dataset
from sklearn.preprocessing import StandardScaler

class customTrainer(Trainer):
    def log(self, logs: dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        logs["regression_head"] = str(self.model.regression_head)
        super().log(logs)

def compute_metrics(pred):
    # value=REAL, logits=PRED
    logits, value = pred
    logits, value = torch.as_tensor(logits), torch.as_tensor(value)
    mse = mse_loss(value, logits)
    rmse = torch.sqrt(mse)
    l1 = l1_loss(value, logits)
    return {"rmse": rmse, "l1": l1, "mse": mse}

def rmse_loss(input, target):
    loss = mse_loss(input, target)
    return torch.sqrt(loss)

def tokenize(x: str, tokenizer: PreTrainedTokenizer) -> dict:
    return tokenizer(x["descrip"], max_length=512, truncation=True)

def create_dataset(pandas_df: pd.DataFrame, limit_size: int, tokenizer: PreTrainedTokenizer, z_threhsold: float) -> Dataset:
    # eliminate outliers with z-scores < 3 and scale
    scaler = StandardScaler()
    pandas_df = pandas_df.dropna()
    pandas_df = pandas_df.iloc[:limit_size]
    prices = pandas_df["price"].to_numpy()
    z_scores = (prices-np.mean(prices))/np.std(prices)
    pandas_df = pandas_df.iloc[z_scores < z_threhsold]
    pandas_df["price"] = np.asarray(scaler.fit_transform(pandas_df["price"].to_numpy().reshape(-1,1)).tolist())

    dataset = Dataset.from_pandas(pandas_df)
    dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.remove_columns("descrip")

    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print(dataset.shape)

    return dataset