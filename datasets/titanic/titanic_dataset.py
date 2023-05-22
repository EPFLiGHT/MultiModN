import os
import numpy as np
from typing import Optional, List
from datasets import PartitionDataset, FeatureWiseDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch import Tensor
import pandas as pd
from torch._utils import _accumulate as accumulate

DATA_ABS_PATH = os.path.dirname(os.path.realpath(__file__))

class TitanicDataset(Dataset):
    def __init__(
        self,
        features: List[str],
        targets: List[str],
        dropna: bool = True,
        dropna_columns: List[str] = [],
        std: bool = True,
    ):
        data_path = os.path.join(DATA_ABS_PATH, "../../data/titanic/titanic.csv")

        df = pd.read_csv(data_path).set_index('PassengerId')
        df['id'] = df.index
        aug_df = titanic_preprocessing(df)
        aug_df = aug_df[list(set(features + targets + dropna_columns))]

        if dropna:
            aug_df.dropna(inplace=True)
        
        aug_df = aug_df[features + targets]

        if std:
            aug_std_df = pd.DataFrame(StandardScaler().fit_transform(aug_df[features]), columns=features, index=aug_df.index)
            aug_std_df[targets] = aug_df[targets]
            aug_df = aug_std_df
        
        self.X = aug_df[features].values
        self.y = aug_df[targets].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return Tensor(self.X[idx]), self.y[idx]

    def partition_dataset(self, partitions: Optional[List[int]] = None) -> PartitionDataset:
        return PartitionDataset(self.X, self.y, partitions)

    def featurewise_dataset(self) -> FeatureWiseDataset:
        return FeatureWiseDataset(self.X, self.y)
    
    def split_dataset(self, partitions: Optional[List[int]] = None) -> List[PartitionDataset]:
        if partitions is None:
            partitions = [self.X.shape[1]]
        
        if sum(partitions) != self.X.shape[1]:
            raise ValueError(
                "Paritions sum doesn't match data dimension. Expected: {}, got: {}"
                .format(sum(partitions), self.X.shape[1])
            )

        partition_offsets = list(accumulate(partitions[:-1]))
        X_split = np.split(self.X, partition_offsets, axis=1)

        return [PartitionDataset(X_split[i], self.y, [partition]) for i, partition in enumerate(partitions)]

def titanic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    aug_df = df.copy()
    aug_df["Relatives"] = aug_df["SibSp"] + aug_df["Parch"]
    aug_df = pd.get_dummies(aug_df, columns=['Sex'], drop_first=True)

    cabin_mapping = list(enumerate(sorted(aug_df['Cabin'].dropna().unique())))
    cabin_mapping = [(b, a) for (a, b) in cabin_mapping]
    aug_df['Cabin_num'] = aug_df['Cabin'].map(dict(cabin_mapping))

    aug_df['Embarked'] = aug_df['Embarked'].map(dict(zip(['S','C','Q'],[0,1,2])))

    return aug_df