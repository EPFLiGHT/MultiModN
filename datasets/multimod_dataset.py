from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union
from torch.utils.data import Dataset, Subset
import torch
from torch import Generator, Tensor
from torch._utils import _accumulate as accumulate
import numpy as np


class MultiModDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    def random_split(
            self,
            probabilities: Union[List[float], Tuple[float, ...]],
            seed: int,
            balanced_target_idx: Optional[int] = None,
    ) -> List[Subset]:
        generator = Generator().manual_seed(seed)

        sum_p = sum(probabilities)

        shuffled_indices = torch.randperm(len(self), generator=generator).tolist()

        indices_by_class_value = {}
        if balanced_target_idx is None:
            indices_by_class_value = {"Unbalanced": shuffled_indices}
        else:
            indices_by_class_value = {}
            for idx in shuffled_indices:
                target_value = self[idx][1][balanced_target_idx]

                if target_value in indices_by_class_value:
                    indices_by_class_value[target_value].append(idx)
                else:
                    indices_by_class_value[target_value] = [idx]

        splitted_indices = [[]] * len(probabilities)

        for indices in indices_by_class_value.values():
            lengths = [int(len(indices) * p / sum_p) for p in probabilities]
            # to include the left out element when (len(self) * p / sum_p) is not an integer
            lengths[0] += len(indices) - sum(lengths)

            for i, (offset, length) in enumerate(zip(accumulate(lengths), lengths)):
                splitted_indices[i] = (
                        splitted_indices[i] + indices[offset - length: offset]
                )

        return [Subset(self, indices) for indices in splitted_indices]


class PartitionDataset(MultiModDataset):
    """Tabular dataset for MultiModN initiable with X and y np arrays"""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 partitions: Optional[List[int]] = None):
        if partitions is None:
            self.partitions = [X.shape[1]]
        else:
            self.partitions = partitions

        if sum(self.partitions) != X.shape[1]:
            raise ValueError(
                "Paritions sum doesn't match data dimension. Expected: {}, got: {}"
                .format(sum(self.partitions), X.shape[1])
            )

        self.n_partitions = len(self.partitions)
        partition_offsets = list(accumulate(self.partitions[:-1]))

        self.X = np.split(X, partition_offsets, axis=1)
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], np.ndarray]:
        """Returns an array of tensors corresponding to the sample at each partition"""

        tensor_array = [
            Tensor(self.X[partition_idx][idx])
            for partition_idx in range(self.n_partitions)
        ]

        return tensor_array, self.y[idx]


class FeatureWiseDataset(PartitionDataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]

        super().__init__(X, y, [1] * n_features)


class JointDatasets(MultiModDataset):
    def __init__(self, datasets: List[Dataset]):
        assert all(len(dataset) == len(datasets[0]) for dataset in
                   datasets), "Datasets must have the same length"

        self.datasets = datasets

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], np.ndarray]:
        tensor_array = [
            torch.cat(dataset[idx][0])
            for dataset in self.datasets
        ]

        return tensor_array, self.datasets[0][idx][1]
