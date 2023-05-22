from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class MoMoNetHistory():
    """Training history of MoMoNet"""

    def __init__(
        self,
        targets: List[str]
    ):
        self.decoder_names: List[str] = targets

        self.state_change_loss: List[np.ndarray] = []

        self.loss: Dict[str, List[np.ndarray]] = {
            'train': [],
        }
        self.accuracy: Dict[str, List[np.ndarray]] = {
            'train': [],
        }

    def plot(
        self,
        filepath: str,
        targets_to_display: List[str],
        show_state_change: bool = False,
    ):
        n_cols = len(self.loss)
        n_rows = 2
        fig, ax = plt.subplots(figsize=(10*n_cols, 5*n_rows), nrows=n_rows, ncols=n_cols)

        # Plot state change loss curve
        if show_state_change:
            ax.plot(list(map(lambda x: x[-1], self.state_change_loss)), label='State change loss')
        
        # Plot curves for each target
        for i, target_name in enumerate(targets_to_display):
            if target_name not in self.decoder_names:
                raise ValueError(
                    "Target name '{}' is not part of the MoMoNet history".format(target_name)
                )

            # Plot loss curves
            for col_idx, (key, value) in enumerate(self.loss.items()):
                label = f"{target_name}"
                ax[0][col_idx].plot(list(map(lambda x: x[-1][i], value)), label=label)
                ax[0][col_idx].legend(loc="best")
                ax[0][col_idx].set_title(f"{key.capitalize()} Loss")
                ax[0][col_idx].grid(True)
            
            # Plot accuracy curves
            for col_idx, (key, value) in enumerate(self.accuracy.items()):
                label = f"{target_name}"
                ax[1][col_idx].plot(list(map(lambda x: x[-1][i], value)), label=label)
                ax[1][col_idx].legend(loc="best")
                ax[1][col_idx].set_title(f"{key.capitalize()} Accuracy")
                ax[1][col_idx].grid(True)

        plt.tight_layout()
        fig.savefig(filepath)

    def get_results(self) -> pd.DataFrame:
        n_metrics = len(self.loss) + len(self.accuracy) + 1
        start_index = 0
        columns = []
        results = np.zeros((len(self.decoder_names), n_metrics))

        # State change loss, same value for each row (independent of decoder)
        results[:, start_index] = [self.state_change_loss[-1][-1]]*len(self.decoder_names)
        columns.append('State change loss')
        start_index += 1

        # Losses
        for j, (key, value) in enumerate(self.loss.items()):
            columns.append('{} loss'.format(display_title(key)))

            for i, _ in enumerate(self.decoder_names):
                results[i, j+start_index] = value[-1][-1][i]
        start_index += len(self.loss)

        # Accuracies
        for j, (key, value) in enumerate(self.accuracy.items()):
            columns.append('{} accuracy'.format(display_title(key)))

            for i, _ in enumerate(self.decoder_names):
                results[i, j+start_index] = value[-1][-1][i]
        start_index += len(self.loss)

        df = pd.DataFrame(results, columns=columns)
        df.index = self.decoder_names

        return df

    def print_results(self):
        print(self.get_results())

    def save_results(self, path):
        df = self.get_results()

        df.to_csv(path, index_label='Target')

def display_title(key: str):
    title = key.replace('_', ' ')
    title = title.capitalize()

    return title
