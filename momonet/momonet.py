import random
import torch
from torch.optim import Optimizer
from torch import Tensor
from torch.utils.data import DataLoader

from momonet.encoders.momo_encoder import MoMoEncoder
from momonet.decoders.momo_decoder import MoMoDecoder
from momonet.state import InitState, TrainableInitState
from momonet.history import MoMoNetHistory
from typing import List, Optional, Iterable, Tuple, Callable, Union
import torch.nn as nn
import numpy as np
from torchsummary import summary


class MoMoNet(nn.Module):
    def __init__(
            self,
            state_size: int,
            encoders: List[MoMoEncoder],
            decoders: List[MoMoDecoder],
            err_penalty: float,
            state_change_penalty: float,
            shuffle_mode: Optional[bool] = False,
            init_state: Optional[InitState] = None,
            device: Optional[torch.device] = None,
    ):
        super(MoMoNet, self).__init__()
        self.shuffle_mode = shuffle_mode
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.init_state = TrainableInitState(
            state_size, self.device) if not init_state else init_state
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.err_penalty = err_penalty
        self.state_change_penalty = 0.01 * state_change_penalty
        self.to(self.device)  # Move to device

    def train_epoch(
            self,
            train_loader: DataLoader,
            optimizer: Optimizer,
            criterion: Union[nn.Module, Callable],
            history: Optional[MoMoNetHistory] = None,
            log_interval: Optional[int] = None,
            logger: Optional[Callable] = None,
    ) -> None:
        # If log interval is given and logger is not, use print as default logger
        if log_interval and not logger:
            logger = print
        self.train()

        n_batches = len(train_loader)
        n_samples_epoch = np.ones((len(self.encoders) + 1, 1))

        err_loss_epoch = np.zeros((len(self.encoders) + 1, len(self.decoders)))
        state_change_epoch = np.zeros(len(self.encoders))
        n_correct_epoch = np.zeros((len(self.encoders) + 1, len(self.decoders)))

        for batch_idx, batch in enumerate(train_loader):
            data, target, encoder_sequence = (list(batch) + [None])[:3]

            batch_size = target.shape[0]
            n_samples_epoch[0] += batch_size

            err_loss = torch.zeros((len(self.encoders) + 1, len(self.decoders)))
            state_change = torch.zeros(len(self.encoders))

            data_encoders = [data_encoder.to(self.device) for data_encoder in data]
            target = target.to(self.device)

            optimizer.zero_grad()

            state: Tensor = self.init_state(batch_size)

            for dec_idx, decoder in enumerate(self.decoders):
                target_decoder = target[:, dec_idx]
                output_decoder = decoder(state)
                _, prediction = torch.max(output_decoder, dim=1)

                err_loss[0][dec_idx] = criterion(output_decoder, target_decoder)
                n_correct_epoch[0][dec_idx] += sum(prediction == target_decoder).float()

            for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                               shuffle_mode=self.shuffle_mode,
                                                               train=True):
                encoder = self.encoders[enc_idx]
                data_encoder = data_encoders[data_idx]

                old_state = state.clone()

                # Skip encoder if data contains nan value
                if any(data_encoder.isnan().flatten()):
                    continue

                n_samples_epoch[enc_idx + 1] += batch_size

                state = encoder(state, data_encoder)
                state_change[enc_idx] = torch.mean((state - old_state) ** 2)

                for dec_idx, decoder in enumerate(self.decoders):
                    target_decoder = target[:, dec_idx]
                    output_decoder = decoder(state)
                    _, prediction = torch.max(output_decoder, dim=1)

                    err_loss[enc_idx + 1][dec_idx] = criterion(output_decoder,
                                                               target_decoder)
                    n_correct_epoch[enc_idx + 1][dec_idx] += sum(
                        prediction == target_decoder)

            # Global losses (combining all encoders and decoders) at batch level
            global_err_loss = torch.sum(err_loss) / (
                    len(self.decoders) * (len(self.encoders) + 1))
            global_state_change = torch.sum(state_change) / (len(self.encoders))
            # Loss = global_err_loss * err_penalty +
            #        0.01 * global_state_change * state_change_penalty
            loss = (
                global_err_loss * self.err_penalty +
                global_state_change * self.state_change_penalty
            )
            loss.backward()
            optimizer.step()

            err_loss_epoch += err_loss.detach().numpy()
            state_change_epoch += state_change.detach().numpy()

            if log_interval and batch_idx % log_interval == log_interval - 1:
                logger(
                    f"Batch {batch_idx + 1}/{n_batches}\n"
                    f"\tLoss: {loss.item():.4f}\n"
                    f"\tErr loss: {global_err_loss.item():.4f}\n"
                    f"\tState change: {global_state_change.item():.4f}"
                )

        err_loss_epoch /= n_batches
        state_change_epoch /= n_batches
        accuracy_epoch = n_correct_epoch / n_samples_epoch

        if history is not None:
            history.state_change_loss.append(state_change_epoch)
            history.loss['train'].append(err_loss_epoch)
            history.accuracy['train'].append(accuracy_epoch)

    def test(
            self,
            test_loader: DataLoader,
            criterion: Union[nn.Module, Callable],
            history: Optional[MoMoNetHistory] = None,
            tag: str = 'test',
            log_results: bool = False,
            logger: Optional[Callable] = None,
    ):
        # If log interval is given and logger is not, use print as default logger
        if log_results and not logger:
            logger = print
        self.eval()

        n_batches = len(test_loader)
        n_samples_prediction = np.ones((len(self.encoders) + 1, 1))

        err_loss_prediction = np.zeros((len(self.encoders) + 1, len(self.decoders)))
        n_correct_prediction = np.zeros((len(self.encoders) + 1, len(self.decoders)))

        with torch.no_grad():
            for batch in test_loader:
                data, target, encoder_sequence = (list(batch) + [None])[:3]

                batch_size = target.shape[0]
                n_samples_prediction[0] += batch_size

                err_loss = torch.zeros((len(self.encoders) + 1, len(self.decoders)))

                data_encoders = [data_encoder.to(self.device) for data_encoder in data]
                target = target.to(self.device)

                state: Tensor = self.init_state(batch_size)

                for dec_idx, decoder in enumerate(self.decoders):
                    target_decoder = target[:, dec_idx]
                    output_decoder = decoder(state)
                    _, prediction = torch.max(output_decoder, dim=1)

                    err_loss[0][dec_idx] = criterion(output_decoder, target_decoder)
                    n_correct_prediction[0][dec_idx] += sum(
                        prediction == target_decoder).float()

                for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                                   shuffle_mode=self.shuffle_mode,
                                                                   train=False):
                    encoder = self.encoders[enc_idx]
                    data_encoder = data_encoders[data_idx]

                    # skip encoder if data contains nan value
                    if any(data_encoder.isnan().flatten()):
                        continue

                    n_samples_prediction[enc_idx + 1] += batch_size

                    state = encoder(state, data_encoder)

                    for dec_idx, decoder in enumerate(self.decoders):
                        target_decoder = target[:, dec_idx]
                        output_decoder = decoder(state)
                        _, prediction = torch.max(output_decoder, dim=1)

                        err_loss[enc_idx + 1][dec_idx] = criterion(output_decoder,
                                                                   target_decoder)
                        n_correct_prediction[enc_idx + 1][dec_idx] += sum(
                            prediction == target_decoder)

                err_loss_prediction += err_loss.detach().numpy()

        err_loss_prediction /= n_batches
        accuracy_prediction = n_correct_prediction / n_samples_prediction

        if log_results:
            logger(
                f"{tag.capitalize()} results\n"
                f"\tAverage loss: {np.mean(err_loss_prediction):.4f}\n"
                f"\tAccuracy: {np.mean(accuracy_prediction):.4f}"
            )

        if history is not None:
            if tag not in history.loss:
                history.loss[tag] = []
            history.loss[tag].append(err_loss_prediction)

            if tag not in history.accuracy:
                history.accuracy[tag] = []
            history.accuracy[tag].append(accuracy_prediction)

    def predict(
            self,
            x: List[Tensor],
            encoder_sequence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.eval()

        n_samples = x[0].shape[0]
        full_predictions = np.zeros(
            (len(self.encoders) + 1, len(self.decoders), n_samples))

        with torch.no_grad():
            x_encoders = [x_encoder.to(self.device) for x_encoder in x]

            state: Tensor = self.init_state(n_samples)

            for dec_idx, decoder in enumerate(self.decoders):
                output_decoder = decoder(state)
                _, prediction = torch.max(output_decoder, dim=1)

                full_predictions[0][dec_idx] = prediction.detach().numpy()

            for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                               shuffle_mode=self.shuffle_mode,
                                                               train=False):
                encoder = self.encoders[enc_idx]

                state = encoder(state, x_encoders[data_idx])

                for dec_idx, decoder in enumerate(self.decoders):
                    output_decoder = decoder(state)
                    _, prediction = torch.max(output_decoder, dim=1)

                    full_predictions[enc_idx + 1][dec_idx] = prediction.detach().numpy()

        return full_predictions

    def get_states(
            self,
            data_loader: DataLoader,
    ) -> List[Tensor]:
        self.eval()

        batch_states = []

        with torch.no_grad():
            for batch in data_loader:
                data, _, encoder_sequence = (list(batch) + [None])[:3]

                batch_size = data[0].shape[0]

                data_encoders = [data_encoder.to(self.device) for data_encoder in data]

                state: Tensor = self.init_state(batch_size)

                for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                                   shuffle_mode=self.shuffle_mode,
                                                                   train=False):
                    encoder = self.encoders[enc_idx]
                    data_encoder = data_encoders[data_idx]

                    # skip encoder if data contains nan value
                    if any(data_encoder.isnan().flatten()):
                        continue

                    state = encoder(state, data_encoder)

                batch_states.append(state)

        return list(torch.cat(batch_states, dim=0))

    def display_arch(self, input: np.ndarray):
        for i, enc in enumerate(self.encoders):
            print('Encoder {}:'.format(i))
            state_shape = torch.Size([self.init_state.state_size])

            summary(enc, [state_shape, input[i].shape])
            print()

        for i, dec in enumerate(self.decoders):
            print('Decoder {}:'.format(i))
            state_shape = torch.Size([self.init_state.state_size])

            summary(dec, state_shape)
            print()

    def get_encoder_iterable(
            self,
            encoder_sequence: List[int],
            shuffle_mode: bool,
            train: bool,
    ) -> Iterable[Tuple[int, int]]:
        if encoder_sequence is None:
            encoder_iterable = enumerate(range(len(self.encoders)))
        else:
            encoder_iterable_batch = encoder_sequence.numpy().copy()
            encoder_iterable = encoder_iterable_batch[0]

            if not (encoder_iterable_batch == encoder_iterable).all():
                raise ValueError(
                    "Encoder sequence has different values across the batch. Hint: set batch size to 1 to avoid this error."
                )

            encoder_iterable = enumerate(encoder_iterable)

        if shuffle_mode and train:
            encoder_iterable = list(encoder_iterable)
            random.shuffle(encoder_iterable)

        return encoder_iterable
