import random
import torch
from torch.optim import Optimizer
from torch import Tensor
from torch.utils.data import DataLoader

from multimodn.encoders.multimod_encoder import MultiModEncoder
from multimodn.decoders.multimod_decoder import MultiModDecoder
from multimodn.state import InitState, TrainableInitState
from multimodn.history import MultiModNHistory
from typing import List, Optional, Iterable, Tuple, Callable, Union
import torch.nn as nn
import numpy as np
from torchsummary import summary

from torchmetrics import Metric, MetricCollection, ConfusionMatrix, F1Score, ROC, PrecisionRecallCurve, Accuracy, AUROC

from collections import defaultdict
import itertools
import warnings
#warnings.filterwarnings("ignore")

class FullState(Metric):
    def __init__(self, task = 'binary', num_classes = 2, average = 'macro', ):   
        super().__init__()     
        self.add_state('prediction', default = [], dist_reduce_fx = 'cat')
        self.add_state('target', default = [], dist_reduce_fx = 'cat')
        self.task = task
        self.num_classes = num_classes
        self.average = average
            
    def update(self, preds: Tensor, target: Tensor):            
        assert preds.shape == target.shape        
        self.prediction.append(preds)
        self.target.append(target)
            
    def compute(self):        
        self.prediction = torch.cat(self.prediction).to('cpu')
        self.target = torch.cat(self.target).to('cpu')
        metrics = MetricCollection({
        'f1-score': F1Score(task = self.task, num_classes = self.num_classes, average = self.average, ),
        'auroc': AUROC(task = self.task, num_classes = self.num_classes, average = self.average, ),    
        'accuracy': Accuracy(task = self.task, num_classes = self.num_classes, ),             
        'roc-curve': ROC(task = self.task, num_classes = self.num_classes, ), 
        'pr-curve': PrecisionRecallCurve(task = self.task, num_classes = self.num_classes, ), 
        'confusion matrix': ConfusionMatrix(task = self.task, num_classes = self.num_classes, )},)        
        metrics.update(self.prediction, self.target)
        return metrics.compute()

def get_metrics_collection(task, num_classes, average, device, all_per_batch):      
    if all_per_batch:                  
        return MetricCollection({
            'f1-score': F1Score(task = task, num_classes = num_classes, average = average, ),
            'auroc': AUROC(task = task, num_classes = num_classes, average = average, ),
            'accuracy': Accuracy(task = task, num_classes = num_classes, ),        
            'roc-curve': ROC(task = task, num_classes = num_classes, ), 
            'pr-curve': PrecisionRecallCurve(task = task, num_classes = num_classes, ), 
            'confusion matrix': ConfusionMatrix(task = task, num_classes = num_classes, )},         
        ).to(device)
    else:
        return FullState(task = task, num_classes = num_classes, average = average,).to(device)
            

def store_performance(enc_nr, dec_nr, device, all_per_batch, task = 'binary', num_classes = 2, average = 'macro'):    
    performance_storage = defaultdict(dict)
    encoder_decoder_pairs = list(itertools.product(list(range(enc_nr + 1)), list(range(dec_nr))))
    for pair in encoder_decoder_pairs:
        # first the encoder index, then the decoder's
        performance_storage[pair[0]][pair[1]] = get_metrics_collection(task, num_classes, average, device, all_per_batch)
    return performance_storage
    
def unravel_confusion_matrix(confmat, tp, fp, fn, tn, enc_idx, dec_idx):    
    tp[enc_idx][dec_idx] = confmat[1][1] 
    fp[enc_idx][dec_idx] = confmat[0][1]
    fn[enc_idx][dec_idx] = confmat[1][0]
    tn[enc_idx][dec_idx] = confmat[0][0]
    return tp, fp, fn, tn

def get_results(results_dict):       
    confmat = results_dict['confusion matrix']
    tp = confmat[1][1] 
    fp = confmat[0][1]
    fn = confmat[1][0]
    tn = confmat[0][0]
  
    if (tp + fn) != 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = torch.zeros([1])
    if (tn + fp) != 0:
        specificity = tn / (tn + fp)   
    else:
        specificity = torch.zeros([1])
        
    fpr, tpr, thr_roc = results_dict['roc-curve']
    precision, recall, thr_pr = results_dict['pr-curve']    
    results = results_dict['f1-score'], results_dict['auroc'], results_dict['accuracy'], sensitivity, specificity, fpr, tpr, precision, recall, \
      tn, fp, fn, tp, thr_roc, thr_pr
    results = list(map(lambda metric: metric.cpu().detach(), results))   
  
    return results
        
       
class MultiModN(nn.Module):
    def __init__(
            self,
            state_size: int,
            encoders: List[MultiModEncoder],
            decoders: List[MultiModDecoder],
            err_penalty: float,
            state_change_penalty: float,
            shuffle_mode: Optional[bool] = False,
            init_state: Optional[InitState] = None,
            device: Optional[torch.device] = None,
    ):
        super(MultiModN, self).__init__()
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
            history: Optional[MultiModNHistory] = None,
            log_interval: Optional[int] = None,
            logger: Optional[Callable] = None,      
            all_per_batch: bool = False,     
    ) -> None:
        # If log interval is given and logger is not, use print as default logger
        if log_interval and not logger:
            logger = print
        self.train()
        n_batches = len(train_loader)
        accuracy_epoch = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)        
        tp_epoch = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        tn_epoch = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        fp_epoch = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        fn_epoch = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)

        err_loss_epoch = np.zeros((len(self.encoders) + 1, len(self.decoders)))
        state_change_epoch = np.zeros(len(self.encoders))
        
        local_performance_storage = store_performance(len(self.encoders), len(self.decoders), self.device, all_per_batch = all_per_batch)
        
        for batch_idx, batch in enumerate(train_loader):
            # Note: for multiclass target should be = [0, n_classes -1] for the correctness of CrossEntropyLoss
            data, target, encoder_sequence = (list(batch) + [None])[:3]
            batch_size = target.shape[0]
            err_loss = torch.zeros((len(self.encoders) + 1, len(self.decoders)))
            state_change = torch.zeros(len(self.encoders))

            data_encoders = [data_encoder.to(self.device) for data_encoder in data]
            
            target = target.type(torch.LongTensor)
            target = target.to(self.device)

            optimizer.zero_grad()

            state: Tensor = self.init_state(batch_size)

            for dec_idx, decoder in enumerate(self.decoders):
                target_decoder = target[:, dec_idx]
                output_decoder = decoder(state)
                output_decoder_proba = torch.div(output_decoder, torch.sum(output_decoder, dim=1).reshape(-1,1))                   
                err_loss[0][dec_idx] = criterion(output_decoder, target_decoder)                
                local_performance_storage[0][dec_idx].update(output_decoder_proba[:,1], target_decoder)                
                
            for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                               shuffle_mode=self.shuffle_mode,
                                                               train=True):
                encoder = self.encoders[enc_idx]
                data_encoder = data_encoders[data_idx]

                old_state = state.clone()

                # Skip encoder if data contains nan value
                if any(data_encoder.isnan().flatten()):
                    continue
                state = encoder(state, data_encoder)
                state_change[enc_idx] = torch.mean((state - old_state) ** 2)
                for dec_idx, decoder in enumerate(self.decoders):
                    target_decoder = target[:, dec_idx]
                    output_decoder = decoder(state)
                    output_decoder_proba = torch.div(output_decoder, torch.sum(output_decoder, dim=1).reshape(-1,1))                    
                    err_loss[enc_idx + 1][dec_idx] = criterion(output_decoder,
                                                               target_decoder)
                    local_performance_storage[enc_idx + 1][dec_idx].update(output_decoder_proba[:,1], target_decoder)            

                        
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
        for dec_idx in range(len(self.decoders)):
            for enc_idx in range(len(self.encoders) + 1):
                global_res = local_performance_storage[enc_idx][dec_idx].compute()            
                accuracy_epoch[enc_idx][dec_idx] = global_res['accuracy']
                confmat = global_res['confusion matrix']
                tp_epoch, fp_epoch, fn_epoch, tn_epoch = unravel_confusion_matrix(confmat, tp_epoch, fp_epoch, fn_epoch, \
                                                                                    tn_epoch, enc_idx, dec_idx)
        err_loss_epoch /= n_batches
        state_change_epoch /= n_batches
        accuracy_epoch = accuracy_epoch.detach().cpu().numpy()
        # Compute metrics for the current epoch
        # Use np.where to avoid NaNs, set the whole metric to zero
        # in case of the equality of denominator to zero
        # At the end move all metrics to cpu and convert to numpy for history

        #Note, that here we compute metrics for all encoders and decoders, \
        # and at the history file select the last encoder for the final metric
        sensitivity_denominator = tp_epoch + fn_epoch
        sensitivity_epoch = torch.where(sensitivity_denominator == 0, 0,
                                     tp_epoch / sensitivity_denominator).detach().cpu().numpy()

        specificity_denominator = tn_epoch + fp_epoch
        specificity_epoch = torch.where(specificity_denominator == 0, 0,
                                     tn_epoch / specificity_denominator).detach().cpu().numpy()

        balanced_accuracy_epoch = (sensitivity_epoch + specificity_epoch) / 2

        if history is not None:
            history.state_change_loss.append(state_change_epoch)
            history.loss['train'].append(err_loss_epoch)
            history.accuracy['train'].append(accuracy_epoch)
            history.sensitivity['train'].append(sensitivity_epoch)
            history.specificity['train'].append(specificity_epoch)
            history.balanced_accuracy['train'].append(balanced_accuracy_epoch)
        # Output the results for each decoder with the state vector after the last encoder as input   
        results = [[]] * len(self.decoders)   
        last_enc_idx = len(self.encoders)
        for dec_idx in range(len(self.decoders)):
            sngl_decoder_dict = local_performance_storage[last_enc_idx][dec_idx].compute()                 
            results[dec_idx] = get_results(sngl_decoder_dict)
        return results

    def test(
            self,
            test_loader: DataLoader,
            criterion: Union[nn.Module, Callable],
            history: Optional[MultiModNHistory] = None,
            tag: str = 'test',
            log_results: bool = False,
            logger: Optional[Callable] = None,
            all_per_batch: bool = False,
    ):
        # If log interval is given and logger is not, use print as default logger
        if log_results and not logger:
            logger = print
        self.eval()

        n_batches = len(test_loader)       

        err_loss_prediction = np.zeros((len(self.encoders) + 1, len(self.decoders)))
        
        accuracy_prediction = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        tp_prediction = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        tn_prediction = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        fp_prediction = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        fn_prediction = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        
        local_performance_storage = store_performance(len(self.encoders), len(self.decoders), self.device, all_per_batch = all_per_batch)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                data, target, encoder_sequence = (list(batch) + [None])[:3]

                batch_size = target.shape[0]               

                err_loss = torch.zeros((len(self.encoders) + 1, len(self.decoders)))

                data_encoders = [data_encoder.to(self.device) for data_encoder in data]
                
                target = target.type(torch.LongTensor)
                target = target.to(self.device)

                state: Tensor = self.init_state(batch_size)

                for dec_idx, decoder in enumerate(self.decoders):
                    target_decoder = target[:, dec_idx]
                    output_decoder = decoder(state)
                    output_decoder_proba = torch.div(output_decoder, torch.sum(output_decoder, dim=1).reshape(-1,1))                     
                        
                    err_loss[0][dec_idx] = criterion(output_decoder, target_decoder)
                    
                    local_performance_storage[0][dec_idx].update(output_decoder_proba[:,1], target_decoder)
                    
  
                for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                                   shuffle_mode=self.shuffle_mode,
                                                                   train=False):
                    encoder = self.encoders[enc_idx]
                    data_encoder = data_encoders[data_idx]

                    # skip encoder if data contains nan value
                    if any(data_encoder.isnan().flatten()):
                        continue

                    state = encoder(state, data_encoder)

                    for dec_idx, decoder in enumerate(self.decoders):
                        target_decoder = target[:, dec_idx]
                        output_decoder = decoder(state)
                        output_decoder_proba = torch.div(output_decoder, torch.sum(output_decoder, dim =1).reshape(-1,1))
                        err_loss[enc_idx + 1][dec_idx] = criterion(output_decoder,
                                                                   target_decoder)                        
                        local_performance_storage[enc_idx + 1][dec_idx].update(output_decoder_proba[:,1], target_decoder)                 
                        
                            
                err_loss_prediction += err_loss.detach().numpy()
                
                
        for dec_idx in range(len(self.decoders)):
            for enc_idx in range(len(self.encoders) + 1):
                global_res = local_performance_storage[enc_idx][dec_idx].compute()            
                accuracy_prediction[enc_idx][dec_idx] = global_res['accuracy']
                confmat = global_res['confusion matrix']
                tp_prediction, fp_prediction, fn_prediction, tn_prediction = unravel_confusion_matrix(confmat, tp_prediction, fp_prediction, fn_prediction, \
                                                                                    tn_prediction, enc_idx, dec_idx)
        err_loss_prediction /= n_batches
        accuracy_prediction = accuracy_prediction.detach().cpu().numpy()        
        sensitivity_denominator = tp_prediction + fn_prediction
        sensitivity_prediction = torch.where(sensitivity_denominator == 0, 0,
                                          tp_prediction / sensitivity_denominator).detach().cpu().numpy()

        specificity_denominator = tn_prediction + fp_prediction
        specificity_prediction = torch.where(specificity_denominator == 0, 0,
                                          tn_prediction / specificity_denominator).detach().cpu().numpy()

        balanced_accuracy_prediction = (sensitivity_prediction + specificity_prediction) / 2

        if log_results:
            logger(
                f"{tag.capitalize()} results\n"
                f"\tAverage loss: {np.mean(err_loss_prediction):.4f}\n"
                f"\tAccuracy: {np.mean(accuracy_prediction):.4f}\n"
                f"\tSensitivity: {sensitivity_prediction:.4f}\n"
                f"\tSpecificity: {specificity_prediction:.4f}\n"
                f"\tBalanced accuracy: {balanced_accuracy_prediction:.4f}"
            )

        if history is not None:
            if tag not in history.loss:
                history.loss[tag] = []
            history.loss[tag].append(err_loss_prediction)

            if tag not in history.accuracy:
                history.accuracy[tag] = []
            history.accuracy[tag].append(accuracy_prediction)

            if tag not in history.sensitivity:
                history.sensitivity[tag] = []
            history.sensitivity[tag].append(sensitivity_prediction)

            if tag not in history.specificity:
                history.specificity[tag] = []
            history.specificity[tag].append(specificity_prediction)

            if tag not in history.balanced_accuracy:
                history.balanced_accuracy[tag] = []
            history.balanced_accuracy[tag].append(balanced_accuracy_prediction)
            
        # Output the results for each decoder with the state vector after the last encoder as input   
        results = [[]] * len(self.decoders)   
        last_enc_idx = len(self.encoders)
        for dec_idx in range(len(self.decoders)):
            sngl_decoder_dict = local_performance_storage[last_enc_idx][dec_idx].compute()     
            local_performance_storage[last_enc_idx][dec_idx].reset() # -/(2)/-
            results[dec_idx] = get_results(sngl_decoder_dict)
        return results     


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

                # To predict probabilities instead of final class
                #full_predictions[0][dec_idx] = output_decoder[..., -1].item()

            for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                               shuffle_mode=self.shuffle_mode,
                                                               train=False):
                encoder = self.encoders[enc_idx]
                state = encoder(state, x_encoders[data_idx])

                for dec_idx, decoder in enumerate(self.decoders):
                    output_decoder = decoder(state)
                    _, prediction = torch.max(output_decoder, dim=1)

                    full_predictions[enc_idx + 1][dec_idx] = prediction.detach().numpy()
                    # full_predictions[enc_idx + 1][dec_idx] = output_decoder[..., -1].item()

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
