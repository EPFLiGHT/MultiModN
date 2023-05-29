import sys
import os
from os import path as o
storage_path = o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../.."))
sys.path.append(storage_path)
from tqdm.auto import trange
import torch
from torch import Tensor, sigmoid
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss
from multimodn.multimodn import MultiModN
from multimodn.encoders import MIMIC_MLPEncoder
from multimodn.decoders import MLPDecoder
from multimodn.history import MultiModNHistory
from datasets.mimic import MIMICDataset
from pipelines import utils
import importlib
import haim_api
importlib.reload(haim_api)
from haim_api import HAIMDecoder, HAIM
import argparse
import pickle as pkl

from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import copy

performance_metrics = ['f1', 'auc', 'accuracy', 'sensitivity', 'specificity', 'fpr', 'tpr', 'precision', 'recall', \
    'tn', 'fp', 'fn', 'tp', 'thr_roc', 'thr_pr']

hyperparameters = ['model', 'target', 'both', 'fold', 'miss_perc', 'seed', 'state_size', 'batch_size', 'encoder_hidd_units', 'decoder_hidd_units', 'dropout', 'epochs']

save_logs = hyperparameters + performance_metrics

source_names = ['de', 'vd',  'vmd', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech', 'n_rad']

source_size = [ 6, 1024, 1024, 99, 242, 110, 768, 768, 768]

source_dict = dict(zip(source_names, source_size))

argParser = argparse.ArgumentParser()

argParser.add_argument("-p", "--miss_perc", type=float, help="percentage of samples with systematic missingness")

args = argParser.parse_args()


def main():
    PIPELINE_NAME = utils.extract_pipeline_name(sys.argv[0])    
    criterion = '(auc + bac)'    
    results_directory = os.path.join(storage_path, 'nips', 'results')
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)               
    
    model_type = PIPELINE_NAME + '_' + criterion    
    
    results_file_path = os.path.join(results_directory, model_type + '.csv') 
    
    sources = ['de', 'vd', 'n_ech', 'ts_ce']

    source_spec = '_'.join(sources) 

    targets = ['Enlarged Cardiomediastinum', 'Cardiomegaly']

    pathologies = '_'.join(targets)
    
    # Hyperparameters    
    state_size = 50

    learning_rate = .001

    epochs = 100

    decoder_hidd_units = 32

    err_penalty = 1

    state_change_penalty = 0

    dropout = 0.2

    batch_size_train = 16

    batch_size_val = batch_size_train

    encoder_hidd_units = decoder_hidd_units
    
    keep_missing_values = True
        
    miss_perc = args.miss_perc
    
    if miss_perc == 0:
        put_none = False
    else:
        put_none = True        
    features_to_nan = ['vd_'+ str(k) for k in range(1024)]
    
    boths = [True, False] # If True, then both train and test sets will be degraded, otherwise only the train (given that put_none = True)
    class_label = 1 # Regarding which class put missing values     
    dummy = False # Systematic missingness w.r.t to the aggregated "haim" label (dummy = True) or the actual one (dummy = False)
    nfold = 5    

    for target in targets:    
        model_spec = target            
        # Dataset splitting based on hospitalisation id & aggregated label, i.e. samples with the same haim_id should be all either in train or validation or test subsets
        data_file_path = os.path.join(storage_path, 'datasets/mimic', pathologies, source_spec)
        patient_labels = pd.read_csv(os.path.join(data_file_path,'how_to_split.csv'))
        df = pd.read_csv(os.path.join(data_file_path,'data.csv'))    
        haim_id = np.array(patient_labels['haim_id'])
        labels = np.array(patient_labels['label'])    
        
        seed = 0       
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
                
        for i, (id_train, id_test_val) in enumerate(skf.split(haim_id, labels)):
            torch.manual_seed(seed) 
            ex_prefix = f'seed_{seed}_state_size_{state_size}_batch_size_{batch_size_train}_dec_hidd_units_{decoder_hidd_units}_dropout_{dropout}'
            part_of_hyperparameters = [target, i, miss_perc, seed, state_size, batch_size_train, encoder_hidd_units, decoder_hidd_units, dropout, epochs] 
            haim_id_test_and_val = patient_labels.iloc[list(id_test_val)]['haim_id']  
            labels_test_val = labels[id_test_val] 
            id_test, id_val, _, _ = train_test_split(haim_id_test_and_val, labels_test_val, test_size = .5, stratify = labels_test_val, random_state = seed)
            train_ind = df[df.haim_id.isin(haim_id[id_train])].index         
            val_ind =  df[df.haim_id.isin(id_val)].index
            test_ind = df[df.haim_id.isin(id_test)].index 
            
            if put_none:
                if dummy:
                    train_haim_subset = patient_labels[(patient_labels.haim_id.isin(haim_id[id_train])) & (patient_labels.label == class_label)]['haim_id']
                    train_ind_same_class = df[df.haim_id.isin(train_haim_subset)].index     
                    train_same_len = len(train_ind_same_class)
                    nan_size = round(miss_perc / 100 * train_same_len)
                    indices_to_nan = train_ind_same_class[:nan_size]  # (1) no need to randomize indexes here, since shuffle = True in StratifiedKFold
                    val_haim_subset = patient_labels[(patient_labels.haim_id.isin(id_val)) & (patient_labels.label == class_label)]['haim_id']
                    val_ind_same_class = df[df.haim_id.isin(val_haim_subset)].index   
                    val_same_len = len(val_ind_same_class)
                    nan_size = round(miss_perc / 100 * val_same_len)
                    indices_to_nan = np.concatenate([indices_to_nan, val_ind_same_class[:nan_size]]) # -/(1)/-
                else:
                    train_ind_same_class = df[(df.haim_id.isin(haim_id[id_train])) & (df[target] == class_label)].index     
                    train_same_len = len(train_ind_same_class)
                    nan_size = round(miss_perc / 100 * train_same_len)
                    indices_to_nan = train_ind_same_class[:nan_size] # -/(1)/-
                    val_ind_same_class = df[df.haim_id.isin(id_val) & (df[target] == class_label)].index   
                    val_same_len = len(val_ind_same_class)
                    nan_size = round(miss_perc / 100 * val_same_len)
                    indices_to_nan = np.concatenate([indices_to_nan, val_ind_same_class[:nan_size]]) # -/(1)/-                    
            else:
                indices_to_nan = []             
            dataset_modn = MIMICDataset(sources, targets = [target], dropna=not keep_missing_values, put_none = put_none, indices_to_nan=indices_to_nan, features_to_nan=features_to_nan)  
            partitions = dataset_modn.partitions           
            dataset_modn = dataset_modn.partition_dataset(partitions)   
        
            dataset_haim = MIMICDataset(sources, targets = [target], dropna=not keep_missing_values, put_none = put_none, nanfill = True, indices_to_nan=indices_to_nan, features_to_nan=features_to_nan)      

            train_data, val_data =  Subset(dataset_modn, train_ind), Subset(dataset_modn, val_ind)             
            train_loader = DataLoader(train_data, batch_size_train)
            val_loader = DataLoader(val_data, batch_size_val)       
            
            # ModN model specification
            encoders = [MIMIC_MLPEncoder(state_size, partition, (encoder_hidd_units, encoder_hidd_units), activation = F.relu, dropout = dropout, ) for partition in partitions]

            decoders = [MLPDecoder(state_size, (decoder_hidd_units, decoder_hidd_units ), 2, output_activation = sigmoid) for _ in [target]]
            model_modn =  MultiModN(state_size, encoders, decoders, err_penalty, state_change_penalty) 

            optimizer = torch.optim.Adam(list(model_modn.parameters()), learning_rate)

            criterion = CrossEntropyLoss()

            history =  MultiModNHistory([target])
            
            directory = os.path.join(storage_path, 'models', model_spec, source_spec)    

            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path_modn = os.path.join(directory, PIPELINE_NAME + f'_modn_model_{ex_prefix}.pkl')
            best_model_path_modn = os.path.join(directory, PIPELINE_NAME + f'_modn_best_model_{ex_prefix}.pt')
            
            # ModN training            
            best_auc_bac_sum = 0
            for epoch in trange(epochs):            
                if epoch == epochs - 1:
                    train_buff_modn = model_modn.train_epoch(train_loader, optimizer, criterion, history, last_epoch = True)        
                else:
                    model_modn.train_epoch(train_loader, optimizer, criterion, history)
                val_buff_modn = model_modn.test(val_loader, criterion, history, tag='val')
                auc_bac_sum =  val_buff_modn[0][1] + (val_buff_modn[0][3] + val_buff_modn[0][4]) / 2
                # Save the best model based on the sum of validation auroc and bac
                if auc_bac_sum > best_auc_bac_sum:                                        
                    torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model_modn.state_dict(),                    
                    'auc_bac_val_cum': auc_bac_sum,
                    }, best_model_path_modn)  
                    best_auc_bac_sum = auc_bac_sum
                    val_buff_modn_best = val_buff_modn

            pkl.dump(model_modn, open(model_path_modn, 'wb'))

            directory = os.path.join(storage_path, 'history', model_spec, source_spec)                
            if not os.path.exists(directory):
                os.makedirs(directory)
            history_path = os.path.join(directory, PIPELINE_NAME + f'_history_{ex_prefix}.pkl')
            pkl.dump(history, open(history_path, 'wb'))

            directory = os.path.join(storage_path, 'plots', model_spec, source_spec)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plot_path = os.path.join(directory, PIPELINE_NAME + f'_plot_{ex_prefix}.png')  
            
            targets_to_display = [target]
            history.plot(plot_path, targets_to_display, show_state_change=False)
            history.print_results()

            # ModN testing
            flipped_class_label = 1 - class_label
            if put_none:
                for both in boths: 
                    if not both: 
                        dataset_modn = MIMICDataset(sources, targets = [target])                          
                        dataset_modn = dataset_modn.partition_dataset(partitions)
                        test_data =  Subset(dataset_modn, test_ind)
                        test_loader = DataLoader(test_data, batch_size_val)
                    else:
                        if dummy:
                            test_haim_subset = patient_labels[(patient_labels.haim_id.isin(id_test)) & (patient_labels.label == flipped_class_label)]['haim_id']
                            test_ind_same_class = df[df.haim_id.isin(test_haim_subset)].index
                            test_same_len = len(test_ind_same_class)
                            nan_size = round(miss_perc / 100 * test_same_len)                
                            indices_to_nan =  test_ind_same_class[:nan_size] # -/(1)/-                          
                        else:
                            test_ind_same_class = df[df.haim_id.isin(id_test) & (df[target] == flipped_class_label)].index
                            test_same_len = len(test_ind_same_class)                       
                            nan_size = round(miss_perc / 100 * test_same_len)                
                            indices_to_nan =  test_ind_same_class[:nan_size] # -/(1)/-                         
                        dataset_modn = MIMICDataset(sources, targets = [target], put_none = put_none, indices_to_nan=indices_to_nan, features_to_nan=features_to_nan)                     
                        partitions = dataset_modn.partitions           
                        dataset_modn = dataset_modn.partition_dataset(partitions)
                        test_data =  Subset(dataset_modn, test_ind)
                        test_loader = DataLoader(test_data, batch_size_val)                                       

                    part_of_hyperparameters = [target, both, i, miss_perc, seed, state_size, batch_size_train, encoder_hidd_units, decoder_hidd_units, dropout, epochs]          
                    
                    test_modn = model_modn.test(test_loader, criterion)        
                    checkpoint = torch.load(best_model_path_modn)  
                    model_modn.load_state_dict(checkpoint['model_state_dict'])
                    test_modn_best = model_modn.test(test_loader, criterion)
                    results_modn_best = pd.DataFrame(columns=save_logs)
                    test_modn_best_sngl = list(map(lambda metric: metric.numpy(), test_modn_best[0]))
                    row = ['modn'] + part_of_hyperparameters + test_modn_best_sngl
                    results_modn_best.loc[0] = row
                    if os.path.isfile(results_file_path):
                        results_modn_best.to_csv(results_file_path, mode='a', index=False, header=False)
                    else:
                        results_modn_best.to_csv(results_file_path, mode='w', index=False)
            else:
                dataset_modn = MIMICDataset(sources, targets = [target], dropna=not keep_missing_values,)                          
                dataset_modn = dataset_modn.partition_dataset(partitions)
                test_data =  Subset(dataset_modn, test_ind)
                test_loader = DataLoader(test_data, batch_size_val)
                
                part_of_hyperparameters = [target, None, i, miss_perc, seed, state_size, batch_size_train, encoder_hidd_units, decoder_hidd_units, dropout, epochs]              
                test_modn = model_modn.test(test_loader, criterion)        
                checkpoint = torch.load(best_model_path_modn)  
                model_modn.load_state_dict(checkpoint['model_state_dict'])
                test_modn_best = model_modn.test(test_loader, criterion)
                results_modn_best = pd.DataFrame(columns=save_logs)
                test_modn_best_sngl = list(map(lambda metric: metric.numpy(), test_modn_best[0]))
                row = ['modn'] + part_of_hyperparameters + test_modn_best_sngl
                results_modn_best.loc[0] = row
                if os.path.isfile(results_file_path):
                    results_modn_best.to_csv(results_file_path, mode='a', index=False, header=False)
                else:
                    results_modn_best.to_csv(results_file_path, mode='w', index=False)
                    
            train_data, val_data =  Subset(dataset_haim, train_ind), Subset(dataset_haim, val_ind) 
            train_loader = DataLoader(train_data, batch_size_train)
            val_loader = DataLoader(val_data, batch_size_val)
            
            # HAIM model specification
            hidden_layers = (decoder_hidd_units, decoder_hidd_units)
            n_features = dataset_haim.X.shape[1]            
            haim_decoder = HAIMDecoder(n_features, hidden_layers)
            model_haim =  HAIM(haim_decoder) 
            optimizer = torch.optim.Adam(list(model_haim.parameters()), learning_rate)
            criterion = CrossEntropyLoss()

           
            directory = os.path.join(storage_path, 'models', model_spec, source_spec)
            model_path_haim = os.path.join(directory, PIPELINE_NAME + f'_haim_model_{ex_prefix}.pkl')
            best_model_path_haim = os.path.join(directory, PIPELINE_NAME + f'_haim_best_model_{ex_prefix}.pt')

            # HAIM training
            best_auc_bac_sum = 0
            for epoch in trange(epochs):     
                if epoch == epochs - 1:
                    train_buff_haim = model_haim.train_epoch(train_loader, optimizer, criterion, last_epoch = True)                    
                else:
                    model_haim.train_epoch(train_loader, optimizer, criterion)
                    val_buff_haim = model_haim.test(val_loader, criterion)
                auc_bac_sum = val_buff_haim[1] + (val_buff_haim[3] + val_buff_haim[4]) / 2
                if auc_bac_sum > best_auc_bac_sum:                                        
                    torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model_haim.state_dict(),                    
                    'auc_bac_val': auc_bac_sum,
                    }, best_model_path_haim)  
                    best_auc_bac_sum = auc_bac_sum
                    val_buff_haim_best = val_buff_haim
                    
            pkl.dump(model_haim, open(model_path_haim, 'wb'))
            
            # HAIM testing
            if put_none:
                for both in boths: 
                    if not both: 
                        dataset_haim = MIMICDataset(sources, targets = [target]) 
                        test_data =  Subset(dataset_haim, test_ind)
                        test_loader = DataLoader(test_data, batch_size_val)
                    else:
                        dataset_haim = MIMICDataset(sources, targets = [target], dropna=not keep_missing_values, put_none = put_none, nanfill = True, indices_to_nan=indices_to_nan, features_to_nan=features_to_nan)                    
                        test_data =  Subset(dataset_haim, test_ind)
                        test_loader = DataLoader(test_data, batch_size_val)
                    part_of_hyperparameters = [target, both, i, miss_perc, seed, state_size, batch_size_train, encoder_hidd_units, decoder_hidd_units, dropout, epochs]                       
                    test_haim = model_haim.test(test_loader, criterion)        
                    checkpoint = torch.load(best_model_path_haim)  
                    model_haim.load_state_dict(checkpoint['model_state_dict'])
                    test_haim_best = model_haim.test(test_loader, criterion)
                    results_haim_best = pd.DataFrame(columns = save_logs)
                    test_haim_best_sngl = list(map(lambda metric: metric.numpy(), test_haim_best))
                    row = ['haim'] + part_of_hyperparameters + test_haim_best_sngl
                    results_haim_best.loc[0] = row
                    if os.path.isfile(results_file_path):
                        results_haim_best.to_csv(results_file_path, mode='a', index=False, header=False)
                    else:
                        results_haim_best.to_csv(results_file_path, mode='w', index=False)
            else:
                dataset_haim = MIMICDataset(sources, targets = [target]) 
                test_data =  Subset(dataset_haim, test_ind)
                test_loader = DataLoader(test_data, batch_size_val)

                part_of_hyperparameters = [target, None, i, miss_perc, seed, state_size, batch_size_train, encoder_hidd_units, decoder_hidd_units, dropout, epochs]
                test_haim = model_haim.test(test_loader, criterion)        
                checkpoint = torch.load(best_model_path_haim)  
                model_haim.load_state_dict(checkpoint['model_state_dict'])
                test_haim_best = model_haim.test(test_loader, criterion)
                results_haim_best = pd.DataFrame(columns = save_logs)
                test_haim_best_sngl = list(map(lambda metric: metric.numpy(), test_haim_best))
                row = ['haim'] + part_of_hyperparameters + test_haim_best_sngl
                results_haim_best.loc[0] = row
                if os.path.isfile(results_file_path):
                    results_haim_best.to_csv(results_file_path, mode='a', index=False, header=False)
                else:
                    results_haim_best.to_csv(results_file_path, mode='w', index=False)
            seed += 1

if __name__ == "__main__":
    main()
