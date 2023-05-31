import os
from os import path as o
import numpy as np
import sys
from typing import Optional, List
from datasets import PartitionDataset, FeatureWiseDataset
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import StandardScaler
from torch import Tensor, Generator
import pandas as pd
from torch._utils import _accumulate as accumulate
from typing import Optional, Tuple, List, Union
import torch
from collections import defaultdict

embed_path = 
fname = embed_path + 'cxr_ic_fusion_1103.csv'

source_names = ['de', 'vd',  'vmd', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech', 'n_rad']

source_size = [ 6, 1024, 1024, 99, 242, 110, 768, 768, 768]

source_dict = dict(zip(source_names, source_size))

base_path =  o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../.."))

def mimic_get_overlap_pathologies_data(fname, sources, targets, put_none, indices_to_nan,  features_to_nan,):    
    pathologies = '_'.join(targets)        
    data_dir = os.path.join(base_path, 'datasets/mimic', pathologies) 
    source_spec = '_'.join(sources)   
    features, partitions, viz_features = [], [], []
    if not os.path.exists(data_dir):        
        os.makedirs(data_dir)     
    data_file_path = os.path.join(data_dir, source_spec)
    if not os.path.exists(data_file_path):
        os.makedirs(data_file_path)           
        df = pd.read_csv(fname, on_bad_lines='skip')
        df = df.drop_duplicates(subset = ['img_id', 'img_charttime'])        
        for target in targets:
            df = df[df[target].isin([0,1])]     
        agg_label = 'Agg'
        df['buff'] = df[targets].sum(axis=1)
        df[agg_label] = df.apply(lambda x: 1 if x['buff'] > 1 else 0, axis = 1)
        for source in sources: 
            if source.lower() == 'de':
                df = pd.get_dummies(df, columns = ['de_1', 'de_2', 'de_3', 'de_4', 'de_5'], drop_first=True)
                demo_features = list(df.filter(regex = f'de_(\d)+(_\d)*').columns)
                features = features + demo_features
                partitions = partitions + [len(demo_features)]
            else: 
                buff_features = [source +  '_' +  str(i) for i in range(source_dict[source])]
                features = features + buff_features
                partitions = partitions + [len(buff_features)]
                if source.lower().startswith('v'):
                    viz_features = viz_features + buff_features      
        labels = df[targets]    
        sources_with_index = list(map(lambda x: x+'_(\d)+(_\d)*', sources))
        filter_expression = '|'.join(sources_with_index)           
        data = df.filter(regex = filter_expression)                 
        data = data[features] # keep features in the order
        data_full = df[features + targets + ['haim_id']]    
        data_full.to_csv(os.path.join(data_file_path, 'data.csv'), index=False)       
        patient_labels = df.groupby(['haim_id']).agg(label_count = (agg_label,'count'),
                                                                label_ones = (agg_label, 'sum')).reset_index()                                                                                                                                
        patient_labels['label'] = patient_labels.apply(lambda row: int(row['label_ones'] >=  row['label_count']/2), axis=1) 
        patient_labels.to_csv(os.path.join(data_file_path,'how_to_split.csv'), index=False)
    else:
        data_full = pd.read_csv(data_file_path+'/data.csv') 
        for source in sources:         
            if source.lower() == 'de':                
                demo_features = list(data_full.filter(regex = f'de_(\d)+(_\d)*').columns)
                features = features + demo_features
                partitions = partitions + [len(demo_features)]
            else: 
                buff_features = [source +  '_' +  str(i) for i in range(source_dict[source])]
                features = features + buff_features
                partitions = partitions + [len(buff_features)]
                if source.lower().startswith('v'):
                    viz_features = viz_features + buff_features  
        data_full = pd.read_csv(data_file_path+'/data.csv') 
        data = data_full[features]        
        labels = data_full[targets]        
    if put_none:         
        if features_to_nan == 'demo':
            features_to_nan = demo_features 
        none_string = [None] * len(features_to_nan)            
        data.loc[indices_to_nan, features_to_nan] = none_string     
    return data, labels, features, partitions

def mimic_get_nips_pathology_data(fname, targets, sources, put_none, indices_to_nan,  features_to_nan):  
    data_dir = os.path.join(base_path, 'datasets/mimic', targets[0])
    source_spec = '_'.join(sources)
    # path to the pre-processed embedding data containing only valid samples for ecm and cardiomegaly
    fname = os.path.join(base_path, 'datasets/mimic', 'Enlarged Cardiomediastinum_Cardiomegaly', source_spec, 'data.csv')   
    features, partitions, viz_features = [], [], []        
    if not os.path.exists(data_dir):
        os.makedirs(data_dir) 
    data_file_path = os.path.join(data_dir, source_spec)
    if not os.path.exists(data_file_path):
        os.makedirs(data_file_path)                               
        df = pd.read_csv(fname, on_bad_lines='skip')          
        for source in sources: 
            if source.lower() == 'de':
                demo_features = list(df.filter(regex = f'de_(\d)+(_\d)*').columns)
                features = features + demo_features
                partitions = partitions + [len(demo_features)]
            else: 
                buff_features = [source +  '_' +  str(i) for i in range(source_dict[source])]
                features = features + buff_features
                partitions = partitions + [len(buff_features)]
                if source.lower().startswith('v'):
                    viz_features = viz_features + buff_features        
        labels = df[targets]    
        sources_with_index = list(map(lambda x: x+'_(\d)+(_\d)*', sources))
        filter_expression = '|'.join(sources_with_index)           
        data = df.filter(regex = filter_expression)                 
        data = data[features] # keep features in the order
        data_full = df[features + targets + ['haim_id']]
    
        data_full.to_csv(os.path.join(data_file_path, 'data.csv'), index=False) 
        
        patient_labels = df.groupby(['haim_id']).agg(label_count = (targets[0],'count'),
                                                                label_ones = (targets[0], 'sum')).reset_index()                                                                                                                                  
        patient_labels['label'] = patient_labels.apply(lambda row: int(row['label_ones'] >=  row['label_count']/2), axis=1)

        patient_labels.to_csv(os.path.join(data_file_path,'how_to_split.csv'), index=False)
    else:
        data_full = pd.read_csv(data_file_path+'/data.csv') 
        for source in sources:         
            if source.lower() == 'de':                
                demo_features = list(data_full.filter(regex = f'de_(\d)+(_\d)*').columns)
                features = features + demo_features
                partitions = partitions + [len(demo_features)]
            else: 
                buff_features = [source +  '_' +  str(i) for i in range(source_dict[source])]
                features = features + buff_features
                partitions = partitions + [len(buff_features)]
                if source.lower().startswith('v'):
                    viz_features = viz_features + buff_features  
        data_full = pd.read_csv(data_file_path+'/data.csv') 
        data = data_full[features]        
        labels = data_full[targets]        
    if put_none:         
        if features_to_nan == 'demo':
            features_to_nan = demo_features
        none_string = [None] * len(features_to_nan)         
        data.loc[indices_to_nan, features_to_nan] = none_string      
    return data, labels, features, partitions 
 

class MIMICDataset(Dataset):
    
    def __init__(
        self,                  
        sources: List[str],
        targets: List[str] = [],
        dropna: bool = False,          
        nanfill: bool = False,
        std: bool = True,
        put_none: bool = False,
        indices_to_nan: List[int] = [], 
        features_to_nan: List[str] = [],             
    ):
        if len(targets) == 1:
            data, labels, features, partitions = mimic_get_nips_pathology_data(fname, targets, sources, put_none, indices_to_nan, features_to_nan, )
        else: 
            data, labels, features, partitions = mimic_get_overlap_pathologies_data(fname, sources, targets, put_none, indices_to_nan,  features_to_nan,)         
            
        self.y = labels.values

        if dropna:
            data.dropna(inplace=True)            
        if std:
            data = pd.DataFrame(StandardScaler().fit_transform(data[features]), columns=features, index=data.index)
            
        if nanfill:      
            print('Number of samples with missing values = ', sum(data.isnull().any(axis=1).values))
            data.fillna(0, inplace = True)
        self.X = data.values
        self.partitions = partitions    


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return Tensor(self.X[idx]), self.y[idx]    

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
            lengths[0] += len(indices) - sum(lengths)

            for i, (offset, length) in enumerate(zip(accumulate(lengths), lengths)):
                splitted_indices[i] = (
                        splitted_indices[i] + indices[offset - length: offset]
                )
        return splitted_indices
    
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