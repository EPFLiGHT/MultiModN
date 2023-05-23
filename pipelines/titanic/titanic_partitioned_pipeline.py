import sys
import os
from os import path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../..")))

from tqdm.auto import trange
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from multimodn.multimodn import MultiModN
from multimodn.encoders import MLPEncoder
from multimodn.decoders import LogisticDecoder
from multimodn.history import MultiModNHistory
from datasets.titanic import TitanicDataset
from pipelines import utils
import torch.nn.functional as F
import pickle as pkl

def main():
    PIPELINE_NAME = utils.extract_pipeline_name(sys.argv[0])
    print('Running ̣{}...'.format(utils.get_display_name(PIPELINE_NAME)))
    args = utils.parse_args()

    torch.manual_seed(args.seed)

    features = ['Fare', 'Pclass', 'Age', 'Relatives', 'Embarked']
    partitions = [3, 2]
    targets = ['Survived']

    # Training / Validation / Testing ratios
    datasplit = (0.8, 0.2, 0)
    target_idx_to_balance = 0 # Balance 'Survived' during split
    
    # Batch size: set 0 for full batch
    batch_size = 32

    # Representation state size
    state_size = 5

    learning_rate = 0.01
    epochs = 300 if not args.epoch else args.epoch

    ##############################################################################
    ###### Create dataset and data loaders
    ##############################################################################
    # Get list of aligned datasets containg subsets of features according to partitions
    dataset = TitanicDataset(
        features,
        targets,
        dropna=True,
        std=True,
    ).partition_dataset(partitions)

    train_data, val_data, test_data = dataset.random_split(datasplit, args.seed, target_idx_to_balance)

    if batch_size == 0:
        batch_size_train = len(train_data)
        batch_size_val = len(val_data)
        batch_size_test = len(test_data)
    else:
        batch_size_train = batch_size
        batch_size_val = batch_size
        batch_size_test = batch_size

    train_loader = DataLoader(train_data, batch_size_train)
    val_loader = DataLoader(val_data, batch_size_val)

    ##############################################################################
    ###### Set encoder and decoders
    ##############################################################################
    encoders = [MLPEncoder(state_size, partition, (5, 5), F.relu) for partition in partitions]
    decoders = [LogisticDecoder(state_size) for _ in targets]

    model = MultiModN(state_size, encoders, decoders, 0.7, 0.3)

    optimizer = torch.optim.Adam(list(model.parameters()), learning_rate)

    criterion = CrossEntropyLoss()

    history = MultiModNHistory(targets)

    ##############################################################################
    ###### Train and Test model
    ##############################################################################
    for _ in trange(epochs):
        model.train_epoch(train_loader, optimizer, criterion, history)
        model.test(val_loader, criterion, history, tag='val')

    ##############################################################################
    ###### Store model and history
    ##############################################################################
    directory = o.join(o.dirname(os.path.realpath(__file__)), 'models')

    if args.save_model:
        if not o.exists(directory):
            os.makedirs(directory)
        model_path = o.join(directory, PIPELINE_NAME + '_model.pkl')
        pkl.dump(model, open(model_path, 'wb'))

    if args.save_history:
        if not o.exists(directory):
            os.makedirs(directory)
        history_path = o.join(directory, PIPELINE_NAME + '_history.pkl')
        pkl.dump(history, open(history_path, 'wb'))

    ##############################################################################
    ###### Save learning curves
    ##############################################################################
    if args.save_plot:
        directory = o.join(o.dirname(os.path.realpath(__file__)), 'plots')
        if not o.exists(directory):
            os.makedirs(directory)
        plot_path = o.join(directory, PIPELINE_NAME + '.png')

        targets_to_display = targets

        history.plot(plot_path, targets_to_display, show_state_change=False)

    ##############################################################################
    ###### Display results and save them
    ##############################################################################
    if args.save_results:
        directory = o.join(o.dirname(os.path.realpath(__file__)), 'results')
        if not o.exists(directory):
            os.makedirs(directory)
        results_path = o.join(directory, PIPELINE_NAME + '.csv')

        history.print_results()
        history.save_results(results_path)

if __name__ == "__main__":
    main()
