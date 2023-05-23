import argparse
import logging
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline for MultiModN')

    parser.add_argument(
        '-e', '--epoch',
        dest='epoch',
        help='Number of epochs for MultiModN training',
        required=False,
        type=int,
    )

    parser.add_argument(
        '-s', '--seed',
        dest='seed',
        help='Set random seed',
        default=0,
        required=False,
        type=int,
    )

    parser.add_argument(
        '-m', '--save_model',
        dest='save_model',
        help='Whether to save model',
        default=True,
        required=False,
        type=string_to_bool,
    )

    parser.add_argument(
        '-y', '--save_history',
        dest='save_history',
        help='Whether to save history',
        default=True,
        required=False,
        type=string_to_bool,
    )

    parser.add_argument(
        '-p', '--save_plot',
        dest='save_plot',
        help='Whether to save learning curves',
        default=True,
        required=False,
        type=string_to_bool,
    )

    parser.add_argument(
        '-r', '--save_results',
        dest='save_results',
        help='Whether to save results',
        default=True,
        required=False,
        type=string_to_bool,
    )

    return parser.parse_args()


def string_to_bool(str):
    if isinstance(str, bool):
        return str
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_pipeline_name(filename):
    pipeline_name = filename.split('/')[-1].split('.')[0].replace('_pipeline', '')

    return pipeline_name


def get_display_name(name: str):
    display_name = name.replace('_', ' ').capitalize()

    return display_name


def _create_log_path(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fname = Path("logs") / f"{name}.log"
    _create_log_path(fname.parent)
    fh = logging.FileHandler(filename=fname)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger