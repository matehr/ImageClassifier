import argparse

def _get_common_args():
    parser = argparse.ArgumentParser(description="Collect user input")
    parser.add_argument("data_dir", type=str, metavar='data_dir')
    parser.add_argument('--arch', default='densenet')
    parser.add_argument('--gpu', const=True, default=False, action='store_const')
    
    return parser

def get_train_args():
    parser = _get_common_args()
    parser.add_argument('--save_dir', default='.')
    parser.add_argument('--arch', default='densenet')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1)
    return parser.parse_args()

def get_predict_args():
    parser = _get_common_args()
    parser.add_argument('checkpoint', type=str, metavar='checkpoint')
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--category_names', default='cat_to_name.json')
    return parser.parse_args()