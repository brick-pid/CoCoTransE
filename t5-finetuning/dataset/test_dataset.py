import json
import sys
sys.path.append("/home/sjw/ljb/code_translation")
from dataset.dataset_manager import create_it_dataset
from dataset.cj_dataset import create_cj_dataset

# dataset = create_it_dataset(tgt="cangjie")
# print(dataset)

dataset = create_cj_dataset()