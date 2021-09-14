from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import recall_score, f1_score
from torch import nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def compute_metrics(model:nn.Module, l_dataset:DataLoader, l_label_vocab:dict, device):
    all_predictions = list()
    all_labels = list()
    for indexed_elem in tqdm(l_dataset):
        indexed_in = indexed_elem["inputs"].to(device)
        indexed_labels = indexed_elem["outputs"].to(device)
        indexed_char = indexed_elem["char"].to(device)
        predictions = model(indexed_in, indexed_char)
        predictions = torch.argmax(predictions, -1).view(-1)
        labels = indexed_labels.view(-1)
        valid_indices = labels != 0
        
        valid_predictions = predictions[valid_indices]
        valid_labels = labels[valid_indices]
        
        all_predictions.extend(valid_predictions.tolist())
        all_labels.extend(valid_labels.tolist())

    # global precision. Does take class imbalance into account.
    micro_precision = sk_precision(all_labels, all_predictions, average="micro")
    # precision per class and arithmetic average of them. Does not take into account class imbalance.
    macro_precision = sk_precision(all_labels, all_predictions, average="macro",zero_division=0)
    per_class_precision = sk_precision(all_labels, all_predictions, labels = list(range(len(l_label_vocab))), average=None, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro",zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    
    return {"micro_precision":micro_precision,
            "macro_precision":macro_precision, 
            "per_class_precision":per_class_precision,
            "f1":f1,
            "recall":recall,
            "all_predictions":all_predictions,
            "all_labels":all_labels}