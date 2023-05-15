import os
import dgl
import copy
import torch
import pickle
import argparse
import torch.optim as optim
import torch.nn.functional as F
from model import CorrectAndSmooth
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def evaluate(y_pred, y_true, mask):
    a = y_pred[mask].flatten()
    b = y_true[mask].flatten()
    return torch.sum(a == b) / len(a)


def main():
    # check cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    with open("Graph_ResNet50_Food.pickle", "rb") as f:
        g = pickle.load(f)

    feat = g.ndata["feat"]
    labels = g.ndata["label"]
    feat = (feat - feat.mean(0)) / feat.std(0)
    g.ndata["feat"] = feat
    g = g.to(device)
    feats = g.ndata["feat"]
    labels = labels.to(device)
    train_mask = g.ndata["train_mask"].to(device)
    val_mask = g.ndata["val_mask"].to(device)
    test_mask = g.ndata["test_mask"].to(device)
    n_features = feats.shape[1]
    n_classes = 101
    y_soft = g.ndata["y_soft"]
    y_pred = g.ndata["y_hard"]

    print("--------------- Before ------------------")
    valid_acc = evaluate(y_pred, labels, val_mask)
    test_acc = evaluate(y_pred, labels, test_mask)
    print(f"Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}")
    
    
    print("---------- Correct & Smoothing ----------")
    cs = CorrectAndSmooth(
        num_correction_layers=args.num_correction_layers,
        correction_alpha=args.correction_alpha,
        correction_adj=args.correction_adj,
        num_smoothing_layers=args.num_smoothing_layers,
        smoothing_alpha=args.smoothing_alpha,
        smoothing_adj=args.smoothing_adj,
        autoscale=args.autoscale,
        scale=args.scale,
    )

    y_soft = cs.correct(g, y_soft, labels[train_mask], train_mask)
    y_soft = cs.smooth(g, y_soft, labels[train_mask], train_mask)
    y_pred = y_soft.argmax(dim=-1, keepdim=True)
    valid_acc = evaluate(y_pred, labels, val_mask)
    test_acc = evaluate(y_pred, labels, test_mask)
    print(f"Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    """
    Correct & Smoothing Hyperparameters
    """
    parser = argparse.ArgumentParser(description="Base predictor(C&S)")
    # C & S
    parser.add_argument("--num-correction-layers", type=int, default=50)
    parser.add_argument("--correction-alpha", type=float, default=0.979)
    parser.add_argument("--correction-adj", type=str, default="DAD")
    parser.add_argument("--num-smoothing-layers", type=int, default=50)
    parser.add_argument("--smoothing-alpha", type=float, default=0.756)
    parser.add_argument("--smoothing-adj", type=str, default="DAD")
    parser.add_argument("--autoscale", action="store_true")
    parser.add_argument("--scale", type=float, default=20.0)

    args = parser.parse_args()
    # print(args)

    main()
