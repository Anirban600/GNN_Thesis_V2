import os
import tqdm
import time
import pickle
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF

import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler

from pick_train_mask import LabelBalancedSampler

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y

def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys), task='multiclass', num_classes=47)

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
        return MF.accuracy(pred, label, task='multiclass', num_classes=47)

def run(args, device, g, dataset):
    all_val_acc, all_test_acc, all_time, all_epochs = [], [], [], []
    
    labels = g.ndata['label'].to(device)
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>  Choose Changes  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    
    with open("../Correct_And_Smooth_OGB_Products/predicted_labels.pickle", "rb") as f:
        y_pred = pickle.load(f).flatten().to(device)
    
    y_pred[train_mask.bool()] = labels[train_mask.bool()]
    y_pred[val_mask.bool()] = labels[val_mask.bool()]
    
    
    # Calculate Choose Probability
    class_count = np.vstack(np.unique(y_pred.cpu().numpy(), return_counts=True)).T
    add = [[i, 0] for i in range(47) if i not in class_count[:, 0].tolist()]
    if add: class_count = np.vstack((class_count, np.array(add)))
    class_count = class_count[class_count[:, 0].argsort()]
    class_probs = 1 / np.log(10 + class_count[:, 1]) ** 2
    
    src = g.edges()[0]
    src_label = y_pred[src]
    src_prob = class_probs[src_label.cpu().numpy()]
    
    g.edata["prob"] = torch.tensor(src_prob).to(device)
    
    sampler_choose = NeighborSampler([10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
                              prob="prob",
                              prefetch_node_feats=['feat'],
                              prefetch_labels=['label'])
    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
    use_uva = (args.mode == 'mixed')
    
    sampler_default = NeighborSampler([10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
                          prefetch_node_feats=['feat'],
                          prefetch_labels=['label'])
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>  Pick code sneppit  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    A = g.adj(scipy_fmt='coo')
    lbs = LabelBalancedSampler(A, labels.cpu(), train_mask.cpu().numpy())
    probs = lbs.all_probabilities()
    
    print("Pick Probabilities Calculation Done...")
    
    def give_train_data():
        sample_idx = np.random.choice(train_idx.cpu(), size=20000, replace=False, p=probs)
        sample_idx = torch.tensor(sample_idx, device=device)
        
        train_dataloader = DataLoader(g, sample_idx, sampler_choose, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)
        return train_dataloader
    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    val_dataloader = DataLoader(g, val_idx, sampler_default, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)
    
    
    for iteration in range(args.n_iter):
        t_iteration = time.time()
        best_val_acc = 0
        patience = 0
        
        print(f"{'-'*40}  Iteration {iteration + 1}  {'-'*40}")
    
        model = SAGE(in_size, 256, out_size).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        
        for epoch in range(100000):
            t_epoch = time.time()
            model.train()
            total_loss = 0
            train_dataloader = give_train_data()
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label']
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                
            acc = evaluate(model, g, val_dataloader)
            
            if acc > best_val_acc:
                best_val_acc = acc
                patience = 0
                torch.save(model.state_dict(), "best_model.pt")
            else: patience += 1
                
            print(f"Iteration : {iteration + 1} | Epoch :{epoch : 3d} | Loss : {total_loss / (it+1):.4f} | Val Accuracy : {acc.item():.4f} | Epoch Time : {int(time.time() - t_epoch)}s | Patience {patience: 3d}/{args.patience}")
            
            if patience == args.patience:
                all_epochs.append(epoch - patience)
                break
        
        
        model.load_state_dict(torch.load("best_model.pt"))
        val_acc = evaluate(model, g, val_dataloader)
        test_acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
        total_time = int(time.time() - t_iteration)
        
        all_val_acc.append(val_acc.item())
        all_test_acc.append(test_acc.item())
        all_time.append(total_time)
        
        print("="*80)
        print(f"Val Accuracy : {val_acc.item():.4f} | Test Accuracy {test_acc.item():.4f} | Iteration Time : {total_time}s")
        print("="*80)

    all_val_acc = np.array(all_val_acc)
    all_test_acc = np.array(all_test_acc)
    all_time = np.array(all_time)
    all_epochs = np.array(all_epochs)
    
    print()
    print(f"{'='*40}  Final Result {'='*40}")
    print(f"Total Iterations : {args.n_iter}")
    print(f"All Val Acc : {all_val_acc.round(4).tolist()}")
    print(f"All Test Acc : {all_test_acc.round(4).tolist()}")
    print(f"Validation Accuracy : {all_val_acc.mean():.4f} ± {all_val_acc.std():.3f}")
    print(f"Test Accuracy : {all_test_acc.mean():.4f} ± {all_test_acc.std():.3f}")
    print(f"Average Time : {all_time.mean():.3f}s")
    print(f"Average Epochs : {all_epochs.mean():.3f}s")
    print("="*90)
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='puregpu', choices=['cpu', 'mixed', 'puregpu'])
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    
    print('Loading OGB Products data...')
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]
    g = g.to(device)

    run(args, device, g, dataset)
