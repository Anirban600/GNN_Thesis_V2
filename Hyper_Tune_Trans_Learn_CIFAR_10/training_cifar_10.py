import time
import pickle
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import datasets, transforms

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ------------------------------  Prepare CIFAR-10  Dataset   -------------------------------

transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

training_data = datasets.CIFAR10(root='./CIFAR_10', train=True, download=False, transform=transform)
testset = datasets.CIFAR10(root='./CIFAR_10', train=False, download=False, transform=transform)

trainset, valset = random_split(training_data, [0.85, 0.15], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
val_dataloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)
test_dataloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
classes = 10

print("CIFAR-10 > Training, Validation and Testing Data Loaded Succeddfully.")

# ------------------------------  Prepare EfficientNet Model   -------------------------------

model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
model.classifier = None

all_params_count = len(list(model.parameters()))
leave_layers = 46
effective_param = all_params_count - leave_layers

for param in model.parameters():
    if effective_param: param.requires_grad = False
    else: break
    effective_param -= 1

model.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(output_size=1),
    nn.Flatten(),
    nn.Linear(in_features=1280, out_features=1180),
    nn.Linear(in_features=1180, out_features=classes)
)

# ------------------------------------  Useful Methods  -------------------------------------

def accuracy(logits, labels):
    if len(logits.shape) > 1: _, indices = torch.max(logits, dim=1)
    else: indices = logits
    correct = torch.sum(indices == labels)
    return correct.item() / len(labels)


def evaluate(model, dataloader):
    model.eval()
    pred, actual = [], []
    with torch.no_grad():
        for data in dataloader:
            feat = data[0].to(device)
            label = data[1].to(device)

            logits = model(feat)
            _, out = torch.max(logits, dim=1)

            pred.append(out)
            actual.append(label)
            
    pred = torch.cat(pred)
    actual = torch.cat(actual)
    return accuracy(pred, actual)

# -----------------------------------   Training Loop  ----------------------------------

epochs = 1000
loss_fcn = nn.CrossEntropyLoss()
model.to(device)
optimizer = optim.NAdam(model.parameters(), lr=0.00375, betas=(0.9, 0.999), weight_decay=0)
best_val_acc, patience = 0, 10

all_train, all_val = [], []

for epoch in range(epochs):
    start = time.time()
    model.train()
    torch.cuda.synchronize()
    
    pred, actual = [], []

    # for feat, labels in tqdm(train_dataloader, ascii=" ▖▘▝▗▚▞█"):
    for feat, labels in tqdm(train_dataloader, ascii=" >="):
        feat = feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(feat)
        loss = loss_fcn(logits, labels)
        loss.backward()
        optimizer.step()
        
        _, out = torch.max(logits, dim=1)

        pred.append(out)
        actual.append(labels)
        
    pred = torch.cat(pred)
    actual = torch.cat(actual)
    
    train_acc = accuracy(pred, actual)
    val_acc = evaluate(model, val_dataloader)
    time_taken = time.time() - start
    minute, sec = list(map(int, divmod(time_taken, 60)))
    all_train.append(train_acc)
    all_val.append(val_acc)
    
    if val_acc > best_val_acc:
        with open("best_model.pickle", "wb") as f:
            pickle.dump(model, f)
        patience = 0
        best_val_acc = val_acc
    else: patience += 1
    
    print(f"Epoch: {epoch + 1}/{epochs} | Train Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f} | Time Taken: {minute}m {sec}s | Patience: {patience}/10")
    print('-' * 100)
    
    if patience == 10: break



with open("best_model.pickle", "rb") as f:
    model = pickle.load(f)
    
print('=' * 100)
train_acc = evaluate(model, train_dataloader)
val_acc = evaluate(model, val_dataloader)
test_acc = evaluate(model, test_dataloader)
print(f"Train Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f} | Test Accuracy: {test_acc:.4f}")
print('=' * 100)


with open("Train_Test_Data_V2.pickle", "wb") as f:
    pickle.dump((all_train, all_val), f)

with open("Final_model_V2.pickle", "wb") as f:
    pickle.dump(model, f)
    
print("Model and Results Saved Successfully...")