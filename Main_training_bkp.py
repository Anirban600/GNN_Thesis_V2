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

# ----------------- Load Data -----------------------
    
transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

trainset = datasets.CIFAR10(root='./CIFAR_10', train=True, download=False, transform=transform)
testset = datasets.CIFAR10(root='./CIFAR_10', train=False, download=False, transform=transform)

train_dataloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
test_dataloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

classes = 10

print("CIFAR-10 > Training and Testing Data Loaded Succeddfully.")


# class MyDataset(Dataset):
#     def __init__(self, filename):
#         with open(filename, "rb") as f:
#             data = pickle.load(f)
#         self._data = data
#
#     def __getitem__(self, idx):
#         return self._data[idx]
#
#     def __len__(self):
#         return len(self._data)    
#   
#
# def stuff_data_bird():
#     train_root = "/media/extra_storage/anirban/birdsnap/dataloaders/train/"
#     test_root = "/media/extra_storage/anirban/birdsnap/dataloaders/test.pickle"
#     val_root = "/media/extra_storage/anirban/birdsnap/dataloaders/val.pickle"
#
#     dataset_list = []
#     for file in tqdm(os.listdir(train_root)):
#         file_path = os.path.join(train_root, file)
#         dataset_list.append(MyDataset(file_path))
#
#     concat_data = ConcatDataset(dataset_list)
#
#     with open(test_root, 'rb') as f: test_data = pickle.load(f)
#     with open(val_root, 'rb') as f: val_data = pickle.load(f)
#
#     train_dataloader = DataLoader(concat_data, batch_size=256, shuffle=True, drop_last=False)
#     test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False, drop_last=False)
#     val_dataloader = DataLoader(val_data, batch_size=256, shuffle=False, drop_last=False)
#
#     print("BirdSnap > Training, Testing and Validation Data Loaded Succeddfully.")
#
#     return train_dataloader, test_dataloader, val_dataloader

# ----------------- Create Model -----------------------

model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
model.classifier = None

# freezing weights
all_params_count = len(list(model.parameters()))
leave_layers = 51
effective_param = all_params_count - leave_layers

for param in model.parameters():
    if effective_param: param.requires_grad = False
    else: break
    effective_param -= 1

model.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(output_size=1),
    nn.Flatten(),
    nn.Linear(in_features=1280, out_features=1107),
    nn.Linear(in_features=1107, out_features=classes)
)

# --------------------  Training Loop  ----------------------

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



epochs = 100

loss_fcn = nn.CrossEntropyLoss()
model.to(device)
optimizer = optim.NAdam(model.parameters(), lr=0.00375, betas=(0.9, 0.999), weight_decay=4e-08)

all_train, all_test = [], []

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
    test_acc = evaluate(model, test_dataloader)
    time_taken = time.time() - start
    minute, sec = list(map(int, divmod(time_taken, 60)))
    all_train.append(train_acc)
    all_test.append(test_acc)

    print(f"Epoch: {epoch + 1}/{epochs} | Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f} | Time Taken: {minute}m {sec}s")
    print('-' * 100)

with open("Train_Test_Data.pickle", "wb") as f:
    pickle.dump((all_train, all_test), f)