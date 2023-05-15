import time
import torch
import pickle
import optuna
import warnings
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from optuna.trial import TrialState
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split

warnings.filterwarnings('ignore')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ----------------- Load CIFAR-10 Data -----------------------
    
transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
all_train = datasets.CIFAR10(root='./CIFAR_10', train=True, download=False, transform=transform)
trainset, valset = random_split(all_train, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

def stuff_data_cifar10(trial):
    batch_size = trial.suggest_categorical("batch_size", [4, 16, 64, 128, 256, 512])
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("CIFAR-10 > Training, Testing and Validation Data Loaded Succeddfully.")
    
    return train_dataloader, val_dataloader

# ----------------- Create Model -----------------------

def get_model(trial, classes):
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    model.classifier = None
    
    # all_params_count = len(list(model.parameters())) # = 211
    all_params_count = 211
    leave_layers = trial.suggest_int("leave_layers", 0, 100)
    effective_param = all_params_count - leave_layers
    
    for param in model.parameters():
        if effective_param: param.requires_grad = False
        else: break
        effective_param -= 1

    hidden_size = trial.suggest_int("hidden", 1000, 3000, 20)
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(in_features=1280, out_features=hidden_size),
        nn.Linear(in_features=hidden_size, out_features=classes)
    )
    
    return model

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


epochs = 15
classes = 10

def training(trial):
    loss_fcn = nn.CrossEntropyLoss()
    train_dataloader, val_dataloader = stuff_data_cifar10(trial)
    model = get_model(trial, classes).to(device)
    
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "NAdam", "RMSprop"])
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    lr = trial.suggest_categorical("lr", [0.0125, 0.00375, 0.075, 0.115, 0.15])
    weight_decay = trial.suggest_categorical("weight_decay", [0, 4e-8, 4e-7, 4e-6, 4e-5])
    optimizer = optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    all_train, all_val = [], []
    
    for epoch in range(epochs):
        start = time.time()
        model.train()
        torch.cuda.synchronize()

        pred, actual = [], []

        for feat, labels in train_dataloader:
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

        print(f"Epoch: {epoch + 1}/{epochs} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f} | Time Taken: {minute}m {sec}s")
        # print('-' * 100)
        
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return max(all_val)
        
        

if __name__ == "__main__":
    study_name = "Efficient_Transfer_Learning_V2"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True)
    study.optimize(training, n_trials=1000, timeout=None, n_jobs=8)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    df = study.trials_dataframe()
    df.to_csv('optuna_study_V2.csv')