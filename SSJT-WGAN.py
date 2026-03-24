import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ones
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
from torch import optim, nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import torchvision.utils as vutils
import warnings
warnings.filterwarnings('ignore')

class_num = 8
data_dim = 64
nz = 100
N_size = 64
input_dim = 64
lambda_gp = 10
select_number = 50
# epochs = 10000
epochs = 5000
batch_size = 40
max_test_accuracy = 0.0
max_test_f1 = 0.0
advWeight = 0.1
threshold = 0.9
datasets = "chiller"
peizhi = '0.1+0.9'
seed = 36

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
scaler = MinMaxScaler()
label_encoder = LabelEncoder()


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=8):
        super(ResNet1D, self).__init__()
        self.in_planes = 16
        self.embDim = 128 * block.expansion
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, out.size(2))
        emb = out.view(out.size(0), -1)
        out = self.linear(emb)
        return out

    def get_embedding_dim(self):
        return self.embDim


def ResNet18_1D():
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2])


class WGAN_GP_generator(nn.Module):
    def __init__(self, nz, N_size):
        super(WGAN_GP_generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, N_size),
            nn.ReLU(True)
        )

    def forward(self, input):
        return self.main(input)


class WGAN_GP_discriminator(nn.Module):
    def __init__(self, input_dim):
        super(WGAN_GP_discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, input):
        return self.main(input)


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = interpolates.view(interpolates.size(0), -1)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def load_original_data(D_name):
    if D_name == 'chiller':
        file = r"data\chiller\chiller_select_" + str(select_number) + ".csv"
        data_ = pd.read_csv(file, sep=',', header='infer')
        test_data = pd.read_csv(r'data\chiller\chiller_test_300.csv', sep=',', header='infer')
    elif D_name == 'AHU':
        file = r"data\AHU\AHU_select_" + str(select_number) + ".csv"
        data_ = pd.read_csv(file, sep=',', header='infer')
        test_data = pd.read_csv(r'data\AHU\AHU_test_300.csv', sep=',', header='infer')
    elif D_name == 'SZVAV':
        file = r"data\SZVAV\SZVAV_select_" + str(select_number) + ".csv"
        data_ = pd.read_csv(file, sep=',', header='infer')
        test_data = pd.read_csv(r'data\SZVAV\SZVAV_test_300.csv', sep=',', header='infer')
    elif D_name == 'SZCAV':
        file = r"data\SZCAV\SZCAV_select_" + str(select_number) + ".csv"
        data_ = pd.read_csv(file, sep=',', header='infer')
        test_data = pd.read_csv(r'data\SZCAV\SZCAV_test_150.csv', sep=',', header='infer')

    return data_, test_data


def load_select_data(data, select_num, save=True):
    size = select_num * class_num
    F1 = data[data['fault type'] == 'F1'].sample(n=select_num)
    F2 = data[data['fault type'] == 'F2'].sample(n=select_num)
    F3 = data[data['fault type'] == 'F3'].sample(n=select_num)
    F4 = data[data['fault type'] == 'F4'].sample(n=select_num)
    F5 = data[data['fault type'] == 'F5'].sample(n=select_num)
    F6 = data[data['fault type'] == 'F6'].sample(n=select_num)
    F7 = data[data['fault type'] == 'F7'].sample(n=select_num)
    Normal = data[data['fault type'] == 'Normal'].sample(n=select_num)

    if datasets == 'SZCAV':
        F8 = data[data['fault type'] == 'F8'].sample(n=select_num)
        F9 = data[data['fault type'] == 'F9'].sample(n=select_num)
        F10 = data[data['fault type'] == 'F10'].sample(n=select_num)
        F11 = data[data['fault type'] == 'F11'].sample(n=select_num)
        F12 = data[data['fault type'] == 'F12'].sample(n=select_num)
        F13 = data[data['fault type'] == 'F13'].sample(n=select_num)
        F14 = data[data['fault type'] == 'F14'].sample(n=select_num)
        sdata = pd.concat([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, Normal], ignore_index=True)
    else:
        sdata = pd.concat([F1, F2, F3, F4, F5, F6, F7, Normal], ignore_index=True)

    X = sdata.iloc[:, data.columns != "fault type"]
    labels = sdata.iloc[:, data.columns == "fault type"]

    return (X, labels)


def reset_models():
    global netG, netD, netC, optD, optG, optC

    netG = WGAN_GP_generator(nz=nz, N_size=N_size)
    netD = WGAN_GP_discriminator(input_dim=input_dim)
    netC = ResNet18_1D()

    netG.to(device)
    netD.to(device)
    netC.to(device)

    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-3)
    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optC = optim.Adam(netC.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-3)


def validate():
    global test_accuracy, test_f1
    netC.eval()
    all_predicted = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = netC(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = (correct / total) * 100
    f1 = f1_score(all_labels, all_predicted, average='weighted')
    print(f'Accuracy of the network on the 300 test samples: {accuracy} %')
    print(f'F1 Score of the network on the 300 test images: {f1:.4f}')
    text = f"The finally Test F1 Score of the 300 test samples each class: {f1}\n"
    file.write(text)
    cm = confusion_matrix(all_labels, all_predicted)
    text_cm = f"Confusion Matrix for LAST EPOCH:\n{cm}\n"
    file_report.write(text_cm)
    report = classification_report(all_labels, all_predicted)
    text_report = f"Classification Report for LAST EPOCH:\n{report}\n"
    file_report.write(text_report)
    max_test_accuracy = accuracy
    max_test_f1 = f1
    netC.train()


def train():
    global test_accuracy, test_f1, epoch
    text = f"Datasize:50\n"
    file.write(text)
    csv_file_path = f'\\result\\{datasets}\\{select_number}\\{peizhi}++losses_{iepoch}.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    csv_file = open(csv_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Discriminator Loss', 'Generator Loss', 'Classifier Loss'])
    for epoch in range(epochs):
        netC.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        for i, data in enumerate(subTrainLoader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            tmpBatchSize = len(labels)
            for _ in range(5):
                r = torch.randn(tmpBatchSize, nz, device=device)
                fakeImageBatch = netG(r)
                predictionsReal = netD(inputs)
                lossDiscriminatorReal = -torch.mean(predictionsReal)
                predictionsFake = netD(fakeImageBatch)
                lossDiscriminatorFake = torch.mean(predictionsFake)
                gradient_penalty = compute_gradient_penalty(netD, inputs, fakeImageBatch)
                lossDiscriminator = lossDiscriminatorReal + lossDiscriminatorFake + lambda_gp * gradient_penalty
                optD.zero_grad()
                lossDiscriminator.backward(retain_graph=True)
                optD.step()
            r = torch.randn(tmpBatchSize, nz, device=device)
            fakeImageBatch = netG(r)
            predictionsFake = netD(fakeImageBatch)
            lossGenerator = -torch.mean(predictionsFake)

            inputs_c = inputs.detach().unsqueeze(1)
            predictions = netC(inputs_c)
            realClassifierLoss = criterion(predictions, labels)
            optC.zero_grad()
            realClassifierLoss.backward(retain_graph=True)
            optC.step()
            r = torch.randn(500, nz, device=device)
            fakeImageBatch_all = netG(r)
            fakeImageBatch_reshaped = fakeImageBatch_all.detach().unsqueeze(1)
            predictionsFake = netC(fakeImageBatch_reshaped.detach())
            predictedLabels = torch.argmax(predictionsFake, 1)
            confidenceThresh = threshold
            probs = F.softmax(predictionsFake, dim=1)
            mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for i in range(len(probs))])
            toKeep = mostLikelyProbs > confidenceThresh
            if sum(toKeep) != 0:
                fakeClassifierLoss = criterion(predictionsFake[toKeep], predictedLabels[toKeep]) * advWeight
                lossGenerator_all = lossGenerator + fakeClassifierLoss
                optG.zero_grad()
                lossGenerator_all.backward(retain_graph=True)
                optG.step()
                optC.zero_grad()
                fakeClassifierLoss.backward()
                optC.step()
            optD.zero_grad()
            optG.zero_grad()
            optC.zero_grad()
            generatorLosses.append(lossGenerator.item())
            discriminatorLosses.append(lossDiscriminator.item())
            classifierLosses.append(realClassifierLoss.item())
            if (i % 100 == 0):
                netC.eval()
                _, predicted = torch.max(predictions, 1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels.data).sum().item()
                train_accuracy = 100 * correct_train / total_train
                text = f"Train Accuracy: {train_accuracy}"
                file.write(text + '\n')
                netC.train()
        csv_writer.writerow([epoch, discriminatorLosses[-1], generatorLosses[-1], classifierLosses[-1]])
        print(f"Epoch {epoch} Complete")
        print(f'Accuracy of the train dataset: {train_accuracy} %')
        validate()
    csv_file.close()


if __name__ == "__main__":
    traindataset, testdataset = load_original_data(datasets)
    (X_train, y_train) = load_select_data(traindataset, select_number)
    X_train = scaler.fit_transform(X_train)
    y_train = label_encoder.fit_transform(y_train)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    subTrainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    (X_test, y_test) = load_select_data(testdataset, 300)
    X_test = scaler.transform(X_test)
    y_test = label_encoder.fit_transform(y_test)
    X_test = np.expand_dims(X_test, axis=1)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generatorLosses = []
    discriminatorLosses = []
    classifierLosses = []
    loss = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    reset_models()
    max_test_accuracy = 0.0
    max_test_f1 = 0.0
    os.makedirs(fr"result\{datasets}\{select_number}", exist_ok=True)
    file = open(fr"result\{datasets}\{select_number}\External Classifier.txt", "w")
    file_report = open(fr"result\{datasets}\{select_number}\report.txt", "w")
    stime = time.time()
    train()
    etime = time.time()
    file.write(f"testing time: {etime - stime:.4f}\n")
    file.write(f"FINAL Epoch Test Accuracy: {test_accuracy:.4f}%\n")
    file.write(f"FINAL Epoch Test F1 Score: {test_f1:.4f}\n")
    file.close()
    file_report.close()
    
