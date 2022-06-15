import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import numpy as np

# C0_05 IS IMG_4014
# C0_06 IS IMG_4014
# C0_07 is IMG_4014
# C0_09 IS IMG_4015
# C0_10 is IMG_4015
# C0_13 is IMG_4013 TWICE
# C0_14 is IMG_4010 TWICE


n_epochs = 30
batch_size_train = 100
batch_size_test = 10
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)


class Move(Dataset):
    def __init__(self, filename) -> None:
        super().__init__()
        with open(filename, 'rb') as f:
            neural_data = np.load(f)

        self.x = torch.from_numpy(neural_data[:, :399])
        self.y = torch.from_numpy(neural_data[:, 399:])
        self.n_samples = len(neural_data)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)


dataset = Move("saves/neural")
train_data, test_data = torch.utils.data.random_split(dataset, [(int)(len(dataset)*0.8), len(
    dataset)-(int)(len(dataset)*0.8)], generator=torch.Generator().manual_seed(random_seed))
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(
    dataset=test_data, batch_size=batch_size_test, shuffle=False)

mse = nn.MSELoss()
model = Model(266, 500, 6)
optim = torch.optim.SGD(
    lr=learning_rate, params=model.parameters(), momentum=momentum)

device = torch.device('cpu')
total_steps = len(train_loader)
print(total_steps)
loss_arr = np.array([0, 0])
for epoch in range(n_epochs):
    for i, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data.float())
        loss = mse(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_arr = np.vstack(
            [loss_arr, [i+total_steps*epoch, loss.detach().numpy()]])
        if i % log_interval == 0:
            print(
                f"EPOCH: {epoch+1}/{n_epochs}, step {i}/{total_steps},loss = {loss}")
plt.plot(loss_arr[1:, 0], loss_arr[1:, 1], label="Correct Pairing")
plt.xlabel("Num. Steps")
plt.ylabel("Loss")
plt.title("Model Training Progress")

plt.legend()
plt.show()
total_tests = len(test_loader)
sum_err = 0
with torch.no_grad():
    for i, (data, label) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data.float())
        loss = mse(output, label)
        sum_err += loss

print(f"Average evaluation loss: {sum_err/total_tests}")

torch.save(model.state_dict(), "saves/NN_weights")
