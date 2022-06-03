import torch
import torch.nn as nn

import numpy as np

FRAMERATE = 30


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size//2)
        self.l3 = nn.Linear(hidden_size//2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(1, self.hidden_size)
        c0 = torch.zeros(1, self.hidden_size)

        outputs = []
        for input_t in x.split(1, dim=0):
            h0, c0 = self.lstm(input_t, (h0, c0))
            h0 = self.relu(h0)
            output = self.relu(self.l2(h0))
            outputs.append(self.l3(output))

        return torch.cat(outputs, dim=0)


model = Model(266, 250, 6)
model.load_state_dict(torch.load("saves/NN_weights"))

with open("saves/C0_07", 'rb') as f:
    test_data = np.load(f)

for i in range(len(test_data)):
    for j in range(266):
        if j % 2 == 0:
            test_data[i][j] = test_data[i][j]/3840
        else:
            test_data[i][j] = test_data[i][j]/2160

# output = np.zeros(6)
# for vector in test_data:
#     in_tensor = torch.from_numpy(vector)
#     out_tensor = model(in_tensor.float()).detach().numpy()
#     output = np.vstack([output, out_tensor])

in_tensor = torch.from_numpy(test_data)
output = model(in_tensor.float()).detach().numpy()

output = output[1:]
time_arr = np.linspace(0, len(output)/FRAMERATE, len(output))
final = np.hstack([time_arr.reshape(len(output), 1), output])
with open("saves/NN_out", "wb") as f:
    np.save(f, final)

np.savetxt("NN_out.dat", final, fmt='%10.5f', delimiter="\t")
