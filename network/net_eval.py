import torch
import torch.nn as nn

import numpy as np

FRAMERATE = 30


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


model = Model(399, 500, 6)
model.load_state_dict(torch.load("saves/NN_weights"))

with open("saves/C0_07", 'rb') as f:
    test_data = np.load(f)

for i in range(len(test_data)):
    for j in range(266):
        if j % 2 == 0:
            test_data[i][j] = test_data[i][j]/3840
        else:
            test_data[i][j] = test_data[i][j]/2160

output = np.zeros(6)
for vector in test_data:
    in_tensor = torch.from_numpy(vector)
    out_tensor = model(in_tensor.float()).detach().numpy()
    output = np.vstack([output, out_tensor])

output = output[1:]
output[:, 0] *= (np.pi * 2)
output[:, 1] *= (np.pi * 2)
output[:, 2] *= (np.pi)
output[:, 3] *= (np.pi * 2)
output[:, 4] *= (np.pi * 2)
output[:, 5] *= (np.pi * 2)
time_arr = np.linspace(0, len(output)/FRAMERATE, len(output))
final = np.hstack([time_arr.reshape(len(output), 1), output])


with open("saves/NN_out", "wb") as f:
    np.save(f, final)

np.savetxt("NN_out.dat", final, fmt='%10.5f', delimiter="\t")
