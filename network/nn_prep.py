import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# C0_05 IS IMG_4014
# C0_06 IS IMG_4014
# C0_07 is IMG_4014
# C0_09 IS IMG_4015
# C0_10 is IMG_4015 TWICE
# C0_11 is IMG_4015 SLOW
# C0_13 is IMG_4013 TWICE
# C0_14 is IMG_4010 TWICE
FRAMERATE = 30  # FPS in original video, should be 30

"""
Loading C005
"""
print("Processing C0_05")
with open("saves/C0_05", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4010.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)
combined_data = np.zeros(399+6)

for i in tqdm(range(length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

combined_data = combined_data[1:]


"""
Loading C006
"""
print("Processing C0_06")
with open("saves/C0_06", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4012.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range(length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C007
"""
print("Processing C0_07")

with open("saves/C0_07", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4013.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range(length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C009
"""
print("Processing C0_09")

with open("saves/C0_09", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4015.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range(length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C010
"""
print("Processing C0_10")
# First segment 0-38 seconds
# Second segment 1:57 to end
with open("saves/C0_10", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4012.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range(38*FRAMERATE)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

for i in tqdm(range((60+57)*FRAMERATE, length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C013
"""
print("Processing C0_13")
# First segment 0-1:11
# Second segment 1:26 to end
with open("saves/C0_13", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4011.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range((60+11)*FRAMERATE)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

for i in tqdm(range((60+26)*FRAMERATE, length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])


"""
Loading C014
"""
print("Processing C0_14")
# First segment 0-1:11
# Second segment 1:26 to end
with open("saves/C0_14", 'rb') as f:
    human = np.load(f)

df = pd.read_csv('datafiles/IMG_4012.dat', sep='\t',
                 header=None, engine='python')
robot = df.to_numpy(dtype=np.float32)

length = len(human)

for i in tqdm(range((60+12)*FRAMERATE)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

for i in tqdm(range((60+14)*FRAMERATE, length)):
    ratio = i/length
    index = (int)(np.around(ratio * len(robot)))
    curr_row = np.hstack([human[i], robot[index][1:7]])
    combined_data = np.vstack([combined_data, curr_row])

print(f"Final shape is {combined_data.shape}")

for i in range(len(combined_data)):
    for j in range(266):
        if j % 2 == 0:
            combined_data[i][j] = combined_data[i][j]/3840
        else:
            combined_data[i][j] = combined_data[i][j]/2160

combined_data[:, 399] = combined_data[:, 399]/(2*np.pi)
combined_data[:, 400] = combined_data[:, 400]/(2*np.pi)
combined_data[:, 401] = combined_data[:, 401]/(np.pi)
combined_data[:, 402] = combined_data[:, 402]/(2*np.pi)
combined_data[:, 403] = combined_data[:, 403]/(2*np.pi)
combined_data[:, 404] = combined_data[:, 404]/(2*np.pi)

with open(os.path.join("saves", "neural"), 'wb') as f:
    np.save(f, combined_data)
