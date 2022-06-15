# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from tqdm import tqdm

# %%
TIMESTEP = 0.001  # needs to evenly divide 0.04, should match input to t_toss when called
FRAMERATE = 30  # FPS in original video, should be 25 or 30
prev_hand_kp = {'left': [0, 0], 'right': [0, 0]}
prev_thumb_kp = [0, 0]
NUM_SINES = 5

video = "C0_14"
framecount = 1414

# %%


def readfile(n):
    if n < 10:
        numstring = "000" + str(n)
    elif n < 100:
        numstring = "00" + str(n)
    elif n < 1000:
        numstring = "0" + str(n)
    else:
        numstring = str(n)

    # filename = "~/Desktop/Expressive-MP/waypoints/" + video + "/" + video + "_00000000" + numstring + "_keypoints.json"
    filename = "~/autolab/DUET/waypoints/" + video + "/" + "20220516_RobotDance_cam1_" + \
        video[:2] + "0" + video[3:] + "_Trim_00000000" + \
        numstring + "_keypoints.json"
    item = pd.read_json(filename)
    return item


def get_angle(first, second):
    xdiff1 = first[0] - second[0]
    ydiff1 = first[1] - second[1]
    return np.arctan2(ydiff1, xdiff1)


def extract_angles(shoulder, elbow, wrist, fingertip, thumbtip):
    """
    Given the positions of the shoulder, elbow, wrist, 
    and fingertip, extract the three desired angles.
    """

    theta1 = get_angle(elbow, shoulder)
    theta2 = get_angle(wrist, elbow)
    theta3 = get_angle(fingertip, wrist)
    theta4 = get_angle(thumbtip, wrist)

    return [theta1, theta2, theta3, theta4]


def get_hand_kpt(d, side='right'):
    """
    Hand keypoint detection is extremely noisy, so we do 
    the best we can and allow for lots of smoothing later.
    """
    global prev_hand_kp

    keypoints = d[f'hand_{side}_keypoints_2d']
    for i in [12, 16, 8, 20, 11, 15, 7, 19]:
        p = keypoints[3 * i: 3 * i + 2]
        if p[0] != 0 and p[1] != 0:
            prev_hand_kp[side] = p
            return p
    return prev_hand_kp[side]


def get_thumb_kpt(d, side='right'):
    """
    Get the thumb detection in order to find the rotation angle of the wrist.
    """
    global prev_thumb_kp
    thumb = None

    keypoints = d[f'hand_{side}_keypoints_2d']
    for i in [4, 3, 2, 1]:
        p = keypoints[3 * i: 3 * i + 2]
        if p[0] != 0 and p[1] != 0:
            thumb = p
            break
    if thumb == None:
        return prev_thumb_kp
    else:
        return thumb


def linear_interp(array, num):
    interpolated = array[0]
    for i in range(len(array) - 1):
        interp = np.linspace(array[i], array[i + 1], num, endpoint=False)
        interpolated = np.vstack([interpolated, interp])

    return interpolated[1:]


def fourier_filter(array, thresh):
    fourier = np.fft.rfft(array)
    fourier[np.abs(fourier) < thresh] = 0
    filtered = np.fft.irfft(fourier)
    return filtered


def num_deriv(array, t):
    stack = None
    for a in array.T:
        grad = np.gradient(a, t, axis=0)
        stack = (grad if stack is None else np.vstack([stack, grad]))
    return stack.T


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# %%
"""
run this cell to read in the data. note that the "for n in range(x)" 
line must be changed to fit the length of the trajectory which you are
attempting to extract. also note that the smoothing in the next cell
crops the ends of the trajectory to make it smoother.
"""

theta_list = []

point_combos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8],
                [8, 9], [8, 12], [9, 10], [12, 13], [
                    10, 11], [13, 14], [11, 22],
                [14, 19], [11, 24], [14, 21], [
                    19, 20], [22, 23], [0, 15], [0, 16],
                [16, 18], [15, 17], [4, 'right_hand'], [4, 'right_thumb'], [7, 'left_hand'], [7, 'left_thumb']]

for n in range(1000):
    if FRAMERATE == 30 and n % 6 == 0:
        continue
    if len(readfile(n)["people"]) == 0:
        theta_list.append(thetas)
        continue

    d = readfile(n)["people"][0]
    if n == 0:
        print("OpenPose output:\n")
        print(d)
    series = d['pose_keypoints_2d']
    # pt0 = series[0:2]
    # pt1 = series[3:5]
    # pt2 = series[6:8]
    # pt3 = series[9:11]
    # pt4 = series[12:14]
    frame_angles = []

    def get_point(value):
        if type(value) == int:
            return series[value*3], series[value*3+1]
        else:
            name = value.split("_")
            return get_hand_kpt(d, name[0]) if name[1] == 'hand' else get_thumb_kpt(d, name[0])

    for combo in point_combos:
        frame_angles.append(
            get_angle(get_point(combo[0]), get_point(combo[1])))

    # hand_pt = get_hand_kpt(d)
    # thumb_pt = get_thumb_kpt(d)

    # thetas = extract_angles(pt2, pt3, pt4, hand_pt, thumb_pt)
    thetas = frame_angles
    theta_list.append(frame_angles)

tl = np.array(theta_list)
if FRAMERATE == 30:
    FRAMERATE = 25
    print("framerate adjusted")

# %%
tl[tl < 0] = tl[tl < 0] + 2 * np.pi
tl[tl < 0] = tl[tl < 0] + 2 * np.pi
print(tl[tl < 0])
kernel = np.array([1, 2, 4, 6, 10, 14, 17, 19, 17, 14, 10, 6, 4, 2, 1])
#kernel = np.ones(9)
kernel = kernel / np.sum(kernel)

if len(tl) % 2 != 0:
    tl = tl[:-1]

for i in range(len(tl) - 1):
    for j in range(len(tl[i])):
        if np.abs(tl[i, j] - tl[i + 1, j]) > np.abs(tl[i, j] - tl[i + 1, j] - 2 * np.pi):
            tl[i + 1, j] += 2 * np.pi
        if np.abs(tl[i, j] - tl[i + 1, j]) > np.abs(tl[i, j] - tl[i + 1, j] + 2 * np.pi):
            tl[i + 1, j] -= 2 * np.pi

smoothed_thetas = []
for i in range(len(tl[0])):
    smoothed_thetas.append(np.convolve(tl[:, i], kernel, mode='same'))

smoothed_thetas = np.vstack(smoothed_thetas)
print(smoothed_thetas.shape)
data = smoothed_thetas[:]

t = np.arange(len(data[0])) / FRAMERATE

# %%
for i in range(len(tl[0])):
    plt.plot(np.arange(len(tl)), tl[:, i],
             color='blue')
plt.title("Raw Joint Angle Data:")

df = pd.read_csv(
    '~/autolab/DUET/datafiles/IMG_4010.dat', sep='\t', dtype=np.float)
tmp = df.to_numpy(dtype=np.float)

print("ff", len(tmp))
traj = []
for i in np.arange(0, len(tmp), len(tmp)/len(data[0])):
    traj.append(tmp[(int)(i)])

traj = np.array(traj)
for i in range(1, 7):
    plt.plot(traj[:, i], color='red')
plt.legend()
plt.show()
# %%

point_combos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8],
                [8, 9], [8, 12], [9, 10], [12, 13], [
                    10, 11], [13, 14], [11, 22],
                [14, 19], [11, 24], [14, 21], [
                    19, 20], [22, 23], [0, 15], [0, 16],
                [16, 18], [15, 17], [4, 'right_hand'], [4, 'right_thumb'], [7, 'left_hand'], [7, 'left_thumb']]

neck = data[0]
chest_l = data[1]
shoulder_l = data[2] - chest_l
elbow_l = data[3] - shoulder_l
wrist_l = data[-4] - elbow_l
wrist_l2 = data[-3] - elbow_l

chest_r = data[4]
elbow_r = data[5] - chest_r
wrist_r = data[-2] - elbow_r
wrist_r2 = data[-1] - elbow_r

torso = data[7] - neck


# %%
plt.plot(np.arange(len(data[0])), neck, label="neck")
plt.plot(np.arange(len(data[0])), chest_l, label="chest_l")
plt.plot(np.arange(len(data[0])), elbow_l, label="elbow_l")
plt.plot(np.arange(len(data[0])), wrist_l, label="wrist_l")
plt.plot(np.arange(len(data[0])), wrist_l2, label="wrist_l2")
plt.plot(np.arange(len(data[0])), chest_r, label="chest_r")
plt.plot(np.arange(len(data[0])), elbow_r, label="elbow_r")
plt.plot(np.arange(len(data[0])), wrist_r, label="wrist_r")
plt.plot(np.arange(len(data[0])), wrist_r2, label="wrist_r2")

for i in range(1, 7):
    plt.plot(traj[:, i], color='black')
# plt.plot(np.arange(len(data[0])), data[3] - data[2], label="rotation")
plt.title("Gaussian Smoothed and Interpolated Trajectories")
plt.legend()
plt.show()


# %%
counter = 0
for arr in tqdm([neck, chest_l, elbow_l, wrist_l, wrist_l2, chest_r, elbow_r, wrist_r, wrist_r2]):
    # create data of complex numbers
    fft_arr = np.fft.fft(arr)
    fft_arr[np.abs(fft_arr) < 10] = 0
    fft_arr = fft_arr[:20]
    # extract real part
    x = [ele.real for ele in fft_arr if ele != 0]
    # extract imaginary part
    y = [ele.imag for ele in fft_arr if ele != 0]

    plt.scatter(x, y, label=f"{counter} th joint for human")
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    counter += 1

i = 0
for arr in tqdm(traj.T[1:7]):
    # create data of complex numbers
    fft_arr = np.fft.fft(arr)
    # extract real part
    x = [ele.real for ele in fft_arr]
    # extract imaginary part
    y = [ele.imag for ele in fft_arr]

    plt.scatter(x, y, color='black', label=f"ROBOT{i}")
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    i += 1
# plot the complex numbers
plt.legend()
plt.show()

base = np.cos(np.arange(len(t)) * 2 * NUM_SINES * np.pi / len(t))
# base -= np.cos(np.arange(len(t)) * 2 * 4 * np.pi / len(t))
# base += np.cos(np.arange(len(t)) * 2 * 2 * np.pi / len(t))
# base -= np.cos(np.arange(len(t)) * 2 * np.pi / len(t))
# base += np.sin(np.arange(len(t)) * 3 * np.pi / len(t))
# base -= np.sin(np.arange(len(t)) * 14 * np.pi / len(t))

# base *= 0.6

plt.plot(np.arange(len(data[0])), base)

# %%
# fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# axs[0, 0].plot(np.arange(len(data[0])), sh,
#                label="shoulder signal", c="orange")
# axs[0, 0].plot(np.arange(len(data[0])), sh_filtered,
#                label="fourier filtered", c="gold")
# axs[0, 0].set_title('Shoulder')
# axs[0, 1].plot(np.arange(len(data[0])), el, label="elbow signal", c="crimson")
# axs[0, 1].plot(np.arange(len(data[0])), el_filtered,
#                label="fourier filtered", c="darkred")
# axs[0, 1].set_title('Elbow')
# axs[1, 0].plot(np.arange(len(data[0])), wr, label="wrist signal", c="skyblue")
# axs[1, 0].plot(np.arange(len(data[0])), wr_filtered,
#                label="fourier filtered", c="gray")
# axs[1, 0].set_title('Wrist Flexion')
# axs[1, 1].plot(np.arange(len(data[0])), rt, label="rotation signal", c="lime")
# axs[1, 1].plot(np.arange(len(data[0])), rt_filtered,
#                label="fourier filtered", c="green")
# axs[1, 1].set_title('Wrist Rotation')
# # axs[1, 1].plot(np.arange(len(data[0])), sh_filtered, label="shoulder", c="orange")
# # axs[1, 1].plot(np.arange(len(data[0])), el_filtered, label="elbow", c="red")
# # axs[1, 1].plot(np.arange(len(data[0])), wr_filtered, label="wrist", c="skyblue")
# # axs[1, 1].set_title('Motion')
# plt.legend(loc='upper left')
# plt.show()

# %%
pass

# %%
# position = np.vstack([
#     np.pi - base * 0.8,  # shoulder rotation
#     sh_filtered * 1,  # shoulder abduction
#     el_filtered * 1,  # elbow angle
#     wr_filtered,  # wrist flexion
#     # wrist yaw (currently unused due to self-collisions)
#     np.ones_like(t) * np.pi / 2,
#     rt_filtered * 1 - np.pi / 2,  # wrist roll
# ]).T

# position = linear_interp(position, int(1 / FRAMERATE / TIMESTEP))
# t_int = linear_interp(t[np.newaxis].T, int(1 / FRAMERATE / TIMESTEP))

# velocity = num_deriv(position, TIMESTEP)
# acceleration = num_deriv(velocity, TIMESTEP)
# jerk = num_deriv(acceleration, TIMESTEP)

# output = np.hstack([t_int, position, velocity, acceleration, jerk])
# output = np.round_(output, decimals=5)
# np.savetxt("datafiles/" + video + ".dat", output, fmt="%10.5f", delimiter='\t')

# %%
