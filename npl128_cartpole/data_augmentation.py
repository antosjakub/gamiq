import numpy as np

# the data is is the format:
# cart_pos, cart_vel, pole_angle, pole_vel, target (1 or 0)
data = np.loadtxt("gym_cartpole_data.txt")

def mirror_flip(data: np.ndarray):
    n_rows, n_cols = data.shape
    data_new = np.zeros((2*n_rows, n_cols))
    data_new[:int(n_rows)] = data
    # mirror flip
    data_new[int(n_rows):, 0:4] = -data[:, 0:4]
    data_new[int(n_rows):, 4] = data[:, 4] == 0
    return data_new

def translate(data: np.ndarray, n_copies: int, min_x: float, max_x: float):
    # change pos
    data_new = np.zeros((n_copies * data.shape[0], data.shape[1]))
    for i, data in enumerate(data):
        data_copies = np.outer(np.ones(n_copies), data)
        random_pos =  np.random.uniform(min_x,max_x,n_copies-1)
        data_copies[1:n_copies,0] = random_pos
        data_new[(n_copies*i):(n_copies*(i+1)),:] = data_copies
    return data_new

def variate(data: np.ndarray, n_copies: int):
    # change all
    data_new =  np.random.uniform(-1,1,(n_copies * data.shape[0], data.shape[1]))
    data_new[:,0] = np.zeros(n_copies * data.shape[0])
    data_new[:,1] *= 0.01
    data_new[:,2] *= 0.001
    data_new[:,3] *= 0.001
    data_new[:,4] = np.zeros(n_copies * data.shape[0])
    for i, data_pnt in enumerate(data):
        data_copies = np.outer(np.ones(n_copies), data_pnt)
        data_new[(n_copies*i)] = np.zeros(data.shape[1])
        data_new[(n_copies*i):(n_copies*(i+1)),:] += data_copies
    return data_new


print("original dataset size:", data.shape)
data = mirror_flip(data)
data = translate(data, 10, -3,3)
data = variate(data, 10)
print("after data augmentation:", data.shape)

np.random.shuffle(data)

np.savetxt('gym_cartpole_data_augmented.txt', data, delimiter='', fmt='%10.5f')   