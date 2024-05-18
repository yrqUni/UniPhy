####################################################################################################
# Load the Moving MNIST dataset
# n_frames_output minimum value is 1
####################################################################################################
from torch.utils.data import DataLoader
from MMNIST import MovingMNIST
root = './MMNIST/'
is_train = True
n_frames_input = 10
n_frames_output = 10
num_objects = [3]
dataset = MovingMNIST(root=root, is_train=is_train, n_frames_input=n_frames_input, n_frames_output=n_frames_output, num_objects=num_objects, num_samples=int(1e10))
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
dataiter = iter(dataloader)
input, output = dataiter.next()
print(f"Input Shape: {input.shape}")
print(f"Output Shape: {output.shape}")
