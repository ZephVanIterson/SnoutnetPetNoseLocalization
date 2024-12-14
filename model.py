import torch
import torch.nn.functional as F
import torch.nn as nn

class SnoutNet(nn.Module):

	# Input 227x227x3
	# Layer 1: 3x3x3 conv with ReLU and Max Pooling, output 57x57x64
	# Layer 2: 3x3x64 conv with ReLU and Max Pooling, output 15x15x128
	# Layer 3: 3x3x128 conv with ReLU and Max Pooling, output 4x4x256
	# Layer 4: FC layer with 1024 neurons and ReLU
	# Layer 5: FC layer with 1024 neurons and ReLU
	# Layer 6: FC layer with 2 neurons
	def __init__(self):
		super(SnoutNet, self).__init__()

		### Layers
		## Note: Solution to output dimension problem could be applying padding=2 to all conv layers and padding=1
		## to all MaxPool layers (Not all of these are necessary, could also investigate using stride)

		# Layer 1
		self.conv1 = nn.Conv2d(3, 64, 3, padding=2)
		self.pool1 = nn.MaxPool2d(4, 4, padding=1)

		# Layer 2
		self.conv2 = nn.Conv2d(64, 128, 3, padding=2)
		self.pool2 = nn.MaxPool2d(4, 4, padding=1)

		# Layer 3
		self.conv3 = nn.Conv2d(128, 256, 3, padding=2)
		self.pool3 = nn.MaxPool2d(4, 4, padding=1)

		# Layer 4
		self.fc1 = nn.Linear(256*4*4, 1024)

		# Layer 5
		self.fc2 = nn.Linear(1024, 1024)

		# Layer 6
		self.fc3 = nn.Linear(1024, 2)

		### Other Variables
		self.input_shape = (3, 227, 227)


	def forward(self, X):
		X = self.conv1(X)
		X = F.relu(X)
		X = self.pool1(X)

		X = self.conv2(X)
		X = F.relu(X)
		X = self.pool2(X)

		X = self.conv3(X)
		X = F.relu(X)
		X = self.pool3(X)

		X = X.view(-1, 256*4*4)

		X = self.fc1(X)
		X = F.relu(X)

		X = self.fc2(X)
		X = F.relu(X)
		X = self.fc3(X)

		return X

	def verboseForward(self, X):
		X = self.conv1(X)
		X = F.relu(X)
		print("Conv1: ", X.shape)

		X = self.pool1(X)
		print("Pool1: ", X.shape)

		X = self.conv2(X)
		X = F.relu(X)
		print("Conv2: ", X.shape)

		X = self.pool2(X)
		print("Pool2: ", X.shape)

		X = self.conv3(X)
		X = F.relu(X)
		print("Conv3: ", X.shape)

		X = self.pool3(X)
		print("Pool3: ", X.shape)

		X = X.view(-1, 256 * 4 * 4)
		print("Tranformed: ", X.shape)

		X = self.fc1(X)
		X = F.relu(X)
		print("FC1: ", X.shape)

		X = self.fc2(X)
		X = F.relu(X)
		print("FC2: ", X.shape)

		X = self.fc3(X)
		print("Out: ", X.shape)

		return X
