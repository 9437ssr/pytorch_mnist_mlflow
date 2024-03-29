import argparse
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
from mlflow.pyfunc import PythonModel
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
import requests
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_classes = 10
num_layers = 2
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# Set the experiment name
mlflow.set_experiment("mnist-digit-recognition")

mlflow.log_param("num_hidden_layers", num_layers)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", num_epochs)
mlflow.log_param("learning_rate", learning_rate)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./mnist/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./mnist/',
                                          train=False,download=True,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self._get_device())
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self._get_device())
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        # Softmax
        out = self.softmax(out)
        return out
    
    def _get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

def train_model(model):
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metric("neg_log_loss", loss.item())

            if (i+1) % 50 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

                # Test the model
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images = images.reshape(-1, sequence_length, input_size).to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                mlflow.log_metric("Test accuracy vs number of training epochs", 100 * correct / total)

                print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Modify the predict method to customize output
class MnistTorchRNN(PythonModel):

    def load_context(self, context):
        self.model = mlflow.pytorch.load_model(
            context.artifacts["torch-rnn-model"], map_location="cpu")
        self.model.to('cpu')
        self.model.eval()

    def predict(self, context, input_array):
        import numpy as np
        with torch.no_grad():
            input_tensor = torch.from_numpy(
                input_array.reshape(-1, 28, 28).astype(np.float32)).to('cpu')
            model_results = self.model(input_tensor).numpy()
            model_results = np.power(np.e, model_results)
            index_max = np.argmax(model_results)
            return index_max
        

if __name__=="__main__":
    # Start a run and train a model
    mlflow.end_run()
    with mlflow.start_run():
        train_model(model)
        torch.save(model.state_dict(), "model.ckpt")

        # Log the pytorch model
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path="torch-rnn-model-default-params")

        mlflow.pyfunc.log_model(
        artifact_path="pyfunc-rnn",
        artifacts={
            "torch-rnn-model": mlflow.get_artifact_uri("torch-rnn-model-default-params")
        },
        python_model=MnistTorchRNN())

    mlflow.end_run()
