import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.conv1 = nn.Conv(3, 16, 7, stride=4, device=device, dtype=dtype)
        self.bn1 = nn.BatchNorm2d(16, device=device, dtype=dtype)
        
        self.conv2 = nn.Conv(16, 32, 3, stride=2, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm2d(32, device=device, dtype=dtype)
        
        self.conv3 = nn.Conv(32, 32, 3, stride=1, device=device, dtype=dtype)
        self.bn3 = nn.BatchNorm2d(32, device=device, dtype=dtype)
        self.conv4 = nn.Conv(32, 32, 3, stride=1, device=device, dtype=dtype)
        self.bn4 = nn.BatchNorm2d(32, device=device, dtype=dtype)
        
        self.conv5 = nn.Conv(32, 64, 3, stride=2, device=device, dtype=dtype)
        self.bn5 = nn.BatchNorm2d(64, device=device, dtype=dtype)
        
        self.conv6 = nn.Conv(64, 128, 3, stride=2, device=device, dtype=dtype)
        self.bn6 = nn.BatchNorm2d(128, device=device, dtype=dtype)
        
        self.conv7 = nn.Conv(128, 128, 3, stride=1, device=device, dtype=dtype)
        self.bn7 = nn.BatchNorm2d(128, device=device, dtype=dtype)
        self.conv8 = nn.Conv(128, 128, 3, stride=1, device=device, dtype=dtype)
        self.bn8 = nn.BatchNorm2d(128, device=device, dtype=dtype)
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        
        self.relu = nn.ReLU()
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        identity = x
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(x + identity)
        
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        
        identity = x
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(x + identity)
        
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model = seq_model
        self.device = device
        self.dtype = dtype
        
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.seq_len = seq_len

        if seq_model == 'rnn':
            self.seq_model_layer = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
            self.output_dim = hidden_size
        elif seq_model == 'lstm':
            self.seq_model_layer = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
            self.output_dim = hidden_size
        elif seq_model == 'transformer':
            num_head = 1
            dim_head = embedding_size
            self.seq_model_layer = nn.Transformer(
                embedding_size, hidden_size, num_layers,
                num_head=num_head,
                dim_head=dim_head,
                dropout=0.0,
                causal=True,
                device=device,
                dtype=dtype,
                batch_first=False,
                sequence_len=seq_len,
            )
            self.output_dim = embedding_size
        else:
            raise ValueError(f"Unknown seq_model: {seq_model}")
        
        self.linear = nn.Linear(self.output_dim, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        
        emb = self.embedding(x)
        output, h_new = self.seq_model_layer(emb, h)
        output = output.reshape((seq_len * bs, self.output_dim))
        out = self.linear(output)
        
        return out, h_new
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
