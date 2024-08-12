# Author: Maialen Espí Landa
# Date: 11.08.2024
# Description: Quantum neural network implementation using PyTorch, 
#              featuring a custom quantum gate over a harmonic oscillator and training 
#              with the L-BFGS optimizer.


import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Gate(torch.nn.Module):
    """
    A quantum gate model that applies a kick and rotation operation to a quantum state.
    
    Attributes:
        zeta (torch.nn.Parameter): The parameter for the kick operation.
        time (torch.nn.Parameter): The parameter for the rotation operation.
        a (torch.Tensor): The annihilation operator.
        a_dagger (torch.Tensor): The creation operator.
        sigma_x (torch.Tensor): The Pauli-X matrix.
    """

    def __init__(self, a, a_dagger):
        super().__init__()
        self.zeta = torch.nn.Parameter(torch.tensor(1.0), requires_grad = True)
        self.time = torch.nn.Parameter(torch.tensor(1.0), requires_grad = True)
        self.a = a
        self.a_dagger = a_dagger
        self.sigma_x = torch.tensor([[0, 1], [1, 0]], dtype = torch.complex64)

    def kick(self, state, eta):
        """
        Applies a kick operation to the quantum state.
        
        Args:
            state (torch.Tensor): The quantum state.
            eta (torch.Tensor): The parameter for the kick operation.
        
        Returns:
            torch.Tensor: The quantum state after the kick operation.
        """
        kick_matrix = torch.matrix_exp(
            -1j * eta * torch.kron(self.a + self.a_dagger, self.sigma_x)
        )
        kick_matrix = kick_matrix.repeat(state.shape[0], 1, 1).to(torch.complex64)
        return torch.bmm(kick_matrix, state)

    def rotation(self, state, delta_time):
        """
        Applies a rotation operation to the quantum state.
        
        Args:
            state (torch.Tensor): The quantum state.
            delta_time (torch.Tensor): The time parameter for the rotation operation.
        
        Returns:
            torch.Tensor: The quantum state after the rotation operation.
        """
        rot_matrix = torch.matrix_exp(
            -1j * delta_time.view(-1, 1, 1) 
            * torch.kron(self.a @ self.a_dagger, torch.eye(2))
        ).to(torch.complex64)
        return torch.bmm(rot_matrix, state)

    def forward(self, X, state):
        """
        Forward pass through the gate.
        
        Args:
            X (torch.Tensor): Input features.
            state (torch.Tensor): The quantum state.
        
        Returns:
            torch.Tensor: The quantum state after applying the gate.
        """
        state = self.kick(state, self.zeta)
        state = self.rotation(state, X + self.time)
        return state


class QNN(torch.nn.Module):
    """
    A quantum neural network model with multiple layers of quantum gates.
    
    Attributes:
        cut_off (int): The dimension cut-off for the Hilbert space.
        a (torch.Tensor): The annihilation operator.
        a_dagger (torch.Tensor): The creation operator.
        initial_state (torch.Tensor): The initial quantum state.
        measurement (torch.Tensor): The measurement operator.
        lyrs (torch.nn.ModuleList): List of quantum gate layers.
        loss_fn (torch.nn.Module): Loss function for training.
    """

    def __init__(self, n_layers, cut_off):
        super().__init__()
        self.cut_off = cut_off
        self.a, self.a_dagger = self.create_operators()
        self.initial_state = self.create_initial_state()
        self.measurement = self.create_measurement()
        self.lyrs = torch.nn.ModuleList([Gate(self.a, self.a_dagger) for _ in range(n_layers)])
        self.loss_fn = torch.nn.MSELoss()

    def create_operators(self):
        """
        Creates the annihilation and creation operators.
        
        Returns:
            tuple: The annihilation (a) and creation (a_dagger) operators.
        """
        a = torch.zeros((self.cut_off, self.cut_off), dtype = torch.complex64)
        for n in range(1, self.cut_off):
            a[n - 1, n] = torch.sqrt(torch.tensor(n, dtype = torch.complex64))
        a_dagger = a.T.conj()
        return a, a_dagger
    
    def create_initial_state(self, n = 3):
        """
        Creates the initial quantum state.
        
        Args:
            n (int, optional): Coherent state parameter. Defaults to 3.
        
        Returns:
            torch.Tensor: The initial quantum state.
        """
        oscillator_zero = torch.zeros(self.cut_off, dtype = torch.complex64)
        oscillator_zero[0] = 1
        coherent_state = torch.matmul(
            torch.matrix_exp(n * self.a - n * self.a_dagger), oscillator_zero
        )
        qubit_zero = torch.tensor([1 + 0j, 0 + 0j])
        return torch.kron(coherent_state, qubit_zero)
    
    def create_measurement(self):
        """
        Creates the measurement operator.
        
        Returns:
            torch.Tensor: The measurement operator.
        """
        qubit_zero_prob = torch.zeros(2 * self.cut_off, dtype = torch.complex64)
        qubit_zero_prob[0::2] = 1
        return qubit_zero_prob

    def forward(self, X):
        """
        Forward pass through the quantum neural network.
        
        Args:
            X (torch.Tensor): Input features.
        
        Returns:
            torch.Tensor: The output probability distribution.
        """
        state = torch.tensor(
            np.array(X.shape[0] * [self.initial_state]),
            requires_grad = False,
            dtype = torch.complex64
        ).unsqueeze(2)

        for lyr in self.lyrs:
            state = lyr(X, state)
        
        state = torch.squeeze(state, dim = -1)
        return torch.abs(torch.matmul(state, self.measurement) ** 2)
    
    def train(self, X, y, lr, epochs):
        """
        Trains the quantum neural network using the L-BFGS optimizer.
        
        Args:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target values.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
        
        Returns:
            tuple: Lists of training and validation losses over the epochs.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 42)
        X_train = torch.tensor(X_train, dtype = torch.float32)
        y_train = torch.tensor(y_train, dtype = torch.float32)
        X_val = torch.tensor(X_val, dtype = torch.float32)
        y_val = torch.tensor(y_val, dtype = torch.float32)
        
        optimizer = torch.optim.LBFGS(self.parameters(), lr = lr)
        train_losses, val_losses = [], []

        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn(self(X_train), y_train)
            loss.backward()
            return loss

        for _ in tqdm(range(epochs)):
            optimizer.step(closure)
            train_loss = closure()
            with torch.no_grad():
                loss_val = self.loss_fn(self(X_val), y_val)
            train_losses.append(train_loss.item())
            val_losses.append(loss_val.item())

        return train_losses, val_losses
