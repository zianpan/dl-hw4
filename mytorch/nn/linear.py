import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass

        self.ori_shape = A.shape 
        A_reshaped = A.reshape(-1, self.ori_shape[-1]) 
        Z_reshaped = A_reshaped @ self.W.T + self.b  

        self.A = A_reshaped
        return Z_reshaped.reshape(*self.ori_shape[:-1], -1)

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        shape = self.ori_shape
        N = np.prod(shape[:-1])

        dLdZ_reshaped = dLdZ.reshape(N, -1) 
        A_reshaped = self.A
        dLdA_reshaped = dLdZ_reshaped @ self.W 
        dLdW = dLdZ_reshaped.T @ A_reshaped  
        dLdb = np.sum(dLdZ_reshaped, axis=0) 
        dLdA = dLdA_reshaped.reshape(*shape)

        self.dLdA = dLdA
        self.dLdW = dLdW
        self.dLdb = dLdb
        
        # Return gradient of loss wrt input
        return dLdA
