import torch
import torch.nn as nn


class DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list):
        """
        A flexible Multi-Layer Perceptron that automatically builds layers
        based on the provided input dimensions and layer structure.

        Args:
            input_dim (int): Total number of features concatenated from the registry.
            hidden_layers (list): A list of integers specifying nodes per hidden layer.
                                  Example: [128, 64, 32]
        """
        super(DynamicMLP, self).__init__()

        layers = []
        current_dim = input_dim

        # Dynamically append linear and activation layers
        for next_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU())

            # Optional: If you ever want to add Dropout to prevent overfitting,
            # you can add a placeholder configuration rule here, e.g.:
            # layers.append(nn.Dropout(p=0.1))

            current_dim = next_dim

        # Final output layer mapping to a single logit for binary classification
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())

        # Combine everything into a sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Accepts input tensors of shape (batch_size, input_dim).
        Returns unnormalized logits of shape (batch_size, 1).
        """
        return self.network(x)


if __name__ == "__main__":
    # Quick sanity check code to verify dimensions work correctly
    print("Testing DynamicMLP initialization...")
    test_input = torch.randn(10, 49)  # Simulate a batch of 10 events with 49 features

    # Initialize a dummy architecture
    model = DynamicMLP(input_dim=49, hidden_layers=[128, 64, 32])
    test_output = model(test_input)

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print("Model structure verified successfully!")