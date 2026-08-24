import torch
import torch.nn as nn


def _activation(name):
    """Return a fresh activation module by name."""
    if name in (None, "relu"):
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unknown activation: {name}")


class TensorCNN(nn.Module):
    """Configurable CNN over folded calorimeter tensors plus scalar features.

    The flat input vector is exactly what ``DynamicMLP`` receives. Each branch
    names one feature, slices its contiguous columns, folds them back into a
    ``(channels, height, width)`` image in C-order (matching the reshape used
    by the feature registry), and applies a configured stack of conv/pool
    layers. Any feature not consumed by a branch is passed through as a scalar
    and concatenated with the flattened branch outputs before the dense head.

    The output interface is identical to ``DynamicMLP`` (``forward`` returns a
    probability, ``forward_logits`` returns the pre-sigmoid score), so every
    downstream consumer works unchanged.
    """

    def __init__(self, input_dim, feature_layout, model_config):
        super().__init__()
        # feature_layout: ordered list of (name, start, length) column ranges.
        layout = {name: (start, length) for name, start, length in feature_layout}
        self.input_dim = int(input_dim)

        branch_specs = model_config.get("branches", [])
        if not branch_specs:
            raise ValueError("tensor_cnn requires at least one branch")

        self.branches = nn.ModuleList()
        self._branch_meta = []
        consumed = set()
        flattened_dim = 0
        for spec in branch_specs:
            name = spec["feature"]
            if name not in layout:
                raise ValueError(
                    f"Branch feature '{name}' is not in features_to_use"
                )
            if name in consumed:
                raise ValueError(f"Feature '{name}' used by two branches")
            start, length = layout[name]
            channels, height, width = (int(v) for v in spec["shape"])
            if channels * height * width != length:
                raise ValueError(
                    f"Branch '{name}' shape {spec['shape']} has "
                    f"{channels * height * width} cells but the feature "
                    f"provides {length} columns"
                )
            module, out_dim = self._build_conv_stack(
                channels,
                height,
                width,
                spec.get("layers", []),
                activation=model_config.get("activation", "relu"),
                batchnorm=bool(model_config.get("batchnorm", False)),
            )
            self.branches.append(module)
            # The input transform (e.g. log1p) is applied in preprocessing,
            # before standardization, not here; see train.py. The field is
            # retained on the spec only as documentation of that intent.
            include_raw = bool(spec.get("include_raw", False))
            self._branch_meta.append(
                {
                    "name": name,
                    "start": start,
                    "length": length,
                    "shape": (channels, height, width),
                    "flatten_dim": out_dim,
                    "include_raw": include_raw,
                }
            )
            consumed.add(name)
            flattened_dim += out_dim
            # include_raw: skip connection. The branch's preprocessed columns
            # (transform + standardization applied in train.py) are ALSO fed
            # directly to the dense head, alongside the conv output. The head
            # input then contains the flat representation as a subset, so the
            # conv branch is tested as added information, not a replacement.
            if include_raw:
                flattened_dim += length

        # Every feature not folded into a branch enters the head as a scalar.
        scalar_ranges = [
            (start, length)
            for name, (start, length) in layout.items()
            if name not in consumed
        ]
        self._scalar_ranges = sorted(scalar_ranges)
        scalar_dim = sum(length for _, length in self._scalar_ranges)

        activation = model_config.get("activation", "relu")
        head_layers = []
        current = flattened_dim + scalar_dim
        for hidden in model_config.get("head", [16]):
            head_layers.append(nn.Linear(current, int(hidden)))
            head_layers.append(_activation(activation))
            current = int(hidden)
        head_layers.append(nn.Linear(current, 1))
        head_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*head_layers)

    @staticmethod
    def _build_conv_stack(channels, height, width, layer_specs,
                          activation="relu", batchnorm=False):
        modules = []
        c, h, w = channels, height, width
        for layer in layer_specs:
            kind = layer.get("type", "conv")
            if kind == "conv":
                kernel = int(layer["kernel"])
                out_channels = int(layer["out_channels"])
                padding = int(layer.get("pad", 0))
                modules.append(
                    nn.Conv2d(c, out_channels, kernel_size=kernel, padding=padding)
                )
                # conv -> [BatchNorm] -> activation. BatchNorm stabilizes the
                # activation scale across the imbalanced full dataset, which
                # otherwise drives a first-step collapse into dead units.
                if batchnorm:
                    modules.append(nn.BatchNorm2d(out_channels))
                modules.append(_activation(activation))
                c = out_channels
                h = h - kernel + 1 + 2 * padding
                w = w - kernel + 1 + 2 * padding
            elif kind == "pool":
                size = int(layer.get("size", 2))
                if layer.get("kind", "max") == "max":
                    modules.append(nn.MaxPool2d(kernel_size=size))
                else:
                    modules.append(nn.AvgPool2d(kernel_size=size))
                h //= size
                w //= size
            else:
                raise ValueError(f"Unknown branch layer type: {kind}")
            if h < 1 or w < 1:
                raise ValueError(
                    "Branch layer stack reduces the map below 1x1; "
                    "check kernel and pool sizes against the input shape"
                )
        return nn.Sequential(*modules), c * h * w

    def forward(self, x):
        return torch.sigmoid(self.forward_logits(x))

    def forward_logits(self, x):
        parts = []
        for module, meta in zip(self.branches, self._branch_meta):
            start, length = meta["start"], meta["length"]
            channels, height, width = meta["shape"]
            block = x[:, start:start + length].reshape(-1, channels, height, width)
            parts.append(module(block).reshape(x.shape[0], -1))
            if meta.get("include_raw", False):
                parts.append(x[:, start:start + length])
        for start, length in self._scalar_ranges:
            parts.append(x[:, start:start + length])
        combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        # The head ends with a Sigmoid module (kept for state-dict layout);
        # slice it off so this returns a true pre-sigmoid logit, exactly like
        # DynamicMLP.forward_logits. Applying the full head here would make
        # forward() a double sigmoid, bounding outputs to (0.5, 0.731) and
        # starving the gradients.
        return self.head[:-1](combined)


def build_model(config, input_dim, feature_layout):
    """Return the configured model, defaulting to the legacy MLP.

    A configuration without a ``model`` block (or with ``model.name == "mlp"``)
    reproduces ``DynamicMLP`` exactly, so every existing config is unchanged.
    """
    model_config = config.get("model")
    if model_config is None or model_config.get("name", "mlp") == "mlp":
        return DynamicMLP(
            input_dim=input_dim,
            hidden_layers=config.get("hidden_layers", [32, 16]),
        )
    if model_config["name"] == "tensor_cnn":
        return TensorCNN(input_dim, feature_layout, model_config)
    raise ValueError(f"Unknown model.name: {model_config['name']}")


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
        Returns probabilities of shape (batch_size, 1).
        """
        return torch.sigmoid(self.forward_logits(x))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-sigmoid scores without changing saved parameter names."""
        # The final module is the legacy Sigmoid. Keeping it in ``network``
        # preserves exact state-dict compatibility with every saved checkpoint.
        return self.network[:-1](x)


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
