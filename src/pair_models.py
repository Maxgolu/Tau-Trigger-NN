"""Neural-network models used by the pair-classification experiments."""

import torch
from torch import nn


def _require_declared_value(config, key, expected):
    """Reject a declaration that disagrees with the implemented architecture."""
    if key in config and config[key] != expected:
        raise ValueError(f"{key} must be {expected!r} for {config.get('name')}")


def _initialize_relu_mlp(hidden_layers, output_layer):
    for layer in hidden_layers:
        nn.init.kaiming_uniform_(
            layer.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        nn.init.zeros_(layer.bias)
    nn.init.xavier_uniform_(output_layer.weight)
    nn.init.zeros_(output_layer.bias)


class PairPtMLP(nn.Module):
    """Small 2 -> 8 -> 1 control using only the two member pT values."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        _initialize_relu_mlp((self.network[0],), self.network[2])

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("PairPtMLP expects inputs with shape (batch, 2)")
        return self.network(inputs)

    def predict_proba(self, inputs):
        return torch.sigmoid(self(inputs))


class FlatRawPairMLP(nn.Module):
    """Flat 90 -> 32 -> 16 -> 1 classifier for two coarse tensors."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(90, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        _initialize_relu_mlp(
            (self.network[0], self.network[2]),
            self.network[4],
        )

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != 90:
            raise ValueError("FlatRawPairMLP expects inputs with shape (batch, 90)")
        return self.network(inputs)

    def predict_proba(self, inputs):
        return torch.sigmoid(self(inputs))


class SharedMemberPairMLP(nn.Module):
    """Shared per-member encoder with configurable fusion and output target."""

    def __init__(
        self,
        member_width,
        *,
        fusion="ordered",
        output_classes=1,
        initialization="explicit_relu",
        member_encoder=None,
        activation="relu",
    ):
        super().__init__()
        if int(member_width) <= 0:
            raise ValueError("member_width must be positive")
        if fusion not in {
            "ordered",
            "symmetric",
            "symmetric_sum_absolute_difference",
        }:
            raise ValueError("unsupported member fusion")
        if int(output_classes) not in {1, 3}:
            raise ValueError("output_classes must be 1 or 3")
        if initialization not in {"explicit_relu", "pytorch_default"}:
            raise ValueError("unsupported shared-member initialization")
        if activation not in {"relu", "leaky_relu_0.01"}:
            raise ValueError("unsupported shared-member activation")
        self.member_width = int(member_width)
        self.fusion = fusion
        self.output_classes = int(output_classes)
        self.activation_name = activation
        if member_encoder is None:
            self.member_encoder = nn.Linear(self.member_width, 16)
            encoder_linear_layers = (self.member_encoder,)
        else:
            if activation != "leaky_relu_0.01":
                raise ValueError(
                    "the 32 -> 16 member encoder requires leaky_relu_0.01"
                )
            expected = [self.member_width, 32, 16]
            if list(member_encoder) != expected:
                raise ValueError(f"member_encoder must be {expected}")
            self.member_encoder = nn.Sequential(
                nn.Linear(self.member_width, 32),
                nn.LeakyReLU(0.01),
                nn.Linear(32, 16),
            )
            encoder_linear_layers = (
                self.member_encoder[0],
                self.member_encoder[2],
            )
        self.pair_hidden_1 = nn.Linear(32, 32)
        self.pair_hidden_2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, self.output_classes)
        if initialization == "explicit_relu":
            _initialize_relu_mlp(
                (*encoder_linear_layers, self.pair_hidden_1, self.pair_hidden_2),
                self.output,
            )

    def _activate(self, values):
        if self.activation_name == "relu":
            return torch.relu(values)
        return torch.nn.functional.leaky_relu(values, negative_slope=0.01)

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1:] != (2, self.member_width):
            raise ValueError(
                "SharedMemberPairMLP expects inputs with shape "
                f"(batch, 2, {self.member_width})"
            )
        left = self.member_encoder(inputs[:, 0])
        right = self.member_encoder(inputs[:, 1])
        if not isinstance(self.member_encoder, nn.Sequential):
            left = self._activate(left)
            right = self._activate(right)
        if self.fusion == "ordered":
            pair = torch.cat((left, right), dim=1)
        else:
            pair = torch.cat((left + right, torch.abs(left - right)), dim=1)
        hidden = self._activate(self.pair_hidden_1(pair))
        hidden = self._activate(self.pair_hidden_2(hidden))
        return self.output(hidden)

    def predict_proba(self, inputs):
        logits = self(inputs)
        if self.output_classes == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)[:, 2:3]


class SharedHighResolutionPairModel(nn.Module):
    """Shared member encoder for high-resolution EM2 pair inputs.

    Each member can contribute zero, one, or two 12x12 EM2 image channels and
    a declared scalar vector. The same image and scalar branches process both
    members. Their two 16-value embeddings are concatenated with the declared
    event-context vector and passed to a small pair head. This covers the
    complete input, controlled source ablations, and the compatible two-layer
    encoder comparison without changing the pair interface.
    """

    def __init__(
        self,
        member_scalar_width=47,
        context_width=1,
        *,
        image_channels=1,
        image_depth=1,
    ):
        super().__init__()
        self.member_scalar_width = int(member_scalar_width)
        self.context_width = int(context_width)
        self.image_channels = int(image_channels)
        self.image_depth = int(image_depth)
        if self.member_scalar_width <= 0:
            raise ValueError("member_scalar_width must be positive")
        if self.context_width < 0:
            raise ValueError("context_width must be nonnegative")
        if self.image_channels not in {0, 1, 2}:
            raise ValueError("image_channels must be 0, 1, or 2")
        if self.image_depth not in {1, 2}:
            raise ValueError("image_depth must be 1 or 2")
        if self.image_channels == 0 and self.image_depth != 1:
            raise ValueError("an absent image branch cannot have depth two")
        image_layers = []
        image_width = 0
        if self.image_channels:
            image_layers.extend(
                (
                    nn.Conv2d(self.image_channels, 4, kernel_size=3),
                    nn.BatchNorm2d(4),
                    nn.LeakyReLU(0.1),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
            image_width = 100
            if self.image_depth == 2:
                image_layers.extend(
                    (
                        nn.Conv2d(4, 8, kernel_size=3),
                        nn.BatchNorm2d(8),
                        nn.LeakyReLU(0.1),
                    )
                )
                image_width = 72
            image_layers.append(nn.Flatten())
        self.image_branch = nn.Sequential(*image_layers)
        self.scalar_branch = nn.Sequential(
            nn.Linear(self.member_scalar_width, 16),
            nn.LeakyReLU(0.1),
        )
        self.member_projection = nn.Sequential(
            nn.Linear(image_width + 16, 16),
            nn.LeakyReLU(0.1),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(32 + self.context_width, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=0.1,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_member(self, image, scalars):
        if self.image_channels == 0:
            if image is not None:
                raise ValueError("image must be omitted when image_channels is zero")
        elif self.image_channels == 1:
            if image.ndim != 3 or image.shape[1:] != (12, 12):
                raise ValueError("member image must have shape (batch, 12, 12)")
            image = image.unsqueeze(1)
        elif image.ndim != 4 or image.shape[1:] != (2, 12, 12):
            raise ValueError("member image must have shape (batch, 2, 12, 12)")
        if scalars.ndim != 2 or scalars.shape[1] != self.member_scalar_width:
            raise ValueError(
                f"member scalars must have shape (batch, {self.member_scalar_width})"
            )
        if image is not None and image.shape[0] != scalars.shape[0]:
            raise ValueError("member image and scalar batches must match")
        scalar_embedding = self.scalar_branch(scalars)
        member_parts = [scalar_embedding]
        if image is not None:
            member_parts.insert(0, self.image_branch(image))
        return self.member_projection(torch.cat(member_parts, dim=1))

    def forward(self, images, scalars, context):
        if self.image_channels == 0:
            if images.ndim != 2 or images.shape[1] != 0:
                raise ValueError("images must have shape (batch, 0)")
        elif self.image_channels == 1:
            if images.ndim != 4 or images.shape[1:] != (2, 12, 12):
                raise ValueError("images must have shape (batch, 2, 12, 12)")
        elif images.ndim != 5 or images.shape[1:] != (2, 2, 12, 12):
            raise ValueError("images must have shape (batch, 2, 2, 12, 12)")
        if scalars.ndim != 3 or scalars.shape[1:] != (2, self.member_scalar_width):
            raise ValueError(
                "scalars must have shape "
                f"(batch, 2, {self.member_scalar_width})"
            )
        if context.ndim != 2 or context.shape[1] != self.context_width:
            raise ValueError(
                f"context must have shape (batch, {self.context_width})"
            )
        if not (
            images.shape[0] == scalars.shape[0] == context.shape[0]
        ):
            raise ValueError("image, scalar, and event-context batches must match")

        left_image = None if self.image_channels == 0 else images[:, 0]
        right_image = None if self.image_channels == 0 else images[:, 1]
        left = self.encode_member(left_image, scalars[:, 0])
        right = self.encode_member(right_image, scalars[:, 1])
        pair_input = torch.cat((left, right, context), dim=1)
        return self.pair_head(pair_input)

    def predict_proba(self, images, scalars, context):
        return torch.sigmoid(self(images, scalars, context))


def build_pair_model(config):
    """Build a pair model from a configuration dictionary.

    The model settings may be supplied directly or under a top-level
    ``model`` key, matching the configuration style used by ``train.py``.
    """
    model_config = config.get("model", config.get("neural_model", config))
    name = model_config.get("name", "pair_pt_mlp")
    if name == "pair_pt_mlp":
        _require_declared_value(model_config, "layers", [2, 8, 1])
        _require_declared_value(model_config, "activation", "relu")
        return PairPtMLP()
    if name == "flat_pair_mlp":
        _require_declared_value(model_config, "layers", [90, 32, 16, 1])
        _require_declared_value(model_config, "activation", "relu")
        return FlatRawPairMLP()
    if name == "shared_member_pair_mlp":
        output_classes = int(model_config.get("output_classes", 1))
        _require_declared_value(model_config, "member_embedding_width", 16)
        _require_declared_value(
            model_config,
            "pair_head",
            [32, 32, 16, output_classes],
        )
        model = SharedMemberPairMLP(
            model_config["member_width"],
            fusion=model_config.get("fusion", "ordered"),
            output_classes=output_classes,
            initialization=model_config.get("initialization", "explicit_relu"),
            member_encoder=model_config.get("member_encoder"),
            activation=model_config.get("activation", "relu"),
        )
        _require_declared_value(
            model_config,
            "trainable_parameters",
            count_parameters(model),
        )
        return model
    if name == "shared_member_high_resolution_em2":
        _require_declared_value(model_config, "member_embedding_width", 16)
        member_scalar_width = int(model_config.get("member_scalar_width", 47))
        context_width = int(model_config.get("context_width", 1))
        image_channels = int(model_config.get("image_channels", 1))
        image_depth = int(model_config.get("image_depth", 1))
        _require_declared_value(
            model_config,
            "pair_head",
            [32 + context_width, 32, 16, 1],
        )
        model = SharedHighResolutionPairModel(
            member_scalar_width,
            context_width,
            image_channels=image_channels,
            image_depth=image_depth,
        )
        _require_declared_value(
            model_config,
            "trainable_parameters",
            count_parameters(model),
        )
        return model
    raise ValueError(f"Unknown pair model: {name}")


def count_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
