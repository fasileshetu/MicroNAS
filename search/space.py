from dataclasses import dataclass, field
from typing import List

VALID_ACTIVATIONS = ['relu', 'sigmoid', 'tanh']
VALID_LAYER_SIZES = [32, 64, 128, 256, 512]

@dataclass
class Architecture:
    hidden_layers: List[int] = field(default_factory=lambda: [128])
    activations: List[str] = field(default_factory=lambda: ['relu'])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0])
    learning_rate: float = 0.001

    def param_count(self) -> int:
        sizes = [784] + self.hidden_layers + [10]
        return sum(sizes[i] * sizes[i+1] for i in range(len(sizes)-1))

    def is_valid(self) -> bool:
        if len(self.hidden_layers) == 0:
            return False
        if len(self.hidden_layers) > 5:
            return False
        if len(self.hidden_layers) != len(self.activations):
            return False
        if len(self.hidden_layers) != len(self.dropout_rates):
            return False
        if any(s not in VALID_LAYER_SIZES for s in self.hidden_layers):
            return False
        if any(a not in VALID_ACTIVATIONS for a in self.activations):
            return False
        if any(d < 0 or d > 0.5 for d in self.dropout_rates):
            return False
        return True

    def __hash__(self):
        return hash((
            tuple(self.hidden_layers),
            tuple(self.activations),
            tuple(self.dropout_rates),
            self.learning_rate
        ))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return f"Architecture(layers={self.hidden_layers}, acts={self.activations}, dropout={self.dropout_rates}, lr={self.learning_rate})"