import pandas as pd
import torch
from collections import deque, defaultdict
from torchvision.transforms import Compose

from nids_framework.data import properties, utilities, transformation_builder
from nids_framework.model import transformer

class LiveClassifier:
    def __init__(self, model_path: str, config_path: str, dataset_name: str,
                 window_size: int, categorical_levels: int, input_shape: int,
                 embed_dim: int, num_heads: int, num_layers: int,
                 dropout_rate: float, feedforward_dim: int):
        self.model_path = model_path
        self.config_path = config_path
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.categorical_levels = categorical_levels
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.feedforward_dim = feedforward_dim
        
        self.model = self.load_model()
        self.buffer = SlidingWindowBuffer(window_size)

        self.properties = properties.NamedDatasetProperties(self.config_path).get_properties(self.dataset_name)
        self.lazy_transformation = self.setup_transformations()

    def load_model(self):
        model = transformer.TransformerClassifier(
            num_classes=1,
            input_dim=self.input_shape,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_dim=self.feedforward_dim,
            dropout=self.dropout_rate,
            window_size=self.window_size,
        )
        model.load_model_weights(self.model_path)
        model.eval()
        return model

    def setup_transformations(self):
        trans_builder = transformation_builder.TransformationBuilder()

        @trans_builder.add_step(order=1)
        def categorical_one_hot(sample, categorical_levels=self.categorical_levels):
            return utilities.one_hot_encoding(sample, categorical_levels)

        transformations = trans_builder.build()
        return Compose(transformations)
    
    def process(self, row, target = "default"):
        window = self.buffer.update(row, target)
        prediction = None

        if window is not None:
            numeric_features = torch.tensor(window[self.properties.numeric_features].values, dtype=torch.float32)
            categorical_features = torch.tensor(window[self.properties.categorical_features].values, dtype=torch.long)
            categorical_features = self.lazy_transformation(categorical_features).float()
            input_data = torch.cat((numeric_features, categorical_features), dim=-1).unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(input_data).squeeze(0)

        return prediction

class SlidingWindowBuffer:
    def __init__(self, size: int) -> None:
        self.groups = defaultdict(lambda: deque(maxlen=size))
        self.size = size

    def update(self, row: dict, target: any) -> pd.DataFrame:
        target_deque = self.groups[target]
        target_deque.append(row)

        if len(target_deque) == self.size:
            return pd.DataFrame(target_deque)

        return None
