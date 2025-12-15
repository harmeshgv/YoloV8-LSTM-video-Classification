from abc import ABC, abstractmethod
from app.shared.device import get_device

class BaseModel(ABC):
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.device = get_device()
        self.loaded = False
        self.model = None

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

    def to_device(self):
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(self.device)

    def is_loaded(self) -> bool:
        return self.loaded
