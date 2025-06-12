from abc import abstractmethod, abstractproperty
from typing import Any, Callable, Dict, List, Optional, Tuple

class ParaMonad:
    @abstractmethod
    def load(self, checkpoint: Dict[str, Any]):
        raise NotImplementedError

    @abstractproperty
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def test_step(self, *args, **kwargs) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, *args, **kwargs) -> Dict[str, float]:
        raise NotImplementedError
