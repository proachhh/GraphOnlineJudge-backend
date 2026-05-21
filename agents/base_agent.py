from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    def __init__(self, name: str = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
