from abc import ABC, abstractmethod
from typing import *

type Messages = List[Dict[str, str]]

class Agent(ABC):
    @abstractmethod
    def generate(self, state: Messages) -> str:
        """Here, the agent might directly generate a response, or it might """
        pass
