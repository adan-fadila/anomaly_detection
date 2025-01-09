from abc import ABC, abstractmethod

class BaseManager(ABC):
    def __init__(self):
        self.algorithms = []

    def add_algorithm(self, algorithm):
        """Add an algorithm to the manager."""
        self.algorithms.append(algorithm)

    @abstractmethod
    def manage(self, *args, **kwargs):
        """Abstract method for managing algorithms."""
        pass
