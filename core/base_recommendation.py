from abc import ABC, abstractmethod

class BaseRecommendationAlgorithm(ABC):
    @abstractmethod
    def recommend_rules(self):
        """Abstract method to generate recommendations."""
        pass
