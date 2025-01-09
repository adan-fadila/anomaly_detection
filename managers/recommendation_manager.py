from core.base_manager import BaseManager

class RecommendationManager(BaseManager):
    def manage(self):
        recommendations = []
        for algo in self.algorithms:
            recommendations.extend(algo.recommend_rules())
        return recommendations
