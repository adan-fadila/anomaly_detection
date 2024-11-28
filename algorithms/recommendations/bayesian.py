import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from core.base_recommendation import BaseRecommendationAlgorithm

class BayesianRecommendation(BaseRecommendationAlgorithm):
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = pd.read_csv(data_file)
        self.model = None
        self.inference = None
        self.prepare_data()
        self.create_bayesian_network()
        self.train_model()

    def prepare_data(self):
        """Prepare the dataset for the Bayesian model."""
        self.data['hour'] = pd.to_datetime(self.data['timestamp']).dt.hour
        self.data['temperature'] = pd.cut(self.data['temperature'], bins=[-float('inf'), 18, 22, 26, float('inf')],
                                          labels=['low', 'normal', 'high', 'very_high'])
        self.data['humidity'] = pd.cut(self.data['humidity'], bins=[-float('inf'), 30, 60, 90, float('inf')],
                                       labels=['low', 'normal', 'high', 'very_high'])
        self.data.drop(columns=['timestamp'], inplace=True)

    def create_bayesian_network(self):
        """Define the structure of the Bayesian Network."""
        self.model = BayesianNetwork([
            ('temperature', 'ac_status'),
            ('humidity', 'ac_status'),
            ('ac_status', 'ac_energy'),
            ('hour', 'ac_status'),
            ('hour', 'fan'),
            ('hour', 'lights'),
            ('hour', 'laundry_machine')
        ])

    def train_model(self):
        """Train the Bayesian Network using the data."""
        self.model.fit(self.data, estimator=BayesianEstimator, prior_type="BDeu")
        self.inference = VariableElimination(self.model)

    def recommend_rules(self):
        """Generate time-based recommendations for all devices."""
        devices = ['ac_status', 'fan', 'lights', 'laundry_machine']
        rules = []

        for device in devices:
            grouped = self.data.groupby(['hour', device]).size().unstack(fill_value=0)
            if 'on' in grouped.columns:
                peak_hours = grouped['on'].idxmax()
                rules.append({
                    'device': device,
                    'recommendation': 'on',
                    'recommended_time': f'{peak_hours}:00 to {(peak_hours + 7) % 24}:00'
                })

        return rules
