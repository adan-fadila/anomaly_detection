import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


class BayesianModel:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = pd.read_csv(data_file)
        self.model = None
        self.inference = None
        self.prepare_data()
        self.create_bayesian_network()
        self.train_model()

    def prepare_data(self):
        # Discretize numeric columns for Bayesian Network
        self.data['temperature'] = pd.cut(self.data['temperature'], bins=[-float('inf'), 18, 22, 26, float('inf')],
                                          labels=['low', 'normal', 'high', 'very_high'])
        self.data['humidity'] = pd.cut(self.data['humidity'], bins=[-float('inf'), 30, 60, 90, float('inf')],
                                       labels=['low', 'normal', 'high', 'very_high'])
        self.data['distance'] = pd.cut(self.data['distance'], bins=[-float('inf'), 10, 50, 100, float('inf')],
                                       labels=['near', 'medium', 'far', 'very_far'])
        self.data['ac_temperature'] = pd.cut(self.data['ac_temperature'], bins=[-float('inf'), 18, 22, 26, float('inf')],
                                             labels=['low', 'normal', 'high', 'very_high'])
        # Drop unused columns for simplicity
        self.data.drop(columns=['timestamp'], inplace=True)

    def create_bayesian_network(self):
        # Define the structure of the Bayesian Network
        self.model = BayesianNetwork([
            ('temperature', 'ac_status'),
            ('humidity', 'ac_status'),
            ('ac_status', 'ac_energy'),
            ('ac_temperature', 'ac_status'),
            ('distance', 'fan'),
            ('fan', 'fan_energy')
        ])

    def train_model(self):
        # Train the Bayesian Network using the data
        self.model.fit(self.data, estimator=BayesianEstimator, prior_type="BDeu")
        self.inference = VariableElimination(self.model)

    def recommend_rules(self):
        # Mock recommendation logic based on Bayesian Network inference
        evidence = {
            'temperature': 'normal',
            'humidity': 'high',
            'distance': 'near'
        }
        devices = ['ac_status', 'fan']
        recommendations = []
        for device in devices:
            query_result = self.inference.query(variables=[device], evidence=evidence)
            probabilities = query_result.values
            recommendation = 'on' if probabilities[1] > 0.5 else 'off'
            recommendations.append({'device': device, 'recommendation': recommendation})
        return recommendations
