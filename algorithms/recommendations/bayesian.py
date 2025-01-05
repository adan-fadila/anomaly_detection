import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

class BayesianRecommendation:
    def __init__(self, log_file):
        self.log_file = log_file
        self.target_columns = ['timestamp', 'lights', 'fan', 'ac_status', 'ac_temperature', 'ac_mode', 'heater_switch', 'laundry_machine']
        self.data = self.parse_log_file()
        self.model = None
        self.inference = None
        self.prepare_data()
        self.create_bayesian_network()
        self.train_model()

    def parse_log_file(self):
        """
        Parse the log file and convert it into a structured DataFrame.
        """
        data = pd.read_csv(self.log_file)
        return data[self.target_columns]

    def prepare_data(self):
        """Prepare the dataset for the Bayesian model."""
        # Extract hour from timestamp
        self.data['hour'] = pd.to_datetime(self.data['timestamp']).dt.hour

        # Process target columns except 'timestamp'
        for column in self.target_columns:
            if column == 'timestamp':
                continue
            if self.data[column].dtype == 'object':
                self.data[column] = self.data[column].astype('category')
            elif self.data[column].dtype in ['float64', 'int64']:
                self.data[column] = pd.to_numeric(self.data[column])

        # Drop the raw timestamp column
        self.data.drop(columns=['timestamp'], inplace=True)

    def create_bayesian_network(self):
        """Define the structure of the Bayesian Network dynamically."""
        edges = []
        columns = [col for col in self.data.columns if col != 'hour']
        for column in columns:
            edges.append(('hour', column))
        self.model = BayesianNetwork(edges)

    def train_model(self):
        """Train the Bayesian Network using the data."""
        self.model.fit(self.data, estimator=BayesianEstimator, prior_type="BDeu")
        self.inference = VariableElimination(self.model)

    def recommend_rules(self):
        """Generate recommendations for each column based on time-based patterns."""
        recommendations = []

        for column in self.data.columns:
            if column == 'hour':
                continue

            if self.data[column].dtype.name == 'category':
                grouped = self.data.groupby(['hour', column]).size().unstack(fill_value=0)

                for value in grouped.columns:
                    peak_hour = grouped[value].idxmax()
                    recommendations.append({
                        'feature': column,
                        'recommendation': value,
                        'recommended_time': f'{peak_hour}:00 to {(peak_hour + 1) % 24}:00'
                    })
            else:  # Numeric columns
                optimal_values = self.data.groupby('hour')[column].mean()
                for hour, value in optimal_values.items():
                    recommendations.append({
                        'feature': column,
                        'recommendation': int(round(value)),
                        'recommended_time': f'{hour}:00 to {(hour + 1) % 24}:00'
                    })

        return recommendations

# Example Usage
# log_file = "path/to/your/logfile.csv"
# recommender = BayesianRecommendation(log_file)
# recommendations = recommender.recommend_rules()
# print(recommendations)
