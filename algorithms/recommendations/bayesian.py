import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import logging

class BayesianRecommendation:
    def __init__(self, log_file):
        self.log_file = log_file
        self.target_columns = ['timestamp', 'lights', 'fan', 'ac_status',  'heater_switch', 'laundry_machine']
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
        print("Parsing log file...")
        data = pd.read_csv(self.log_file)
        print("Parsed data head:\n", data.head())
        return data[self.target_columns]

    def prepare_data(self):
        """Prepare the dataset for the Bayesian model."""
        print("Preparing data...")
        # Extract hour from timestamp
        self.data['hour'] = pd.to_datetime(self.data['timestamp']).dt.hour
        print("Extracted 'hour' from 'timestamp'. Example:\n", self.data[['timestamp', 'hour']].head())

        # Process target columns except 'timestamp'
        for column in self.target_columns:
            if column == 'timestamp':
                continue
            # Convert string columns to categorical
            if self.data[column].dtype == 'object':
                self.data[column] = self.data[column].astype('category')
                print(f"Converted column '{column}' to categorical.")

            # Handle ac_temperature as a special case
            if column == 'ac_temperature':
                self.data[column] = pd.qcut(
                    self.data[column].astype(float),
                    q=5,
                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
                ).astype('category')
                print(f"Binned 'ac_temperature' into categories:\n{self.data[column].head()}")
                
                

        # Drop the raw timestamp column because it is no longer needed
        self.data.drop(columns=['timestamp'], inplace=True)
        print("Dropped 'timestamp'. Data prepared. Head of the dataset:\n", self.data.head())

    def create_bayesian_network(self):
        """Define the structure of the Bayesian Network dynamically."""
        print("Creating Bayesian Network...")
        edges = []
        columns = [col for col in self.data.columns if col != 'hour']
        for column in columns:
            edges.append(('hour', column))
        self.model = BayesianNetwork(edges)
        print("Bayesian Network structure created with edges:\n", edges)

    def train_model(self):
        """Train the Bayesian Network using the data."""
        print("Training Bayesian Network model...")
        self.model.fit(self.data, estimator=BayesianEstimator, prior_type="BDeu")
        print("Model training complete.")
        print("Model CPDs:")
        for cpd in self.model.get_cpds():
            print(cpd)
        self.inference = VariableElimination(self.model)
        print("Inference engine initialized.")

    def recommend_rules(self):
        """Generate recommendations for each column based on time-based patterns."""
        print("Generating recommendations...")
        recommendations = []

        for column in self.data.columns:
            if column == 'hour':
                continue

            if self.data[column].dtype.name == 'category':
                grouped = self.data.groupby(['hour', column]).size().unstack(fill_value=0)
                print(f"Grouped data for column '{column}':\n{grouped.head()}")

                for value in grouped.columns:
                    peak_hour = grouped[value].idxmax()
                    recommendations.append({
                        'device': column,
                        'recommendation': value,
                        'recommended_time': f'{peak_hour}:00 to {(peak_hour + 1) % 24}:00'
                    })
        print(f"{len(recommendations)} Recommendations generated.")
        print("Recommendations:\n", recommendations)

        return recommendations
