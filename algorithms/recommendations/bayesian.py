import pandas as pd
import numpy as np
import json
import logging
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

class BayesianRecommendation:
    def __init__(self, filepath=None):
        # Initialize logger using standard logging
        self.logger = logging.getLogger("SmartHomeRuleGenerator")
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("Initializing SmartHomeRuleGenerator")
        
        # Initialize models for different actions
        self.light_model = GaussianNB()
        self.ac_model = GaussianNB()
        self.scaler = StandardScaler()
        self.data = None
        self.rules = []
        self.probability_tables = {}
        self.filepath = filepath
        self.logger.info("Models and scaler initialized")
        
        # If filepath is provided, load the data immediately
        if filepath:
            self.logger.info(f"Filepath provided in constructor: {filepath}")
            if not self.load_data(filepath):
                self.logger.error(f"Failed to load data from {filepath} during initialization")
    
    def load_data(self, filepath=None):
        """Load data from CSV file"""
        # Use provided filepath or the one from constructor
        file_to_load = filepath if filepath else self.filepath
        
        if not file_to_load:
            self.logger.error("No filepath provided for loading data")
            return False
            
        self.logger.info(f"Attempting to load data from {file_to_load}")
        try:
            self.data = pd.read_csv(file_to_load)
            self.logger.info(f"Successfully loaded data with {len(self.data)} rows")
            
            # Log data summary statistics
            self.logger.info("Data summary statistics:")
            for column in self.data.columns:
                if self.data[column].dtype in [np.float64, np.int64]:
                    self.logger.info(f"  {column}: min={self.data[column].min()}, max={self.data[column].max()}, mean={self.data[column].mean():.2f}, std={self.data[column].std():.2f}")
                else:
                    value_counts = self.data[column].value_counts()
                    self.logger.info(f"  {column}: unique values={len(value_counts)}")
                    for value, count in value_counts.items():
                        self.logger.info(f"    {value}: {count} ({count/len(self.data)*100:.1f}%)")
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.error("Exception details:", exc_info=True)
            return False
    
    def preprocess_data(self):
        """Preprocess the data for training"""
        self.logger.info("Starting data preprocessing")
        if self.data is None:
            self.logger.warning("No data loaded. Please load data first.")
            return False
        
        # Convert string boolean values to actual booleans
        self.logger.debug("Converting boolean string values to actual booleans")
        for col in ['living room motion', 'light_state', 'ac_state']:
            if self.data[col].dtype == object:
                self.logger.debug(f"Converting column {col} from {self.data[col].dtype}")
                self.data[col] = self.data[col].map({'true': True, 'false': False})
                
        # Extract features and targets
        self.logger.info("Extracting features and targets")
        self.features = self.data[['living room temperature', 'living room humidity', 'living room motion']]
        
        # Convert motion boolean to numeric
        self.logger.debug("Converting motion boolean to numeric")
        self.features['living room motion'] = self.features['living room motion'].astype(int)
        
        # Scale numerical features
        self.logger.info("Scaling numerical features")
        numerical_features = ['living room temperature', 'living room humidity']
        self.features[numerical_features] = self.scaler.fit_transform(self.features[numerical_features])
        
        self.light_target = self.data['light_state'].astype(int)
        self.ac_target = self.data['ac_state'].astype(int)
        
        # Log feature-target relationships
        self._log_feature_target_relationships()
        
        self.logger.info("Data preprocessing completed successfully")
        return True
    
    def _log_feature_target_relationships(self):
        """Log feature-target relationships to understand data patterns"""
        self.logger.info("Analyzing feature-target relationships:")
        
        # For each feature, analyze relationship with targets
        for feature in ['living room temperature', 'living room humidity', 'living room motion']:
            self.logger.info(f"Feature: {feature}")
            
            # For continuous features (temperature, humidity)
            if feature in ['living room temperature', 'living room humidity']:
                # Light state relationship
                light_on_mean = self.data[self.data['light_state'] == True][feature].mean()
                light_off_mean = self.data[self.data['light_state'] == False][feature].mean()
                self.logger.info(f"  Light ON average {feature}: {light_on_mean:.2f}")
                self.logger.info(f"  Light OFF average {feature}: {light_off_mean:.2f}")
                
                # AC state relationship
                ac_on_mean = self.data[self.data['ac_state'] == True][feature].mean()
                ac_off_mean = self.data[self.data['ac_state'] == False][feature].mean()
                self.logger.info(f"  AC ON average {feature}: {ac_on_mean:.2f}")
                self.logger.info(f"  AC OFF average {feature}: {ac_off_mean:.2f}")
            
            # For categorical features (motion)
            else:
                # Create contingency tables
                light_motion_table = pd.crosstab(
                    self.data[feature], 
                    self.data['light_state'],
                    normalize='index'
                ) * 100
                
                ac_motion_table = pd.crosstab(
                    self.data[feature], 
                    self.data['ac_state'],
                    normalize='index'
                ) * 100
                
                # Log tables
                self.logger.info(f"  {feature} vs light_state (% of rows):")
                for motion_val in [False, True]:
                    if motion_val in light_motion_table.index:
                        light_on_pct = light_motion_table.loc[motion_val, True] if True in light_motion_table.columns else 0
                        light_off_pct = light_motion_table.loc[motion_val, False] if False in light_motion_table.columns else 0
                        self.logger.info(f"    Motion={motion_val}: Light ON={light_on_pct:.1f}%, Light OFF={light_off_pct:.1f}%")
                
                self.logger.info(f"  {feature} vs ac_state (% of rows):")
                for motion_val in [False, True]:
                    if motion_val in ac_motion_table.index:
                        ac_on_pct = ac_motion_table.loc[motion_val, True] if True in ac_motion_table.columns else 0
                        ac_off_pct = ac_motion_table.loc[motion_val, False] if False in ac_motion_table.columns else 0
                        self.logger.info(f"    Motion={motion_val}: AC ON={ac_on_pct:.1f}%, AC OFF={ac_off_pct:.1f}%")
    
    def train_models(self):
        """Train the Naive Bayes models"""
        self.logger.info("Starting model training")
        if self.features is None:
            self.logger.warning("No processed data available. Please preprocess data first.")
            return False
        
        # Train light model
        self.logger.info("Training light model")
        self.light_model.fit(self.features, self.light_target)
        self.logger.debug(f"Light model class counts: {self.light_model.class_count_}")
        
        # Log light model parameters
        self._log_model_parameters(self.light_model, "Light")
        
        # Train AC model
        self.logger.info("Training AC model")
        self.ac_model.fit(self.features, self.ac_target)
        self.logger.debug(f"AC model class counts: {self.ac_model.class_count_}")
        
        # Log AC model parameters
        self._log_model_parameters(self.ac_model, "AC")
        
        self.logger.info("Models trained successfully")
        return True
    
    def _log_model_parameters(self, model, model_name):
        """Log detailed model parameters with interpretation"""
        self.logger.info(f"{model_name} Model Parameters:")
        
        # Feature names for reference
        feature_names = ['temperature', 'humidity', 'motion']
        
        # Class priors (probability of each class)
        class_priors = model.class_prior_
        classes = model.classes_
        
        # Handle case where only one class exists
        if len(classes) == 1:
            self.logger.warning(f"  WARNING: Only one class found in {model_name} data: {classes[0]}")
            self.logger.info(f"  Prior probability for class {classes[0]}: {class_priors[0]:.4f}")
            
            # Log means for the single class
            theta = model.theta_
            self.logger.info(f"  Feature means for class {classes[0]}:")
            for feat_idx, feat_name in enumerate(feature_names):
                self.logger.info(f"    {feat_name}: {theta[0, feat_idx]:.4f}")
            
            # Log variances for the single class
            variance = model.var_
            self.logger.info(f"  Feature variances for class {classes[0]}:")
            for feat_idx, feat_name in enumerate(feature_names):
                self.logger.info(f"    {feat_name}: {variance[0, feat_idx]:.4f}")
            
            self.logger.warning(f"  Cannot calculate feature importance - only one class available")
            return
        
        # Normal case with two classes
        class_labels = ['OFF', 'ON'] if len(classes) == 2 else [str(c) for c in classes]
        
        self.logger.info(f"  Prior probabilities:")
        for i, (class_val, class_label) in enumerate(zip(classes, class_labels)):
            self.logger.info(f"    {class_label} (value={class_val}): {class_priors[i]:.4f}")
        
        # Theta (mean of each feature for each class)
        theta = model.theta_
        self.logger.info(f"  Feature means for each class:")
        for class_idx, (class_val, class_label) in enumerate(zip(classes, class_labels)):
            self.logger.info(f"    Class {class_label} (value={class_val}):")
            for feat_idx, feat_name in enumerate(feature_names):
                self.logger.info(f"      {feat_name}: {theta[class_idx, feat_idx]:.4f}")
        
        # Variance (variance of each feature for each class)
        variance = model.var_
        self.logger.info(f"  Feature variances for each class:")
        for class_idx, (class_val, class_label) in enumerate(zip(classes, class_labels)):
            self.logger.info(f"    Class {class_label} (value={class_val}):")
            for feat_idx, feat_name in enumerate(feature_names):
                self.logger.info(f"      {feat_name}: {variance[class_idx, feat_idx]:.4f}")
        
        # Feature importance (difference in means between classes)
        if len(classes) == 2:
            self.logger.info(f"  Feature importance (based on theta difference):")
            for feat_idx, feat_name in enumerate(feature_names):
                importance = (theta[1, feat_idx] - theta[0, feat_idx])
                self.logger.info(f"    {feat_name}: {importance:.4f}")
                if importance > 0:
                    self.logger.info(f"      Higher {feat_name} favors {model_name} ON")
                else:
                    self.logger.info(f"      Lower {feat_name} favors {model_name} ON")
        else:
            self.logger.info(f"  Feature importance calculation requires exactly 2 classes")
    
    def _get_threshold_values(self):
        """Extract meaningful threshold values from the data"""
        self.logger.info("Calculating temperature thresholds")
        # Get original data ranges (before scaling)
        temp_mean = np.mean(self.data['living room temperature'])
        temp_std = np.std(self.data['living room temperature'])
        
        # Create thresholds
        thresholds = {
            'temp_low': round(temp_mean - temp_std, 1),
            'temp_high': round(temp_mean + temp_std, 1),
            'temp_medium': round(temp_mean, 1)
        }
        
        # Log threshold calculation
        self.logger.info(f"Temperature statistics: mean={temp_mean:.2f}, std={temp_std:.2f}")
        self.logger.info(f"Calculated thresholds: low={thresholds['temp_low']}, medium={thresholds['temp_medium']}, high={thresholds['temp_high']}")
        
        # Generate visual distribution of temperature
        self._log_temperature_distribution(temp_mean, temp_std, thresholds)
        
        return thresholds
    
    def _log_temperature_distribution(self, mean, std, thresholds):
        """Create a simple text-based visualization of temperature distribution"""
        self.logger.info("Temperature distribution (text visualization):")
        
        # Get min and max temperature
        min_temp = self.data['living room temperature'].min()
        max_temp = self.data['living room temperature'].max()
        
        # Create distribution visualization
        distribution = [' '] * 50  # 50 characters wide
        
        # Mark thresholds
        low_pos = int(((thresholds['temp_low'] - min_temp) / (max_temp - min_temp)) * 49)
        med_pos = int(((thresholds['temp_medium'] - min_temp) / (max_temp - min_temp)) * 49)
        high_pos = int(((thresholds['temp_high'] - min_temp) / (max_temp - min_temp)) * 49)
        
        if 0 <= low_pos < 50:
            distribution[low_pos] = 'L'
        if 0 <= med_pos < 50:
            distribution[med_pos] = 'M'
        if 0 <= high_pos < 50:
            distribution[high_pos] = 'H'
        
        # Create the visualization
        vis = '|' + ''.join(distribution) + '|'
        self.logger.info(f"  {min_temp:.1f}{vis}{max_temp:.1f}")
        self.logger.info(f"  L=Low threshold ({thresholds['temp_low']}), M=Medium ({thresholds['temp_medium']}), H=High ({thresholds['temp_high']})")
    
    def generate_rules(self):
        """Generate rules based on the models and data patterns"""
        self.logger.info("Starting rule generation")
        self.rules = []
        thresholds = self._get_threshold_values()
        
        # Calculate and log all conditional probabilities
        self._calculate_all_conditional_probabilities(thresholds)
        
        # Generate motion-based rules
        self._generate_motion_rules()
        
        # Generate temperature-based rules
        self._generate_temperature_rules(thresholds)
        
        # If no rules were generated, add default ones based on model parameters
        if not self.rules:
            self._generate_default_rules(thresholds)
        
        self.logger.info(f"Rule generation completed. Generated {len(self.rules)} rules.")
        return self.rules
    
    def _calculate_all_conditional_probabilities(self, thresholds):
        """Calculate and log all relevant conditional probabilities"""
        self.logger.info("Calculating all conditional probabilities:")
        
        # Store probabilities in a dictionary for later use
        self.probability_tables = {}
        
        # Motion-based probabilities
        self.logger.info("Motion-based probabilities:")
        
        # P(light=X|motion=true)
        motion_light_table = {
            'light_on': self._calculate_conditional_probability('living room motion', True, 'light_state', True),
            'light_off': self._calculate_conditional_probability('living room motion', True, 'light_state', False)
        }
        self.probability_tables['motion_light'] = motion_light_table
        
        self.logger.info(f"  P(light=on|motion=true) = {motion_light_table['light_on']:.4f}")
        self.logger.info(f"  P(light=off|motion=true) = {motion_light_table['light_off']:.4f}")
        
        # P(ac=X|motion=true)
        motion_ac_table = {
            'ac_on': self._calculate_conditional_probability('living room motion', True, 'ac_state', True),
            'ac_off': self._calculate_conditional_probability('living room motion', True, 'ac_state', False)
        }
        self.probability_tables['motion_ac'] = motion_ac_table
        
        self.logger.info(f"  P(ac=on|motion=true) = {motion_ac_table['ac_on']:.4f}")
        self.logger.info(f"  P(ac=off|motion=true) = {motion_ac_table['ac_off']:.4f}")
        
        # Temperature-based probabilities
        self.logger.info("Temperature-based probabilities:")
        
        # High temperature probabilities
        high_temp_table = {
            'light_on': self._calculate_conditional_probability_numeric(
                'living room temperature', '>', thresholds['temp_high'], 'light_state', True),
            'light_off': self._calculate_conditional_probability_numeric(
                'living room temperature', '>', thresholds['temp_high'], 'light_state', False),
            'ac_on': self._calculate_conditional_probability_numeric(
                'living room temperature', '>', thresholds['temp_high'], 'ac_state', True),
            'ac_off': self._calculate_conditional_probability_numeric(
                'living room temperature', '>', thresholds['temp_high'], 'ac_state', False)
        }
        self.probability_tables['high_temp'] = high_temp_table
        
        self.logger.info(f"  P(light=on|temp>{thresholds['temp_high']}) = {high_temp_table['light_on']:.4f}")
        self.logger.info(f"  P(light=off|temp>{thresholds['temp_high']}) = {high_temp_table['light_off']:.4f}")
        self.logger.info(f"  P(ac=on|temp>{thresholds['temp_high']}) = {high_temp_table['ac_on']:.4f}")
        self.logger.info(f"  P(ac=off|temp>{thresholds['temp_high']}) = {high_temp_table['ac_off']:.4f}")
        
        # Low temperature probabilities
        low_temp_table = {
            'light_on': self._calculate_conditional_probability_numeric(
                'living room temperature', '<', thresholds['temp_low'], 'light_state', True),
            'light_off': self._calculate_conditional_probability_numeric(
                'living room temperature', '<', thresholds['temp_low'], 'light_state', False),
            'ac_on': self._calculate_conditional_probability_numeric(
                'living room temperature', '<', thresholds['temp_low'], 'ac_state', True),
            'ac_off': self._calculate_conditional_probability_numeric(
                'living room temperature', '<', thresholds['temp_low'], 'ac_state', False)
        }
        self.probability_tables['low_temp'] = low_temp_table
        
        self.logger.info(f"  P(light=on|temp<{thresholds['temp_low']}) = {low_temp_table['light_on']:.4f}")
        self.logger.info(f"  P(light=off|temp<{thresholds['temp_low']}) = {low_temp_table['light_off']:.4f}")
        self.logger.info(f"  P(ac=on|temp<{thresholds['temp_low']}) = {low_temp_table['ac_on']:.4f}")
        self.logger.info(f"  P(ac=off|temp<{thresholds['temp_low']}) = {low_temp_table['ac_off']:.4f}")
    
    def _generate_motion_rules(self):
        """Generate rules based on motion sensor data"""
        self.logger.info("Generating motion-based rules")
        
        # Threshold for rule generation (probability must be above this)
        probability_threshold = 0.6
        
        # Motion → Light rules
        motion_light_on_prob = self.probability_tables['motion_light']['light_on']
        motion_light_off_prob = self.probability_tables['motion_light']['light_off']
        
        # Decide which rule to add based on strongest probability
        if motion_light_on_prob > probability_threshold:
            rule = "if Living Room motion true then Living Room LIGHT on"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
            self.logger.info(f"  Reason: When motion is detected, the light is ON {motion_light_on_prob*100:.1f}% of the time")
        elif motion_light_off_prob > probability_threshold:
            rule = "if Living Room motion true then Living Room LIGHT off"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
            self.logger.info(f"  Reason: When motion is detected, the light is OFF {motion_light_off_prob*100:.1f}% of the time")
        else:
            self.logger.info(f"No motion→light rule added: probabilities too low (ON: {motion_light_on_prob*100:.1f}%, OFF: {motion_light_off_prob*100:.1f}%)")
        
        # Motion → AC rules
        motion_ac_on_prob = self.probability_tables['motion_ac']['ac_on']
        motion_ac_off_prob = self.probability_tables['motion_ac']['ac_off']
        
        # Decide which rule to add based on strongest probability
        if motion_ac_on_prob > probability_threshold:
            rule = "if Living Room motion true then Living Room AC on"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
            self.logger.info(f"  Reason: When motion is detected, the AC is ON {motion_ac_on_prob*100:.1f}% of the time")
        elif motion_ac_off_prob > probability_threshold:
            rule = "if Living Room motion true then Living Room AC off"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
            self.logger.info(f"  Reason: When motion is detected, the AC is OFF {motion_ac_off_prob*100:.1f}% of the time")
        else:
            self.logger.info(f"No motion→AC rule added: probabilities too low (ON: {motion_ac_on_prob*100:.1f}%, OFF: {motion_ac_off_prob*100:.1f}%)")
    
    def _generate_temperature_rules(self, thresholds):
        """Generate rules based on temperature readings"""
        self.logger.info("Generating temperature-based rules")
        
        # Threshold for rule generation (probability must be above this)
        probability_threshold = 0.6
        
        # High temperature rules
        high_temp_light_on_prob = self.probability_tables['high_temp']['light_on']
        high_temp_ac_on_prob = self.probability_tables['high_temp']['ac_on']
        
        # High temp → Light rule
        if high_temp_light_on_prob > probability_threshold:
            rule = f"if Living Room temperature > {thresholds['temp_high']} then Living Room LIGHT on"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
            self.logger.info(f"  Reason: When temperature > {thresholds['temp_high']}, the light is ON {high_temp_light_on_prob*100:.1f}% of the time")
        
        # High temp → AC rule
        if high_temp_ac_on_prob > probability_threshold:
            rule = f"if Living Room temperature > {thresholds['temp_high']} then Living Room AC on"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
            self.logger.info(f"  Reason: When temperature > {thresholds['temp_high']}, the AC is ON {high_temp_ac_on_prob*100:.1f}% of the time")
        
        # Low temperature rules
        low_temp_light_on_prob = self.probability_tables['low_temp']['light_on']
        low_temp_ac_off_prob = self.probability_tables['low_temp']['ac_off']
        
        # Low temp → Light rule
        if low_temp_light_on_prob > probability_threshold:
            rule = f"if Living Room temperature < {thresholds['temp_low']} then Living Room LIGHT on"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
            self.logger.info(f"  Reason: When temperature < {thresholds['temp_low']}, the light is ON {low_temp_light_on_prob*100:.1f}% of the time")
        
        # Low temp → AC rule
        if low_temp_ac_off_prob > probability_threshold:
            rule = f"if Living Room temperature < {thresholds['temp_low']} then Living Room AC off"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
            self.logger.info(f"  Reason: When temperature < {thresholds['temp_low']}, the AC is OFF {low_temp_ac_off_prob*100:.1f}% of the time")
    
    def _generate_default_rules(self, thresholds):
        """Generate default rules based on model parameters when no clear patterns emerge from data"""
        self.logger.warning("No rules generated from data patterns. Falling back to model parameters.")
        
        # Check if models have enough classes for rule generation
        light_classes = self.light_model.classes_
        ac_classes = self.ac_model.classes_
        
        # Use GaussianNB parameters to create rules
        temp_index = 0  # Index of temperature in feature array
        
        # Log model parameters we're using
        self.logger.info("Using model parameters to generate default rules:")
        
        # Light model rules
        if len(light_classes) >= 2:
            light_temp_importance = self.light_model.theta_[1][temp_index] - self.light_model.theta_[0][temp_index]
            self.logger.info(f"  Light model temperature importance: {light_temp_importance:.4f}")
            
            # If temperature has positive impact on light being on
            if light_temp_importance > 0:
                rule = f"if Living Room temperature > {thresholds['temp_medium']} then Living Room LIGHT on"
                self.rules.append(rule)
                self.logger.info(f"Added default rule: {rule}")
                self.logger.info(f"  Reason: GaussianNB model shows higher temperatures favor LIGHT ON")
            else:
                rule = f"if Living Room temperature < {thresholds['temp_medium']} then Living Room LIGHT on"
                self.rules.append(rule)
                self.logger.info(f"Added default rule: {rule}")
                self.logger.info(f"  Reason: GaussianNB model shows lower temperatures favor LIGHT ON")
        else:
            self.logger.warning(f"Light model has only one class ({light_classes[0]}), cannot generate temperature-based light rules")
            # Generate a simple rule based on the single class
            if light_classes[0] == 1:  # If lights are always on
                rule = f"if Living Room motion true then Living Room LIGHT on"
                self.rules.append(rule)
                self.logger.info(f"Added default rule: {rule}")
                self.logger.info(f"  Reason: Data shows lights are always ON")
            else:  # If lights are always off
                rule = f"if Living Room motion false then Living Room LIGHT off"
                self.rules.append(rule)
                self.logger.info(f"Added default rule: {rule}")
                self.logger.info(f"  Reason: Data shows lights are always OFF")
        
        # AC model rules
        if len(ac_classes) >= 2:
            ac_temp_importance = self.ac_model.theta_[1][temp_index] - self.ac_model.theta_[0][temp_index]
            self.logger.info(f"  AC model temperature importance: {ac_temp_importance:.4f}")
            
            # If temperature has positive impact on AC being on
            if ac_temp_importance > 0:
                rule = f"if Living Room temperature > {thresholds['temp_medium']} then Living Room AC on"
                self.rules.append(rule)
                self.logger.info(f"Added default rule: {rule}")
                self.logger.info(f"  Reason: GaussianNB model shows higher temperatures favor AC ON")
            else:
                rule = f"if Living Room temperature < {thresholds['temp_medium']} then Living Room AC off"
                self.rules.append(rule)
                self.logger.info(f"Added default rule: {rule}")
                self.logger.info(f"  Reason: GaussianNB model shows lower temperatures favor AC OFF")
        else:
            self.logger.warning(f"AC model has only one class ({ac_classes[0]}), cannot generate temperature-based AC rules")
            # Generate a simple rule based on the single class
            if ac_classes[0] == 1:  # If AC is always on
                rule = f"if Living Room temperature > {thresholds['temp_low']} then Living Room AC on"
                self.rules.append(rule)
                self.logger.info(f"Added default rule: {rule}")
                self.logger.info(f"  Reason: Data shows AC is always ON")
            else:  # If AC is always off
                rule = f"if Living Room temperature < {thresholds['temp_high']} then Living Room AC off"
                self.rules.append(rule)
                self.logger.info(f"Added default rule: {rule}")
                self.logger.info(f"  Reason: Data shows AC is always OFF")
    
    def _calculate_conditional_probability(self, condition_col, condition_val, target_col, target_val):
        """Calculate P(target_val | condition_val) from the data"""
        filtered_data = self.data[self.data[condition_col] == condition_val]
        if len(filtered_data) == 0:
            self.logger.warning(f"No data found where {condition_col}={condition_val}")
            return 0
        
        prob = len(filtered_data[filtered_data[target_col] == target_val]) / len(filtered_data)
        return prob
    
    def _calculate_conditional_probability_numeric(self, condition_col, operator, threshold, target_col, target_val):
        """Calculate P(target_val | condition_col operator threshold) from the data"""
        if operator == '>':
            filtered_data = self.data[self.data[condition_col] > threshold]
        elif operator == '<':
            filtered_data = self.data[self.data[condition_col] < threshold]
        else:
            filtered_data = self.data[self.data[condition_col] == threshold]
            
        if len(filtered_data) == 0:
            self.logger.warning(f"No data found where {condition_col} {operator} {threshold}")
            return 0
        
        count_matches = len(filtered_data[filtered_data[target_col] == target_val])
        total_count = len(filtered_data)
        prob = count_matches / total_count
        
        self.logger.debug(f"For condition [{condition_col} {operator} {threshold}]:")
        self.logger.debug(f"  Total matching condition: {total_count} rows")
        self.logger.debug(f"  Matching target [{target_col}={target_val}]: {count_matches} rows")
        self.logger.debug(f"  Probability: {prob:.4f}")
        
        return prob
    
    def recommend_rules(self):
        """
        Complete execution flow to generate rules and return them as JSON.
        This function runs the entire model pipeline and returns the rules in JSON format.
        
        Returns:
            dict: JSON object with 'recommended_rules' key containing list of rule strings
        """
        self.logger.info("Starting recommend_rules function")
        
        # Check if data is loaded
        if self.data is None:
            if self.filepath:
                self.logger.info("Data not loaded, attempting to load from filepath")
                if not self.load_data():
                    error_msg = "Failed to load data. Cannot generate rules."
                    self.logger.error(error_msg)
                    return {"recommended_rules": [], "error": error_msg}
            else:
                error_msg = "No data available and no filepath provided. Cannot generate rules."
                self.logger.error(error_msg)
                return {"recommended_rules": [], "error": error_msg}
        
        # Run the complete pipeline
        try:
            # Preprocess data
            if not self.preprocess_data():
                error_msg = "Failed to preprocess data."
                self.logger.error(error_msg)
                return {"recommended_rules": [], "error": error_msg}
            
            # Train models
            if not self.train_models():
                error_msg = "Failed to train models."
                self.logger.error(error_msg)
                return {"recommended_rules": [], "error": error_msg}
            
            # Generate rules
            rules = self.generate_rules()
            
            # Log final summary
            self._log_final_summary()
            
            # Create JSON response
            result = {
                "recommended_rules": rules
            }
            
            self.logger.info(f"Successfully generated {len(rules)} rules")
            self.logger.info(f"Returning JSON: {result}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during rule generation: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error("Exception details:", exc_info=True)
            return {"recommended_rules": [], "error": error_msg}
    
    def print_rules(self):
        """Print all generated rules"""
        if not self.rules:
            self.logger.warning("No rules generated yet.")
            return
        
        self.logger.info("\nGenerated Rules:")
        for rule in self.rules:
            self.logger.info(rule)
        
        # Print rule summary with explanations
        self.logger.info("\nRule Summary with Explanations:")
        for i, rule in enumerate(self.rules, 1):
            self.logger.info(f"Rule #{i}: {rule}")
            
            # Extract condition and action from rule
            # Format: "if Living Room X Y then Living Room Z W"
            parts = rule.split(' then ')
            condition = parts[0][3:]  # Remove "if " prefix
            action = parts[1]
            
            # Log explanation based on rule type
            if "motion true" in condition:
                # Motion-based rule
                if "LIGHT on" in action:
                    prob = self.probability_tables['motion_light']['light_on'] * 100
                    self.logger.info(f"  Explanation: When motion is detected, the light is ON {prob:.1f}% of the time")
                elif "LIGHT off" in action:
                    prob = self.probability_tables['motion_light']['light_off'] * 100
                    self.logger.info(f"  Explanation: When motion is detected, the light is OFF {prob:.1f}% of the time")
                elif "AC on" in action:
                    prob = self.probability_tables['motion_ac']['ac_on'] * 100
                    self.logger.info(f"  Explanation: When motion is detected, the AC is ON {prob:.1f}% of the time")
                elif "AC off" in action:
                    prob = self.probability_tables['motion_ac']['ac_off'] * 100
                    self.logger.info(f"  Explanation: When motion is detected, the AC is OFF {prob:.1f}% of the time")
            
            elif "temperature >" in condition:
                # High temperature rule
                threshold = float(condition.split('> ')[1])
                if "LIGHT on" in action:
                    prob = self.probability_tables['high_temp']['light_on'] * 100
                    self.logger.info(f"  Explanation: When temperature > {threshold}, the light is ON {prob:.1f}% of the time")
                elif "AC on" in action:
                    prob = self.probability_tables['high_temp']['ac_on'] * 100
                    self.logger.info(f"  Explanation: When temperature > {threshold}, the AC is ON {prob:.1f}% of the time")
            
            elif "temperature <" in condition:
                # Low temperature rule
                threshold = float(condition.split('< ')[1])
                if "LIGHT on" in action:
                    prob = self.probability_tables['low_temp']['light_on'] * 100
                    self.logger.info(f"  Explanation: When temperature < {threshold}, the light is ON {prob:.1f}% of the time")
                elif "AC off" in action:
                    prob = self.probability_tables['low_temp']['ac_off'] * 100
                    self.logger.info(f"  Explanation: When temperature < {threshold}, the AC is OFF {prob:.1f}% of the time")
            
            self.logger.info("") # Add a blank line for readability
    
    def run(self, filepath=None):
        """Complete execution flow"""
        file_to_use = filepath if filepath else self.filepath
        self.logger.info(f"Starting full execution with file: {file_to_use}")
        
        if not self.load_data(filepath):
            self.logger.error("Failed to load data. Aborting execution.")
            return False
        
        if not self.preprocess_data():
            self.logger.error("Failed to preprocess data. Aborting execution.")
            return False
        
        if not self.train_models():
            self.logger.error("Failed to train models. Aborting execution.")
            return False
        
        rules = self.generate_rules()
        self.print_rules()
        
        self.logger.info("Execution completed successfully")
        self.logger.info(f"Generated {len(rules)} rules based on data patterns")
        
        # Log a final summary of the rules and their justifications
        self._log_final_summary()
        
        return rules
    
    def _log_final_summary(self):
        """Generate a final summary of the analysis with key insights"""
        self.logger.info("\n===== ANALYSIS SUMMARY =====")
        
        # Data summary
        self.logger.info(f"Analyzed {len(self.data)} data points")
        self.logger.info(f"Temperature range: {self.data['living room temperature'].min():.1f} to {self.data['living room temperature'].max():.1f}")
        
        # Motion patterns
        motion_count = self.data['living room motion'].sum()
        motion_pct = motion_count / len(self.data) * 100
        self.logger.info(f"Motion detected in {motion_count} samples ({motion_pct:.1f}% of the time)")
        
        # Device states
        light_on_count = self.data['light_state'].sum()
        light_on_pct = light_on_count / len(self.data) * 100
        self.logger.info(f"Light ON in {light_on_count} samples ({light_on_pct:.1f}% of the time)")
        
        ac_on_count = self.data['ac_state'].sum()
        ac_on_pct = ac_on_count / len(self.data) * 100
        self.logger.info(f"AC ON in {ac_on_count} samples ({ac_on_pct:.1f}% of the time)")
        
        # Rule count by type
        motion_rules = sum(1 for rule in self.rules if "motion true" in rule)
        temp_rules = sum(1 for rule in self.rules if "temperature" in rule)
        
        self.logger.info(f"Generated {len(self.rules)} rules:")
        self.logger.info(f"  - {motion_rules} motion-based rules")
        self.logger.info(f"  - {temp_rules} temperature-based rules")
        
        # Key insights about the data
        self.logger.info("\nKey insights:")
        
        # Check if motion correlates with light
        motion_light_corr = self._calculate_correlation('living room motion', 'light_state')
        if abs(motion_light_corr) > 0.3:
            if motion_light_corr > 0:
                self.logger.info(f"  - Motion and light have a positive correlation ({motion_light_corr:.2f}): motion tends to trigger lights")
            else:
                self.logger.info(f"  - Motion and light have a negative correlation ({motion_light_corr:.2f}): motion tends to turn off lights")
        else:
            self.logger.info(f"  - Motion and light have weak correlation ({motion_light_corr:.2f}): no clear pattern")
        
        # Check if temperature correlates with AC
        temp_ac_corr = self._calculate_correlation('living room temperature', 'ac_state')
        if abs(temp_ac_corr) > 0.3:
            if temp_ac_corr > 0:
                self.logger.info(f"  - Temperature and AC have a positive correlation ({temp_ac_corr:.2f}): higher temperatures trigger AC")
            else:
                self.logger.info(f"  - Temperature and AC have a negative correlation ({temp_ac_corr:.2f}): lower temperatures trigger AC (unusual)")
        else:
            self.logger.info(f"  - Temperature and AC have weak correlation ({temp_ac_corr:.2f}): no clear pattern")
        
        self.logger.info("===== END SUMMARY =====\n")
    
    def _calculate_correlation(self, col1, col2):
        """Calculate correlation between two columns"""
        # Convert columns to numeric if needed
        data1 = self.data[col1].astype(int) if self.data[col1].dtype == bool else self.data[col1]
        data2 = self.data[col2].astype(int) if self.data[col2].dtype == bool else self.data[col2]
        
        return np.corrcoef(data1, data2)[0, 1]