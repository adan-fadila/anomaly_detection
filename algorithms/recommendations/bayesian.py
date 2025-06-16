import pandas as pd
import numpy as np
import json
import logging
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
        self.target_temp_model = GaussianNB()
        self.ac_mode_model = GaussianNB()
        self.scaler = StandardScaler()
        self.ac_mode_encoder = LabelEncoder()
        
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
            
            # Log data summary statistics for relevant columns only
            relevant_columns = ['living room temperature', 'living room motion', 'light_state', 'ac_state', 'targetTemperature', 'targetAcMode']
            self.logger.info("Data summary statistics for relevant columns:")
            
            for column in relevant_columns:
                if column in self.data.columns:
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
        """Preprocess the data for training (focusing only on relevant columns)"""
        self.logger.info("Starting data preprocessing")
        if self.data is None:
            self.logger.warning("No data loaded. Please load data first.")
            return False
        
        # Convert string boolean values to actual booleans
        self.logger.debug("Converting boolean string values to actual booleans")
        for col in ['living room motion', 'light_state', 'ac_state']:
            if col in self.data.columns and self.data[col].dtype == object:
                self.logger.debug(f"Converting column {col} from {self.data[col].dtype}")
                self.data[col] = self.data[col].map({'true': True, 'false': False})
                
        # Extract features (only temperature and motion - no humidity or spaceId)
        self.logger.info("Extracting features and targets")
        self.features = self.data[['living room temperature', 'living room motion']]
        
        # Convert motion boolean to numeric
        self.logger.debug("Converting motion boolean to numeric")
        self.features['living room motion'] = self.features['living room motion'].astype(int)
        
        # Scale only numerical features (temperature)
        self.logger.info("Scaling temperature feature")
        self.features[['living room temperature']] = self.scaler.fit_transform(self.features[['living room temperature']])
        
        # Prepare targets
        self.light_target = self.data['light_state'].astype(int)
        self.ac_target = self.data['ac_state'].astype(int)
        self.target_temp_target = self.data['targetTemperature']
        
        # Encode AC mode for modeling
        self.ac_mode_encoded = self.ac_mode_encoder.fit_transform(self.data['targetAcMode'])
        
        # Log AC mode mapping
        self.logger.info("AC Mode encoding:")
        for i, mode in enumerate(self.ac_mode_encoder.classes_):
            self.logger.info(f"  {mode} -> {i}")
        
        # Log feature-target relationships
        self._log_feature_target_relationships()
        
        self.logger.info("Data preprocessing completed successfully")
        return True
    
    def _log_feature_target_relationships(self):
        """Log feature-target relationships to understand data patterns"""
        self.logger.info("Analyzing feature-target relationships:")
        
        # For each feature, analyze relationship with targets
        for feature in ['living room temperature', 'living room motion']:
            self.logger.info(f"Feature: {feature}")
            
            # For continuous features (temperature)
            if feature == 'living room temperature':
                # Light state relationship
                light_on_data = self.data[self.data['light_state'] == True]
                light_off_data = self.data[self.data['light_state'] == False]
                
                if len(light_on_data) > 0:
                    light_on_mean = light_on_data[feature].mean()
                    self.logger.info(f"  Light ON average {feature}: {light_on_mean:.2f}")
                if len(light_off_data) > 0:
                    light_off_mean = light_off_data[feature].mean()
                    self.logger.info(f"  Light OFF average {feature}: {light_off_mean:.2f}")
                
                # AC state relationship
                ac_on_data = self.data[self.data['ac_state'] == True]
                ac_off_data = self.data[self.data['ac_state'] == False]
                
                if len(ac_on_data) > 0:
                    ac_on_mean = ac_on_data[feature].mean()
                    self.logger.info(f"  AC ON average {feature}: {ac_on_mean:.2f}")
                if len(ac_off_data) > 0:
                    ac_off_mean = ac_off_data[feature].mean()
                    self.logger.info(f"  AC OFF average {feature}: {ac_off_mean:.2f}")
                
                # Target temperature relationship
                if len(ac_on_data) > 0:
                    target_temp_mean = ac_on_data['targetTemperature'].mean()
                    self.logger.info(f"  Average target temperature when AC ON: {target_temp_mean:.1f}")
            
            # For categorical features (motion)
            else:
                # Create contingency tables
                if len(self.data[self.data[feature] == True]) > 0 and len(self.data[self.data[feature] == False]) > 0:
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
        self._log_model_parameters(self.light_model, "Light")
        
        # Train AC model
        self.logger.info("Training AC model")
        self.ac_model.fit(self.features, self.ac_target)
        self._log_model_parameters(self.ac_model, "AC")
        
        # Train target temperature model (only when AC is on)
        ac_on_data = self.data[self.data['ac_state'] == True]
        if len(ac_on_data) > 0:
            self.logger.info("Training target temperature model")
            ac_on_features = self.features[self.data['ac_state'] == True]
            ac_on_target_temps = self.target_temp_target[self.data['ac_state'] == True]
            self.target_temp_model.fit(ac_on_features, ac_on_target_temps)
            self._log_model_parameters(self.target_temp_model, "Target Temperature")
        
        # Train AC mode model (only when AC is on)
        if len(ac_on_data) > 0:
            self.logger.info("Training AC mode model")
            ac_on_features = self.features[self.data['ac_state'] == True]
            ac_on_modes = self.ac_mode_encoded[self.data['ac_state'] == True]
            self.ac_mode_model.fit(ac_on_features, ac_on_modes)
            self._log_model_parameters(self.ac_mode_model, "AC Mode")
        
        self.logger.info("Models trained successfully")
        return True
    
    def _log_model_parameters(self, model, model_name):
        """Log detailed model parameters with interpretation"""
        self.logger.info(f"{model_name} Model Parameters:")
        
        # Feature names for reference (no humidity)
        feature_names = ['temperature', 'motion']
        
        # Class priors (probability of each class)
        class_priors = model.class_prior_
        classes = model.classes_
        
        # Handle case where only one class exists
        if len(classes) == 1:
            self.logger.warning(f"  WARNING: Only one class found in {model_name} data: {classes[0]}")
            self.logger.info(f"  Prior probability for class {classes[0]}: {class_priors[0]:.4f}")
            return
        
        # Normal case with multiple classes
        self.logger.info(f"  Prior probabilities:")
        for i, class_val in enumerate(classes):
            self.logger.info(f"    {class_val}: {class_priors[i]:.4f}")
        
        # Theta (mean of each feature for each class)
        theta = model.theta_
        self.logger.info(f"  Feature means for each class:")
        for class_idx, class_val in enumerate(classes):
            self.logger.info(f"    Class {class_val}:")
            for feat_idx, feat_name in enumerate(feature_names):
                self.logger.info(f"      {feat_name}: {theta[class_idx, feat_idx]:.4f}")
    
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
        
        return thresholds
    
    def _get_most_common_target_temp_and_mode(self, condition_filters=None):
        """
        Get the most common target temperature and AC mode for given conditions
        
        Args:
            condition_filters: List of tuples like [('column', 'operator', value), ...]
                             e.g., [('living room temperature', '>', 25), ('living room motion', '==', True)]
        """
        filtered_data = self.data.copy()
        
        if condition_filters:
            for col, operator, value in condition_filters:
                if operator == '>':
                    filtered_data = filtered_data[filtered_data[col] > value]
                elif operator == '<':
                    filtered_data = filtered_data[filtered_data[col] < value]
                elif operator == '==':
                    filtered_data = filtered_data[filtered_data[col] == value]
                elif operator == '!=':
                    filtered_data = filtered_data[filtered_data[col] != value]
        
        # Filter only when AC is on
        ac_on_data = filtered_data[filtered_data['ac_state'] == True]
        
        if len(ac_on_data) == 0:
            return None, None
        
        # Get most common target temperature
        most_common_temp = ac_on_data['targetTemperature'].mode()
        target_temp = most_common_temp.iloc[0] if len(most_common_temp) > 0 else 22
        
        # Get most common AC mode
        most_common_mode = ac_on_data['targetAcMode'].mode()
        ac_mode = most_common_mode.iloc[0] if len(most_common_mode) > 0 else 'cool'
        
        self.logger.debug(f"For conditions {condition_filters}:")
        self.logger.debug(f"  Most common target temp: {target_temp}")
        self.logger.debug(f"  Most common AC mode: {ac_mode}")
        
        return int(target_temp), ac_mode
    
    def _calculate_combined_conditional_probability(self, conditions, target_col, target_val):
        """
        Calculate conditional probability for multiple conditions combined with AND
        
        Args:
            conditions: List of tuples [(col, operator, value), ...]
            target_col: Target column name
            target_val: Target value to check
            
        Returns:
            Probability P(target_val | condition1 AND condition2 AND ...)
        """
        filtered_data = self.data.copy()
        
        # Apply all conditions
        for col, operator, value in conditions:
            if operator == '>':
                filtered_data = filtered_data[filtered_data[col] > value]
            elif operator == '<':
                filtered_data = filtered_data[filtered_data[col] < value]
            elif operator == '==':
                filtered_data = filtered_data[filtered_data[col] == value]
            elif operator == '!=':
                filtered_data = filtered_data[filtered_data[col] != value]
        
        if len(filtered_data) == 0:
            return 0
        
        count_matches = len(filtered_data[filtered_data[target_col] == target_val])
        prob = count_matches / len(filtered_data)
        
        condition_str = " AND ".join([f"{col} {op} {val}" for col, op, val in conditions])
        self.logger.debug(f"P({target_col}={target_val} | {condition_str}) = {prob:.4f} ({count_matches}/{len(filtered_data)})")
        
        return prob
    
    def _calculate_combined_conditional_probability_or(self, conditions, target_col, target_val):
        """
        Calculate conditional probability for multiple conditions combined with OR
        
        Args:
            conditions: List of tuples [(col, operator, value), ...]
            target_col: Target column name
            target_val: Target value to check
            
        Returns:
            Probability P(target_val | condition1 OR condition2 OR ...)
        """
        # Create a mask for OR conditions
        or_mask = pd.Series([False] * len(self.data), index=self.data.index)
        
        for col, operator, value in conditions:
            if operator == '>':
                condition_mask = self.data[col] > value
            elif operator == '<':
                condition_mask = self.data[col] < value
            elif operator == '==':
                condition_mask = self.data[col] == value
            elif operator == '!=':
                condition_mask = self.data[col] != value
            else:
                condition_mask = pd.Series([False] * len(self.data), index=self.data.index)
            
            or_mask = or_mask | condition_mask
        
        filtered_data = self.data[or_mask]
        
        if len(filtered_data) == 0:
            return 0
        
        count_matches = len(filtered_data[filtered_data[target_col] == target_val])
        prob = count_matches / len(filtered_data)
        
        condition_str = " OR ".join([f"{col} {op} {val}" for col, op, val in conditions])
        self.logger.debug(f"P({target_col}={target_val} | {condition_str}) = {prob:.4f} ({count_matches}/{len(filtered_data)})")
        
        return prob
    
    def generate_rules(self):
        """Generate rules based on the models and data patterns"""
        self.logger.info("Starting rule generation")
        self.rules = []
        thresholds = self._get_threshold_values()
        
        # Calculate and log all conditional probabilities
        self._calculate_all_conditional_probabilities(thresholds)
        
        # Generate single-condition rules (original functionality)
        self._generate_motion_rules()
        self._generate_temperature_rules(thresholds)
        
        # NEW: Generate multi-condition rules
        self._generate_multi_condition_rules(thresholds)
        
        # If no rules were generated, add default ones based on model parameters
        if not self.rules:
            self._generate_default_rules(thresholds)
        
        self.logger.info(f"Rule generation completed. Generated {len(self.rules)} rules.")
        return self.rules
    
    def _generate_multi_condition_rules(self, thresholds):
        """Generate rules with multiple conditions (temperature AND/OR motion)"""
        self.logger.info("Generating multi-condition rules")
        
        # Probability threshold for generating rules
        probability_threshold = 0.7  # Higher threshold for multi-condition rules
        
        # Define condition combinations to test
        temp_conditions = [
            ('living room temperature', '>', thresholds['temp_high']),
            ('living room temperature', '<', thresholds['temp_low'])
        ]
        
        motion_conditions = [
            ('living room motion', '==', True),
            ('living room motion', '==', False)
        ]
        
        # Test AND combinations
        self.logger.info("Testing AND combinations:")
        for temp_cond in temp_conditions:
            for motion_cond in motion_conditions:
                conditions = [temp_cond, motion_cond]
                self._test_and_add_combined_rule(conditions, "AND", thresholds, probability_threshold)
        
        # Test OR combinations
        self.logger.info("Testing OR combinations:")
        for temp_cond in temp_conditions:
            for motion_cond in motion_conditions:
                conditions = [temp_cond, motion_cond]
                self._test_and_add_combined_rule(conditions, "OR", thresholds, probability_threshold)
    
    def _test_and_add_combined_rule(self, conditions, operator, thresholds, probability_threshold):
        """Test a combination of conditions and add rule if probability is high enough"""
        
        # Test for LIGHT rules
        for light_state in [True, False]:
            if operator == "AND":
                prob = self._calculate_combined_conditional_probability(conditions, 'light_state', light_state)
            else:  # OR
                prob = self._calculate_combined_conditional_probability_or(conditions, 'light_state', light_state)
            
            if prob > probability_threshold:
                rule_str = self._format_combined_rule(conditions, operator, "LIGHT", "on" if light_state else "off")
                if rule_str not in self.rules:  # Avoid duplicates
                    self.rules.append(rule_str)
                    self.logger.info(f"Added multi-condition LIGHT rule: {rule_str} (prob={prob:.3f})")
        
        # Test for AC rules
        for ac_state in [True, False]:
            if operator == "AND":
                prob = self._calculate_combined_conditional_probability(conditions, 'ac_state', ac_state)
            else:  # OR
                prob = self._calculate_combined_conditional_probability_or(conditions, 'ac_state', ac_state)
            
            if prob > probability_threshold:
                if ac_state:  # AC ON - need target temp and mode
                    target_temp, ac_mode = self._get_most_common_target_temp_and_mode(conditions)
                    if target_temp and ac_mode:
                        rule_str = self._format_combined_rule(conditions, operator, "AC", f"on {target_temp} {ac_mode}")
                else:  # AC OFF
                    rule_str = self._format_combined_rule(conditions, operator, "AC", "off")
                
                if rule_str not in self.rules:  # Avoid duplicates
                    self.rules.append(rule_str)
                    self.logger.info(f"Added multi-condition AC rule: {rule_str} (prob={prob:.3f})")
    
    def _format_combined_rule(self, conditions, operator, device, action):
        """Format a multi-condition rule string"""
        condition_strs = []
        
        for col, op, value in conditions:
            if col == 'living room temperature':
                condition_strs.append(f"Living Room temperature {op} {value}")
            elif col == 'living room motion':
                motion_str = "true" if value else "false"
                condition_strs.append(f"Living Room motion {motion_str}")
        
        condition_part = f" {operator} ".join(condition_strs)
        return f"if {condition_part} then Living Room {device} {action}"
    
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
        
        self.logger.info(f"  P(ac=on|temp>{thresholds['temp_high']}) = {high_temp_table['ac_on']:.4f}")
        
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
        
        self.logger.info(f"  P(ac=on|temp<{thresholds['temp_low']}) = {low_temp_table['ac_on']:.4f}")
    
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
        elif motion_light_off_prob > probability_threshold:
            rule = "if Living Room motion true then Living Room LIGHT off"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
        
        # Motion → AC rules with target temp and mode
        motion_ac_on_prob = self.probability_tables['motion_ac']['ac_on']
        motion_ac_off_prob = self.probability_tables['motion_ac']['ac_off']
        
        if motion_ac_on_prob > probability_threshold:
            target_temp, ac_mode = self._get_most_common_target_temp_and_mode([('living room motion', '==', True)])
            if target_temp and ac_mode:
                rule = f"if Living Room motion true then Living Room AC on {target_temp} {ac_mode}"
                self.rules.append(rule)
                self.logger.info(f"Added rule: {rule}")
        elif motion_ac_off_prob > probability_threshold:
            rule = "if Living Room motion true then Living Room AC off"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
    
    def _generate_temperature_rules(self, thresholds):
        """Generate rules based on temperature readings"""
        self.logger.info("Generating temperature-based rules")
        
        # Threshold for rule generation (probability must be above this)
        probability_threshold = 0.6
        
        # High temperature rules
        high_temp_ac_on_prob = self.probability_tables['high_temp']['ac_on']
        
        # High temp → AC rule with target temp and mode
        if high_temp_ac_on_prob > probability_threshold:
            target_temp, ac_mode = self._get_most_common_target_temp_and_mode([
                ('living room temperature', '>', thresholds['temp_high'])
            ])
            if target_temp and ac_mode:
                rule = f"if Living Room temperature > {thresholds['temp_high']} then Living Room AC on {target_temp} {ac_mode}"
                self.rules.append(rule)
                self.logger.info(f"Added rule: {rule}")
        
        # Low temperature rules
        low_temp_ac_off_prob = self.probability_tables['low_temp']['ac_off']
        
        # Low temp → AC rule
        if low_temp_ac_off_prob > probability_threshold:
            rule = f"if Living Room temperature < {thresholds['temp_low']} then Living Room AC off"
            self.rules.append(rule)
            self.logger.info(f"Added rule: {rule}")
    
    def _generate_default_rules(self, thresholds):
        """Generate default rules based on model parameters when no clear patterns emerge from data"""
        self.logger.warning("No rules generated from data patterns. Falling back to model parameters.")
        
        # Check if models have enough classes for rule generation
        ac_classes = self.ac_model.classes_
        
        # Use GaussianNB parameters to create rules
        temp_index = 0  # Index of temperature in feature array
        
        # AC model rules
        if len(ac_classes) >= 2:
            ac_temp_importance = self.ac_model.theta_[1][temp_index] - self.ac_model.theta_[0][temp_index]
            self.logger.info(f"  AC model temperature importance: {ac_temp_importance:.4f}")
            
            # If temperature has positive impact on AC being on
            if ac_temp_importance > 0:
                target_temp, ac_mode = self._get_most_common_target_temp_and_mode()
                if target_temp and ac_mode:
                    rule = f"if Living Room temperature > {thresholds['temp_medium']} then Living Room AC on {target_temp} {ac_mode}"
                    self.rules.append(rule)
                    self.logger.info(f"Added default rule: {rule}")
            else:
                rule = f"if Living Room temperature < {thresholds['temp_medium']} then Living Room AC off"
                self.rules.append(rule)
                self.logger.info(f"Added default rule: {rule}")
        else:
            self.logger.warning(f"AC model has only one class ({ac_classes[0]}), cannot generate temperature-based AC rules")
            # Generate a simple rule based on the single class
            if ac_classes[0] == 1:  # If AC is always on
                target_temp, ac_mode = self._get_most_common_target_temp_and_mode()
                if target_temp and ac_mode:
                    rule = f"if Living Room temperature > {thresholds['temp_low']} then Living Room AC on {target_temp} {ac_mode}"
                    self.rules.append(rule)
                    self.logger.info(f"Added default rule: {rule}")
    
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
        """Print all generated rules with enhanced explanations for multi-condition rules"""
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
            parts = rule.split(' then ')
            condition = parts[0][3:]  # Remove "if " prefix
            action = parts[1]
            
            # Enhanced explanation based on rule type
            if " AND " in condition:
                self._explain_and_rule(condition, action)
            elif " OR " in condition:
                self._explain_or_rule(condition, action)
            elif "motion true" in condition:
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
    
    def _explain_and_rule(self, condition, action):
        """Explain AND-based multi-condition rules"""
        self.logger.info(f"  Explanation: This is a multi-condition rule using AND operator")
        self.logger.info(f"  All conditions must be true simultaneously for the action to trigger")
        
        # Parse conditions
        and_parts = condition.split(' AND ')
        temp_condition = None
        motion_condition = None
        
        for part in and_parts:
            if "temperature" in part:
                temp_condition = part.strip()
            elif "motion" in part:
                motion_condition = part.strip()
        
        if temp_condition and motion_condition:
            self.logger.info(f"  When BOTH [{temp_condition}] AND [{motion_condition}] are true,")
            self.logger.info(f"  the system performs: {action}")
    
    def _explain_or_rule(self, condition, action):
        """Explain OR-based multi-condition rules"""
        self.logger.info(f"  Explanation: This is a multi-condition rule using OR operator")
        self.logger.info(f"  Any one of the conditions can trigger the action")
        
        # Parse conditions
        or_parts = condition.split(' OR ')
        temp_condition = None
        motion_condition = None
        
        for part in or_parts:
            if "temperature" in part:
                temp_condition = part.strip()
            elif "motion" in part:
                motion_condition = part.strip()
        
        if temp_condition and motion_condition:
            self.logger.info(f"  When EITHER [{temp_condition}] OR [{motion_condition}] is true,")
            self.logger.info(f"  the system performs: {action}")
    
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
        
        # Data summary (only relevant columns)
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
        
        # Target temperature and AC mode analysis
        if ac_on_count > 0:
            ac_on_data = self.data[self.data['ac_state'] == True]
            avg_target_temp = ac_on_data['targetTemperature'].mean()
            most_common_mode = ac_on_data['targetAcMode'].mode().iloc[0] if len(ac_on_data['targetAcMode'].mode()) > 0 else 'N/A'
            self.logger.info(f"Average target temperature when AC is ON: {avg_target_temp:.1f}")
            self.logger.info(f"Most common AC mode: {most_common_mode}")
        
        # Enhanced rule count by type
        motion_rules = sum(1 for rule in self.rules if "motion true" in rule and " AND " not in rule and " OR " not in rule)
        temp_rules = sum(1 for rule in self.rules if "temperature" in rule and " AND " not in rule and " OR " not in rule)
        and_rules = sum(1 for rule in self.rules if " AND " in rule)
        or_rules = sum(1 for rule in self.rules if " OR " in rule)
        enhanced_ac_rules = sum(1 for rule in self.rules if "AC on" in rule and len(rule.split()) > 8)
        
        self.logger.info(f"Generated {len(self.rules)} rules:")
        self.logger.info(f"  - {motion_rules} single motion-based rules")
        self.logger.info(f"  - {temp_rules} single temperature-based rules")
        self.logger.info(f"  - {and_rules} multi-condition AND rules")
        self.logger.info(f"  - {or_rules} multi-condition OR rules")
        self.logger.info(f"  - {enhanced_ac_rules} enhanced AC rules (with target temp and mode)")
        
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
        
        # Multi-condition insights
        if and_rules > 0:
            self.logger.info(f"  - Generated {and_rules} AND rules: these require ALL conditions to be true simultaneously")
        if or_rules > 0:
            self.logger.info(f"  - Generated {or_rules} OR rules: these trigger when ANY condition is true")
        
        self.logger.info("===== END SUMMARY =====\n")
    
    def _calculate_correlation(self, col1, col2):
        """Calculate correlation between two columns"""
        # Convert columns to numeric if needed
        data1 = self.data[col1].astype(int) if self.data[col1].dtype == bool else self.data[col1]
        data2 = self.data[col2].astype(int) if self.data[col2].dtype == bool else self.data[col2]
        
        return np.corrcoef(data1, data2)[0, 1]