import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# TensorFlow temporarily disabled
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import joblib
import logging
import json
import tempfile
import os
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training and evaluating fraud detection models
    """
    def __init__(self, data_processor):
        """
        Initialize ModelTrainer with data processor
        
        Args:
            data_processor: DataProcessor instance with processed data
        """
        self.data_processor = data_processor
        self.models = {}
        self.results = {}
        self.trained_models = {}
        self.feature_importances = {}
        self.confusion_matrices = {}
        self.roc_curves = {}
        
        # Define available models
        self._define_models()
    
    def _define_models(self):
        """Define available models for fraud detection"""
        self.models = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'model': LogisticRegression(),
                'supports_feature_importance': True
            },
            'random_forest': {
                'name': 'Random Forest',
                'model': RandomForestClassifier(),
                'supports_feature_importance': True
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'model': GradientBoostingClassifier(),
                'supports_feature_importance': True
            },
            'svm': {
                'name': 'Support Vector Machine',
                'model': SVC(probability=True, class_weight='balanced', random_state=42),
                'supports_feature_importance': False
            },
            'neural_network': {
                'name': 'Neural Network',
                'model': self._create_neural_network,
                'supports_feature_importance': False
            }
        }
    
    def _create_neural_network(self, input_dim):
        """
        Create a neural network model
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            tf.keras.models.Sequential: Neural network model
        """
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def get_available_models(self):
        """
        Get available models for training
        
        Returns:
            list: List of available model names and details
        """
        return [{'id': k, 'name': v['name']} for k, v in self.models.items()]
    
    def train_and_evaluate(self, selected_models):
        """
        Train and evaluate selected models. If models exist as .pkl files, load them instead.
        
        Args:
            selected_models (list): List of model IDs to train
            
        Returns:
            dict: Dictionary with model results
        """
        if not selected_models:
            return {}

        os.makedirs("models", exist_ok=True)

        X_train_transformed = self.data_processor.preprocessing_pipeline.transform(self.data_processor.X_train)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, self.data_processor.y_train)
        X_test_transformed = self.data_processor.preprocessing_pipeline.transform(self.data_processor.X_test)


        self.results = {}
        for model_id in selected_models:
            if model_id not in self.models:
                logger.warning(f"Model {model_id} not found, skipping")
                continue

            model_file = os.path.join("models", f"{model_id}.pkl")
            logger.info(f"Processing model: {self.models[model_id]['name']}")

            try:
                if os.path.exists(model_file) and model_id != 'neural_network':
                    logger.info(f"Loading saved model from {model_file}")
                    model = joblib.load(model_file)
                else:
                    if model_id == 'neural_network':
                        input_dim = X_train_transformed.shape[1]
                        model = self.models[model_id]['model'](input_dim)

                        early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=5, restore_best_weights=True
                        )

                        model.fit(
                            X_train_resampled, y_train_resampled,
                            epochs=25,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[early_stopping],
                            verbose=0
                        )
                    else:
                        model = self.models[model_id]['model']
                        model.fit(X_train_resampled, y_train_resampled)
                        joblib.dump(model, model_file)
                        logger.info(f"Saved model to {model_file}")

                # Predictions
                if model_id == 'neural_network':
                    y_pred = (model.predict(X_test_transformed) > 0.5).astype(int).flatten()
                    y_pred_proba = model.predict(X_test_transformed).flatten()
                else:
                    y_pred = model.predict(X_test_transformed)
                    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

                # Store trained model
                self.trained_models[model_id] = model

                # Metrics
                metrics = self._calculate_metrics(self.data_processor.y_test, y_pred, y_pred_proba)
                self.results[model_id] = {
                    'name': self.models[model_id]['name'],
                    'metrics': metrics
                }

                cm = confusion_matrix(self.data_processor.y_test, y_pred)
                self.confusion_matrices[model_id] = cm.tolist()

                fpr, tpr, thresholds = roc_curve(self.data_processor.y_test, y_pred_proba)
                self.roc_curves[model_id] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'auc': metrics['auc']
                }

                report = classification_report(self.data_processor.y_test, y_pred, output_dict=True)
                self.classification_reports = getattr(self, 'classification_reports', {})
                self.classification_reports[model_id] = report

                if self.models[model_id]['supports_feature_importance']:
                    self._calculate_feature_importance(model_id)

                logger.info(f"Model {self.models[model_id]['name']} processed successfully")

            except Exception as e:
                logger.error(f"Error processing model {model_id}: {str(e)}")
                self.results[model_id] = {
                    'name': self.models[model_id]['name'],
                    'error': str(e)
                }

        return self.results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate evaluation metrics for a model
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_pred_proba (array): Predicted probabilities
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'precision': round(precision_score(y_true, y_pred), 4),
            'recall': round(recall_score(y_true, y_pred), 4),
            'f1': round(f1_score(y_true, y_pred), 4),
            'auc': round(roc_auc_score(y_true, y_pred_proba), 4)
        }
        
        return metrics
    
    def _calculate_feature_importance(self, model_id):
        """
        Calculate feature importance for a model
        
        Args:
            model_id (str): ID of the model
        """
        model = self.trained_models[model_id]
        
        try:
            # Get feature importance based on model type
            if model_id == 'logistic_regression':
                # For logistic regression, feature importance is coefficient magnitude
                importance = np.abs(model.coef_[0])
            else:
                # For tree-based models
                importance = model.feature_importances_
            
            # Map importance to feature names
            feature_names = self.data_processor.feature_names
            
            if len(importance) == len(feature_names):
                importance_dict = dict(zip(feature_names, importance))
                
                # Sort by importance
                sorted_importance = sorted(
                    importance_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Convert to format for visualization
                self.feature_importances[model_id] = [
                    {'feature': name, 'importance': float(imp)} 
                    for name, imp in sorted_importance
                ]
            else:
                logger.warning(
                    f"Feature importance length mismatch: {len(importance)} != {len(feature_names)}"
                )
        except Exception as e:
            logger.error(f"Error calculating feature importance for {model_id}: {str(e)}")
    
    def get_feature_importance(self, model_id):
        """
        Get feature importance for a model
        
        Args:
            model_id (str): ID of the model
            
        Returns:
            list: List of feature importance dictionaries
        """
        if model_id not in self.feature_importances:
            raise ValueError(f"Feature importance not available for model {model_id}")
        
        return self.feature_importances[model_id]
    
    def get_confusion_matrix(self, model_id):
        """
        Get confusion matrix for a model
        
        Args:
            model_id (str): ID of the model
            
        Returns:
            list: Confusion matrix as a 2D list
        """
        if model_id not in self.confusion_matrices:
            raise ValueError(f"Confusion matrix not available for model {model_id}")
        
        return {
            'matrix': self.confusion_matrices[model_id],
            'labels': ['Legitimate', 'Fraudulent']
        }
    
    def get_roc_curve(self, model_id):
        """
        Get ROC curve data for a model
        
        Args:
            model_id (str): ID of the model
            
        Returns:
            dict: Dictionary with ROC curve data
        """
        if model_id not in self.roc_curves:
            raise ValueError(f"ROC curve not available for model {model_id}")
        
        return self.roc_curves[model_id]
        
    def get_classification_report(self, model_id):
        """
        Get classification report for a model
        
        Args:
            model_id (str): ID of the model
            
        Returns:
            dict: Classification report as a dictionary
        """
        if not hasattr(self, 'classification_reports') or model_id not in self.classification_reports:
            raise ValueError(f"Classification report not available for model {model_id}")
        
        return self.classification_reports[model_id]
        
    def predict_transaction(self, transaction_data, model_id='random_forest'):
        """
        Predict whether a transaction is fraudulent
        
        Args:
            transaction_data (dict): Dictionary containing transaction details
            model_id (str): ID of the model to use for prediction
            
        Returns:
            dict: Prediction results including probability and binary prediction
        """
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not trained. Please train the model first.")
        
        try:
            # Create a dataframe from transaction data
            transaction_df = pd.DataFrame([transaction_data])
            
            # Process the transaction data
            processed_data = self._preprocess_transaction(transaction_df)
            
            # Transform data using preprocessing pipeline
            transformed_data = self.data_processor.preprocessing_pipeline.transform(processed_data)
            
            # Make prediction
            model = self.trained_models[model_id]
            
            # For neural network models (when implemented)
            if model_id == 'neural_network':
                probability = model.predict(transformed_data)[0][0]
                prediction = int(probability > 0.5)
            else:
                # For scikit-learn models
                probability = model.predict_proba(transformed_data)[0][1]
                prediction = int(probability > 0.5)
            
            return {
                'prediction': prediction,
                'probability': round(float(probability), 4),
                'model': self.models[model_id]['name'],
                'is_fraudulent': 'Yes' if prediction == 1 else 'No',
                'risk_level': self._get_risk_level(probability)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _preprocess_transaction(self, transaction_df):
        """
        Preprocess a transaction for prediction
        
        Args:
            transaction_df (DataFrame): DataFrame with transaction data
            
        Returns:
            DataFrame: Processed transaction data ready for prediction
        """
        # Create a copy to avoid modifying the original
        df = transaction_df.copy()
        
        # Handle date if present
        if 'Transaction Date' in df.columns:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
            df['Transaction Day'] = df['Transaction Date'].dt.day
            df['Transaction Month'] = df['Transaction Date'].dt.month
            df['Transaction Day of Week'] = df['Transaction Date'].dt.dayofweek
        
        # Process address match if both addresses are provided
        if 'Shipping Address' in df.columns and 'Billing Address' in df.columns:
            df['Address Match'] = df.apply(
                lambda row: self.data_processor._compare_addresses(
                    row['Shipping Address'], row['Billing Address']
                ), 
                axis=1
            )
        elif 'Address Match' not in df.columns:
            # If no address data, default to 0 (no match)
            df['Address Match'] = 0
        
        # Keep only necessary columns for modeling
        keep_columns = [
            'Transaction Amount', 'Quantity', 'Customer Age', 
            'Transaction Hour', 'Account Age Days', 'Transaction Day',
            'Transaction Month', 'Transaction Day of Week', 'Address Match',
            'Payment Method', 'Product Category', 'Device Used'
        ]
        
        # Filter columns that exist in the dataframe
        existing_columns = [col for col in keep_columns if col in df.columns]
        
        # Fill missing columns with default values
        for col in set(keep_columns) - set(existing_columns):
            if col in ['Transaction Amount', 'Quantity', 'Customer Age', 'Transaction Hour', 
                      'Account Age Days', 'Transaction Day', 'Transaction Month', 
                      'Transaction Day of Week', 'Address Match']:
                df[col] = 0
            else:
                df[col] = 'unknown'
        
        return df[keep_columns]
    
    def _get_risk_level(self, probability):
        """
        Convert fraud probability to risk level
        
        Args:
            probability (float): Fraud probability
            
        Returns:
            str: Risk level (Low, Medium, High, Very High)
        """
        if probability < 0.25:
            return 'Low'
        elif probability < 0.5:
            return 'Medium'
        elif probability < 0.75:
            return 'High'
        else:
            return 'Very High'
            
    def _get_categorical_label(self, feature_name, value):
        """
        Convert numeric value back to categorical label for display
        
        Args:
            feature_name (str): Name of the feature
            value (float): Numeric value
            
        Returns:
            str: Categorical label
        """
        # Default mapping for common categories - this should be enhanced based on your actual encoding
        mappings = {
            'Payment Method': {
                0: 'Credit Card',
                1: 'PayPal',
                2: 'Apple Pay',
                3: 'Google Pay',
                4: 'Bank Transfer',
                5: 'Cryptocurrency'
            },
            'Product Category': {
                0: 'Electronics',
                1: 'Clothing',
                2: 'Home & Garden',
                3: 'Beauty',
                4: 'Sports',
                5: 'Toys',
                6: 'Jewelry',
                7: 'Digital Content',
                8: 'Health',
                9: 'Travel'
            },
            'Device Used': {
                0: 'Desktop',
                1: 'Mobile',
                2: 'Tablet'
            }
        }
        
        # Round to the nearest integer if needed
        if isinstance(value, float):
            value = int(round(value))
            
        # Try to get the label from the mapping
        if feature_name in mappings and value in mappings[feature_name]:
            return mappings[feature_name][value]
        else:
            # Return a placeholder if mapping is not available
            return f"{feature_name} {value}"
    
    def predict_test_data(self, model_id):
        """
        Predict fraud on the test dataset and return evaluation metrics
        
        Args:
            model_id (str): ID of the model to use for prediction
            
        Returns:
            dict: Dictionary containing predictions, metrics, and test data
        """
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not trained. Please train the model first.")
        
        try:
            # Get the test data
            X_test = self.data_processor.X_test
            y_test = self.data_processor.y_test
            
            # Get the model
            model = self.trained_models[model_id]
            
            # Make predictions
            if model_id == 'neural_network':
                y_pred_proba = model.predict(X_test).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            # Calculate classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': metrics['auc']
            }
            
            # Create sample of test data with predictions for display (first 100 examples)
            sample_size = min(100, len(X_test))
            
            # Get the original feature names
            feature_names = self.data_processor.feature_names
            
            # Get a sample of raw test data if available
            test_data_sample = []
            
            # Get indices of true positives, true negatives, false positives, and false negatives
            tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
            tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
            fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
            fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
            
            # Get samples from each category
            sample_indices = []
            for indices in [tp_indices, fp_indices, fn_indices, tn_indices]:
                if len(indices) > 0:
                    # Take a proportional number of samples from each category
                    n_samples = max(1, min(25, len(indices)))
                    selected_indices = np.random.choice(indices, size=n_samples, replace=False)
                    sample_indices.extend(selected_indices)
            
            # If we don't have enough samples, add more
            if len(sample_indices) < sample_size:
                remaining = sample_size - len(sample_indices)
                all_indices = set(range(len(X_test)))
                unused_indices = list(all_indices - set(sample_indices))
                if unused_indices:
                    additional_samples = np.random.choice(unused_indices, 
                                                         size=min(remaining, len(unused_indices)), 
                                                         replace=False)
                    sample_indices.extend(additional_samples)
            
            # Limit to sample_size
            sample_indices = sample_indices[:sample_size]
            
            # Get the sample data
            for idx in sample_indices:
                try:
                    # Create a safe copy of the features for display
                    feature_dict = {}
                    for i, feature_name in enumerate(feature_names):
                        # Ensure values are safely converted to the appropriate type for JSON
                        value = X_test[idx][i]
                        # Handle categorical features (non-numeric values)
                        if feature_name in ['Payment Method', 'Product Category', 'Device Used']:
                            # These might be one-hot encoded in X_test, so we need to handle differently
                            if isinstance(value, (int, float, np.integer, np.floating)):
                                # If it's a numeric value from encoding, use a placeholder
                                feature_dict[feature_name] = self._get_categorical_label(feature_name, value)
                            else:
                                feature_dict[feature_name] = str(value)
                        else:
                            # Numeric features
                            feature_dict[feature_name] = float(value) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
                    
                    sample = {
                        'features': feature_dict,
                        'actual': int(y_test[idx]),
                        'predicted': int(y_pred[idx]),
                        'probability': float(y_pred_proba[idx]),
                        'correct': y_test[idx] == y_pred[idx],
                        'category': self._get_prediction_category(y_test[idx], y_pred[idx])
                    }
                    test_data_sample.append(sample)
                except Exception as e:
                    logger.error(f"Error processing test sample at index {idx}: {str(e)}")
                    # Continue with other samples if one fails
            
            # Sort by prediction category for easier viewing
            test_data_sample.sort(key=lambda x: x['category'])
            
            return {
                'metrics': metrics,
                'confusion_matrix': {
                    'matrix': cm,
                    'labels': ['Legitimate', 'Fraudulent']
                },
                'classification_report': report,
                'roc_curve': roc_data,
                'test_data_sample': test_data_sample,
                'model_name': self.models[model_id]['name']
            }
            
        except Exception as e:
            logger.error(f"Error making test data predictions: {str(e)}")
            raise
            
    def _get_prediction_category(self, actual, predicted):
        """
        Get the prediction category (TP, TN, FP, FN)
        
        Args:
            actual (int): Actual class
            predicted (int): Predicted class
            
        Returns:
            str: Category (TP, TN, FP, FN)
        """
        if actual == 1 and predicted == 1:
            return 'TP'  # True Positive
        elif actual == 0 and predicted == 0:
            return 'TN'  # True Negative
        elif actual == 0 and predicted == 1:
            return 'FP'  # False Positive
        else:  # actual == 1 and predicted == 0
            return 'FN'  # False Negative
            
    def get_best_model(self):
        """
        Get the best performing model based on AUC score
        
        Returns:
            str: ID of the best performing model
        """
        if not self.results:
            return None
        
        best_model = None
        best_auc = -1
        
        for model_id, result in self.results.items():
            if 'error' in result:
                continue
                
            auc = result['metrics']['auc']
            if auc > best_auc:
                best_auc = auc
                best_model = model_id
        
        return best_model
