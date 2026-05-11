import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging
import re

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for processing e-commerce transaction data
    """
    def __init__(self, file_path):
        """
        Initialize DataProcessor with the path to the data file
        
        Args:
            file_path (str): Path to the data file
        """
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.preprocessing_pipeline = None
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load data from file"""
        try:
            logger.info(f"Loading data from {self.file_path}")
            self.raw_data = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully: {self.raw_data.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self):
        """Preprocess data for analysis and modeling"""
        if self.raw_data is None:
            logger.error("No data loaded to preprocess")
            return
        
        try:
            # Make a copy of the data
            data = self.raw_data.copy()
            
            # Convert datetime column
            data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])
            
            # Extract additional features from datetime
            data['Transaction Day'] = data['Transaction Date'].dt.day
            data['Transaction Month'] = data['Transaction Date'].dt.month
            data['Transaction Day of Week'] = data['Transaction Date'].dt.dayofweek
            
            # Process shipping and billing address
            data['Address Match'] = data.apply(lambda row: self._compare_addresses(row['Shipping Address'], row['Billing Address']), axis=1)
            
            # Keep only needed columns for processing
            keep_columns = [
                'Transaction Amount', 'Quantity', 'Customer Age', 
                'Transaction Hour', 'Account Age Days', 'Transaction Day',
                'Transaction Month', 'Transaction Day of Week', 'Address Match',
                'Payment Method', 'Product Category', 'Device Used', 'Is Fraudulent'
            ]
            
            self.processed_data = data[keep_columns]
            
            # Split data for modeling
            self._prepare_modeling_data()
            
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def _compare_addresses(self, shipping, billing):
        """
        Compare shipping and billing addresses.
        
        Args:
            shipping (str): Shipping address
            billing (str): Billing address
            
        Returns:
            int: 1 if addresses match, 0 otherwise
        """
        # Clean and standardize addresses
        try:
            if pd.isna(shipping) or pd.isna(billing):
                return 0
                
            # Remove line breaks and standardize case
            shipping = re.sub(r'\s+', ' ', str(shipping).lower().strip())
            billing = re.sub(r'\s+', ' ', str(billing).lower().strip())
            
            return 1 if shipping == billing else 0
        except Exception as e:
            logger.warning(f"Error comparing addresses: {str(e)}")
            return 0
    
    def _prepare_modeling_data(self):
        """Prepare data for modeling by splitting into train/test sets and setting up preprocessing pipeline"""
        # Define features and target
        X = self.processed_data.drop('Is Fraudulent', axis=1)
        y = self.processed_data['Is Fraudulent']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define categorical and numerical features
        categorical_features = ['Payment Method', 'Product Category', 'Device Used']
        numerical_features = [
            'Transaction Amount', 'Quantity', 'Customer Age', 
            'Transaction Hour', 'Account Age Days', 'Transaction Day',
            'Transaction Month', 'Transaction Day of Week', 'Address Match'
        ]
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Fit preprocessing pipeline
        self.preprocessing_pipeline.fit(self.X_train)
        
        # Get feature names
        self._get_feature_names()
    
    def _get_feature_names(self):
        """Extract feature names from preprocessing pipeline"""
        # Get numerical feature names directly
        numerical_features = [
            'Transaction Amount', 'Quantity', 'Customer Age', 
            'Transaction Hour', 'Account Age Days', 'Transaction Day',
            'Transaction Month', 'Transaction Day of Week', 'Address Match'
        ]
        
        # Get categorical feature names from one-hot encoding
        ohe = self.preprocessing_pipeline.named_transformers_['cat'].named_steps['onehot']
        categorical_features = []
        
        # Extract features for each categorical column
        categorical_columns = ['Payment Method', 'Product Category', 'Device Used']
        for i, col in enumerate(categorical_columns):
            categories = ohe.categories_[i]
            for category in categories:
                categorical_features.append(f"{col}_{category}")
        
        # Combine all feature names
        self.feature_names = numerical_features + categorical_features
    
    def get_basic_statistics(self):
        """Get basic statistics about the dataset"""
        if self.raw_data is None:
            return {}
        
        stats = {
            'total_records': len(self.raw_data),
            'fraud_count': self.raw_data['Is Fraudulent'].sum(),
            'fraud_percentage': round(self.raw_data['Is Fraudulent'].mean() * 100, 2),
            'non_fraud_count': len(self.raw_data) - self.raw_data['Is Fraudulent'].sum(),
            'payment_methods': self.raw_data['Payment Method'].value_counts().to_dict(),
            'product_categories': self.raw_data['Product Category'].value_counts().to_dict(),
            'devices': self.raw_data['Device Used'].value_counts().to_dict()
        }
        
        return stats
