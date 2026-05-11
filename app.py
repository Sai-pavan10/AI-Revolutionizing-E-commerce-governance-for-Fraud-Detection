import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from data_processor import DataProcessor
from visualization import Visualization
from model_trainer import ModelTrainer
import pandas as pd
import numpy as np
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Initialize data processor, visualization and model trainer
data_processor = None
visualization = None
model_trainer = None

# Load the data on startup
def initialize_components():
    global data_processor, visualization, model_trainer
    try:
        data_file = 'Dataset/Fraudulent_E-Commerce_Transaction_Data_2.csv'
        data_processor = DataProcessor(data_file)
        data_processor.preprocess_data()
        
        visualization = Visualization(data_processor)
        
        model_trainer = ModelTrainer(data_processor)
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")

# Initialize components on startup
initialize_components()

@app.route('/')
def index():
    """Render the main dashboard page"""
    if data_processor is None:
        flash("Error: Data not loaded. Please check logs.", "danger")
        return render_template('index.html', data_loaded=False)
    
    # Get basic statistics about the dataset
    stats = data_processor.get_basic_statistics()
    return render_template('dashboard.html', 
                          stats=stats, 
                          data_loaded=True)

@app.route('/eda')
def eda():
    """Render the EDA page with visualizations"""
    if visualization is None:
        flash("Error: Visualization component not initialized.", "danger")
        return redirect(url_for('index'))
    
    try:
        # Generate fraud distribution and correlation plots (most important ones)
        fraud_dist = visualization.get_fraud_distribution()
        correlation = visualization.get_correlation_plot()
        
        # Generate additional visualizations
        payment_method = visualization.get_payment_method_distribution()
        product_category = visualization.get_product_category_distribution()
        hour = visualization.get_transaction_hour_distribution()
        
        # Get basic statistics
        stats = data_processor.get_basic_statistics()
        
        return render_template('eda.html',
                              fraud_dist=fraud_dist,
                              correlation=correlation,
                              payment_method=payment_method,
                              product_category=product_category,
                              hour=hour,
                              stats=stats)
    except Exception as e:
        logger.error(f"Error rendering EDA page: {str(e)}")
        flash(f"Error generating visualizations: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/models')
def models_page():
    """Render the models page"""
    if model_trainer is None:
        flash("Error: Model trainer not initialized.", "danger")
        return redirect(url_for('index'))
    
    available_models = model_trainer.get_available_models()
    return render_template('models.html', models=available_models)

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train selected models and return results"""
    selected_models = request.form.getlist('models')
    
    if not selected_models:
        flash("Please select at least one model to train.", "warning")
        return redirect(url_for('models_page'))
    
    try:
        results = model_trainer.train_and_evaluate(selected_models)
        return render_template('compare.html', results=results)
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        flash(f"Error training models: {str(e)}", "danger")
        return redirect(url_for('models_page'))

@app.route('/api/feature_importance/<model_name>')
def feature_importance(model_name):
    """Get feature importance for a specific model"""
    try:
        importance = model_trainer.get_feature_importance(model_name)
        return jsonify(importance)
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/confusion_matrix/<model_name>')
def confusion_matrix(model_name):
    """Get confusion matrix for a specific model"""
    try:
        cm = model_trainer.get_confusion_matrix(model_name)
        return jsonify(cm)
    except Exception as e:
        logger.error(f"Error getting confusion matrix: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/roc_curve/<model_name>')
def roc_curve(model_name):
    """Get ROC curve data for a specific model"""
    try:
        roc_data = model_trainer.get_roc_curve(model_name)
        return jsonify(roc_data)
    except Exception as e:
        logger.error(f"Error getting ROC curve: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/classification_report/<model_name>')
def classification_report(model_name):
    """Get classification report for a specific model"""
    try:
        report_data = model_trainer.get_classification_report(model_name)
        return jsonify(report_data)
    except Exception as e:
        logger.error(f"Error getting classification report: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Render the prediction page and handle prediction requests"""
    if model_trainer is None:
        flash("Error: Model trainer not initialized.", "danger")
        return redirect(url_for('index'))
    
    # Get available trained models
    trained_models = []
    for model_id, model_info in model_trainer.models.items():
        if model_id in model_trainer.trained_models:
            trained_models.append({
                'id': model_id,
                'name': model_info['name']
            })
    
    # If no models are trained yet, redirect to models page
    if not trained_models:
        flash("Please train at least one model before making predictions.", "warning")
        return redirect(url_for('models_page'))
    
    # Get best model if available
    best_model = model_trainer.get_best_model()
    
    if request.method == 'POST':
        try:
            # Get form data
            transaction_data = {
                'Transaction Amount': float(request.form.get('amount', 0)),
                'Quantity': int(request.form.get('quantity', 1)),
                'Customer Age': int(request.form.get('customer_age', 30)),
                'Transaction Hour': int(request.form.get('transaction_hour', 12)),
                'Account Age Days': int(request.form.get('account_age', 100)),
                'Transaction Day': int(request.form.get('transaction_day', 15)),
                'Transaction Month': int(request.form.get('transaction_month', 6)),
                'Transaction Day of Week': int(request.form.get('day_of_week', 3)),
                'Address Match': 1 if request.form.get('address_match') == 'yes' else 0,
                'Payment Method': request.form.get('payment_method', 'Credit Card'),
                'Product Category': request.form.get('product_category', 'Electronics'),
                'Device Used': request.form.get('device_used', 'Desktop')
            }
            
            model_id = request.form.get('model_id', best_model or trained_models[0]['id'])
            
            # Make prediction
            prediction_result = model_trainer.predict_transaction(transaction_data, model_id)
            
            return render_template('prediction_result.html', 
                                 result=prediction_result,
                                 transaction=transaction_data,
                                 trained_models=trained_models,
                                 selected_model=model_id)
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            flash(f"Error making prediction: {str(e)}", "danger")
            return render_template('predict.html', 
                                 trained_models=trained_models,
                                 best_model=best_model)
    
    # GET request
    return render_template('predict.html', 
                         trained_models=trained_models,
                         best_model=best_model)

@app.route('/test_data_prediction', methods=['GET', 'POST'])
def test_data_prediction():
    """Render the test data prediction page and handle prediction requests"""
    if model_trainer is None or data_processor is None:
        flash("Error: Required components not initialized.", "danger")
        return redirect(url_for('index'))
    
    # Get available trained models
    trained_models = []
    for model_id, model_info in model_trainer.models.items():
        if model_id in model_trainer.trained_models:
            trained_models.append({
                'id': model_id,
                'name': model_info['name']
            })
    
    # If no models are trained yet, redirect to models page
    if not trained_models:
        flash("Please train at least one model before making predictions on test data.", "warning")
        return redirect(url_for('models_page'))
    
    # Get best model if available
    best_model = model_trainer.get_best_model()
    
    if request.method == 'POST':
        try:
            # Get selected model
            model_id = request.form.get('model_id', best_model or trained_models[0]['id'])
            
            # Use model to predict on test data
            test_predictions = model_trainer.predict_test_data(model_id)
            
            # Return the results page with predictions and metrics
            return render_template('test_prediction_result.html', 
                                 results=test_predictions,
                                 model_id=model_id,
                                 model_name=model_trainer.models[model_id]['name'],
                                 trained_models=trained_models)
            
        except Exception as e:
            logger.error(f"Error making test data predictions: {str(e)}")
            flash(f"Error making test data predictions: {str(e)}", "danger")
            return render_template('test_data_prediction.html', 
                                 trained_models=trained_models,
                                 best_model=best_model)
    
    # GET request
    return render_template('test_data_prediction.html', 
                         trained_models=trained_models,
                         best_model=best_model)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for fraud prediction"""
    if model_trainer is None:
        return jsonify({"error": "Model trainer not initialized"}), 500
    
    try:
        # Get JSON data
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get model ID or use best model
        model_id = data.get('model_id')
        if not model_id:
            model_id = model_trainer.get_best_model()
            
        if not model_id or model_id not in model_trainer.trained_models:
            # Check if any models are trained
            for model_id in model_trainer.models:
                if model_id in model_trainer.trained_models:
                    break
            else:
                return jsonify({"error": "No trained models available"}), 400
        
        # Make prediction
        result = model_trainer.predict_transaction(data, model_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
