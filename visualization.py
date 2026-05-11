import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import json
import logging
import traceback

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Custom JSON encoder for NumPy types
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)

class Visualization:
    """
    Class for creating visualizations for e-commerce fraud detection
    """
    def __init__(self, data_processor):
        """
        Initialize visualization class with data processor
        
        Args:
            data_processor: DataProcessor instance with loaded data
        """
        self.data_processor = data_processor
        self.raw_data = data_processor.raw_data
        self.processed_data = data_processor.processed_data
    
    def get_fraud_distribution(self):
        """
        Generate fraud distribution visualization with bar graphs before and after SMOTE
        
        Returns:
            str: JSON string of plotly figure for fraud distribution
        """
        try:
            logger.debug("Starting fraud distribution visualization")
            
            # Check if raw data is available
            if self.raw_data is None:
                logger.error("Raw data is None in get_fraud_distribution")
                return json.dumps({})
            
            # Get original data distribution
            fraud_counts = self.raw_data['Is Fraudulent'].value_counts()
            logger.debug(f"Fraud counts shape: {fraud_counts.shape}")
            
            # Get count values - ensure Python native types
            legitimate_count = int(fraud_counts.get(0, 0))
            fraudulent_count = int(fraud_counts.get(1, 0))
            total_count = legitimate_count + fraudulent_count
            
            # Calculate percentages
            legitimate_pct = round((legitimate_count / total_count) * 100, 1)
            fraudulent_pct = round((fraudulent_count / total_count) * 100, 1)
            
            # Create the figure directly with go
            fig = go.Figure()
            
            # Add original data bars
            fig.add_trace(go.Bar(
                x=['Legitimate', 'Fraudulent'],
                y=[legitimate_count, fraudulent_count],
                name='Original Data',
                marker_color='#3498db'
            ))
            
            # Add SMOTE data bars (equal distribution)
            fig.add_trace(go.Bar(
                x=['Legitimate', 'Fraudulent'],
                y=[legitimate_count, legitimate_count],  # Equal counts after SMOTE
                name='After SMOTE',
                marker_color='#e74c3c'
            ))
            
            # Update layout
            fig.update_layout(
                title='Transaction Distribution: Original vs SMOTE Balanced',
                xaxis_title='Transaction Type',
                yaxis_title='Count',
                barmode='group',
                height=400,
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="right", 
                    x=1
                ),
                margin=dict(l=40, r=20, t=60, b=40),
                annotations=[
                    dict(
                        x='Legitimate',
                        y=legitimate_count,
                        text=f"{legitimate_pct}%",
                        showarrow=False,
                        yshift=10
                    ),
                    dict(
                        x='Fraudulent',
                        y=fraudulent_count,
                        text=f"{fraudulent_pct}%",
                        showarrow=False,
                        yshift=10
                    ),
                    dict(
                        x='Fraudulent',
                        y=legitimate_count,
                        text="50%",
                        showarrow=False,
                        yshift=10
                    ),
                    dict(
                        x='Legitimate',
                        y=legitimate_count,
                        text="50%",
                        showarrow=False,
                        yshift=10,
                        xshift=30
                    )
                ]
            )
            
            # Add an annotation explaining SMOTE
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                text="SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the class distribution<br>by generating synthetic samples of the minority class.",
                showarrow=False,
                font=dict(size=10),
                align="center"
            )
            
            # Convert figure to JSON
            result = json.dumps(fig.to_dict(), cls=NpEncoder)
            logger.debug(f"Fraud distribution visualization JSON length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating fraud distribution visualization: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({})
    
    def get_correlation_plot(self):
        """
        Generate correlation heatmap for numerical features
        
        Returns:
            str: JSON string of plotly figure for correlation heatmap
        """
        try:
            # Get numerical columns only
            numerical_cols = [
                'Transaction Amount', 'Quantity', 'Customer Age', 
                'Transaction Hour', 'Account Age Days', 'Is Fraudulent'
            ]
            
            # Filter to include only columns that exist in the data
            available_cols = [col for col in numerical_cols if col in self.raw_data.columns]
            
            # Calculate correlation matrix
            corr_matrix = self.raw_data[available_cols].corr().round(2)
            
            # Convert numpy values to Python native types for JSON serialization
            z_values = corr_matrix.values.tolist()
            x_values = corr_matrix.columns.tolist()
            y_values = corr_matrix.index.tolist()
            
            # Create text annotations
            annotations = []
            for i, row in enumerate(z_values):
                for j, value in enumerate(row):
                    annotations.append({
                        "xref": "x1",
                        "yref": "y1",
                        "x": x_values[j],
                        "y": y_values[i],
                        "text": str(round(value, 2)),
                        "font": {"color": "white" if abs(value) > 0.5 else "black"},
                        "showarrow": False
                    })
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=x_values,
                y=y_values,
                colorscale='Viridis',
                zmin=-1,
                zmax=1
            ))
            
            fig.update_layout(
                title='Correlation Matrix of Numerical Features',
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                annotations=annotations,
                template='plotly_dark',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                plot_bgcolor='rgba(0, 0, 0, 0)'
            )
            
            return json.dumps(fig.to_dict(), cls=NpEncoder)
        except Exception as e:
            logger.error(f"Error creating correlation visualization: {str(e)}")
            return json.dumps({})
    
    def get_time_series_plot(self):
        """
        Generate time series visualization of transactions by date
        
        Returns:
            str: JSON string of plotly figure for time series
        """
        try:
            if self.raw_data is None:
                logger.error("Raw data is None in get_time_series_plot")
                return json.dumps({})
            
            # Check if Transaction Date column exists
            if 'Transaction Date' not in self.raw_data.columns:
                logger.error("Transaction Date column not found in raw data")
                return json.dumps({})
            
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(self.raw_data['Transaction Date']):
                date_series = pd.to_datetime(self.raw_data['Transaction Date'], errors='coerce')
            else:
                date_series = self.raw_data['Transaction Date']
            
            # Create date column and count transactions by date
            self.raw_data['Date'] = date_series.dt.date
            daily_counts = self.raw_data.groupby(['Date', 'Is Fraudulent']).size().reset_index()
            daily_counts.columns = ['Date', 'Is Fraudulent', 'Count']
            
            # Label the fraud status
            daily_counts['Fraud Status'] = daily_counts['Is Fraudulent'].map({0: 'Legitimate', 1: 'Fraudulent'})
            
            # Create line chart
            fig = go.Figure()
            
            # Add traces for legitimate and fraudulent transactions
            for fraud_status in ['Legitimate', 'Fraudulent']:
                filtered_data = daily_counts[daily_counts['Fraud Status'] == fraud_status]
                fig.add_trace(go.Scatter(
                    x=filtered_data['Date'],
                    y=filtered_data['Count'],
                    mode='lines+markers',
                    name=fraud_status,
                    marker_color='#3498db' if fraud_status == 'Legitimate' else '#e74c3c'
                ))
            
            # Update layout
            fig.update_layout(
                title='Transaction Volume Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Transactions',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400,
                margin=dict(l=40, r=20, t=60, b=40)
            )
            
            result = json.dumps(fig.to_dict(), cls=NpEncoder)
            return result
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({})

    def get_payment_method_distribution(self):
        """
        Generate visualization of transaction distribution by payment method
        
        Returns:
            str: JSON string of plotly figure for payment method distribution
        """
        try:
            if self.raw_data is None:
                logger.error("Raw data is None in get_payment_method_distribution")
                return json.dumps({})
            
            # Check if Payment Method column exists
            if 'Payment Method' not in self.raw_data.columns:
                logger.error("Payment Method column not found in raw data")
                return json.dumps({})
            
            # Group by payment method and fraud status
            payment_counts = self.raw_data.groupby(['Payment Method', 'Is Fraudulent']).size().reset_index()
            payment_counts.columns = ['Payment Method', 'Is Fraudulent', 'Count']
            
            # Label the fraud status
            payment_counts['Fraud Status'] = payment_counts['Is Fraudulent'].map({0: 'Legitimate', 1: 'Fraudulent'})
            
            # Create bar chart
            fig = px.bar(
                payment_counts,
                x='Payment Method',
                y='Count',
                color='Fraud Status',
                barmode='group',
                color_discrete_map={'Legitimate': '#3498db', 'Fraudulent': '#e74c3c'},
                title='Transaction Distribution by Payment Method'
            )
            
            # Calculate fraud rate for each payment method
            fraud_rates = []
            for method in payment_counts['Payment Method'].unique():
                method_data = payment_counts[payment_counts['Payment Method'] == method]
                total = method_data['Count'].sum()
                fraud = method_data[method_data['Fraud Status'] == 'Fraudulent']['Count'].sum() if 'Fraudulent' in method_data['Fraud Status'].values else 0
                fraud_rate = round((fraud / total) * 100, 1) if total > 0 else 0
                fraud_rates.append({
                    'method': method,
                    'rate': fraud_rate,
                    'y_pos': method_data['Count'].max() + 50  # Position above the highest bar
                })
            
            # Add fraud rate annotations
            for item in fraud_rates:
                fig.add_annotation(
                    x=item['method'],
                    y=item['y_pos'],
                    text=f"Fraud: {item['rate']}%",
                    showarrow=False,
                    font=dict(size=10, color='#e74c3c', family="Arial Black")
                )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Payment Method',
                yaxis_title='Number of Transactions',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400,
                margin=dict(l=40, r=20, t=60, b=40)
            )
            
            result = json.dumps(fig.to_dict(), cls=NpEncoder)
            return result
            
        except Exception as e:
            logger.error(f"Error creating payment method distribution: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({})
    
    def get_product_category_distribution(self):
        """
        Generate visualization of transaction distribution by product category
        
        Returns:
            str: JSON string of plotly figure for product category distribution
        """
        try:
            if self.raw_data is None:
                logger.error("Raw data is None in get_product_category_distribution")
                return json.dumps({})
            
            # Check if Product Category column exists
            if 'Product Category' not in self.raw_data.columns:
                logger.error("Product Category column not found in raw data")
                return json.dumps({})
            
            # Count transactions by product category
            category_counts = self.raw_data['Product Category'].value_counts().reset_index()
            category_counts.columns = ['Product Category', 'Count']
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=category_counts['Product Category'], 
                values=category_counts['Count'],
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3
            )])
            
            # Calculate fraud rate by category
            fraud_by_category = self.raw_data[self.raw_data['Is Fraudulent'] == 1]['Product Category'].value_counts()
            
            # Add annotations for fraud rate on each slice
            annotations = []
            for i, category in enumerate(category_counts['Product Category']):
                total = category_counts.loc[category_counts['Product Category'] == category, 'Count'].values[0]
                fraud = fraud_by_category.get(category, 0)
                fraud_rate = round((fraud / total) * 100, 1)
                
                annotations.append(dict(
                    text=f"{fraud_rate}% fraud",
                    x=0.5,
                    y=0.5,
                    font=dict(size=8, color='#e74c3c'),
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=0
                ))
            
            # Update layout
            fig.update_layout(
                title='Transaction Distribution by Product Category',
                height=400,
                showlegend=True,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            result = json.dumps(fig.to_dict(), cls=NpEncoder)
            return result
            
        except Exception as e:
            logger.error(f"Error creating product category distribution: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({})
    
    
    
    def get_transaction_hour_distribution(self):
        """
        Generate visualization of transaction distribution by hour of day
        
        Returns:
            str: JSON string of plotly figure for hour distribution
        """
        try:
            if self.raw_data is None:
                logger.error("Raw data is None in get_transaction_hour_distribution")
                return json.dumps({})
            
            # Check if Transaction Hour column exists
            if 'Transaction Hour' not in self.raw_data.columns:
                logger.error("Transaction Hour column not found in raw data")
                return json.dumps({})
            
            # Group by hour and fraud status
            hour_data = self.raw_data.groupby(['Transaction Hour', 'Is Fraudulent']).size().reset_index()
            hour_data.columns = ['Hour', 'Is Fraudulent', 'Count']
            
            # Label the fraud status
            hour_data['Fraud Status'] = hour_data['Is Fraudulent'].map({0: 'Legitimate', 1: 'Fraudulent'})
            
            # Create bar chart
            fig = px.bar(
                hour_data,
                x='Hour',
                y='Count',
                color='Fraud Status',
                barmode='group',
                color_discrete_map={'Legitimate': '#3498db', 'Fraudulent': '#e74c3c'},
                title='Transaction Distribution by Hour of Day'
            )
            
            # Calculate fraud rate for each hour
            hour_totals = hour_data.groupby('Hour')['Count'].sum().reset_index()
            hour_fraud = hour_data[hour_data['Fraud Status'] == 'Fraudulent'].groupby('Hour')['Count'].sum().reset_index()
            hour_fraud.columns = ['Hour', 'Fraud Count']
            
            # Merge to get fraud percentage
            hour_analysis = pd.merge(hour_totals, hour_fraud, on='Hour', how='left')
            hour_analysis['Fraud Count'] = hour_analysis['Fraud Count'].fillna(0)
            hour_analysis['Fraud Rate'] = round((hour_analysis['Fraud Count'] / hour_analysis['Count']) * 100, 1)
            
            # Add line trace for fraud rate
            fig.add_trace(go.Scatter(
                x=hour_analysis['Hour'],
                y=hour_analysis['Fraud Rate'],
                mode='lines+markers',
                name='Fraud Rate (%)',
                yaxis='y2',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=8)
            ))
            
            # Update layout for dual y-axis
            fig.update_layout(
                xaxis_title='Hour of Day (24h format)',
                yaxis_title='Number of Transactions',
                yaxis2=dict(
                    title='Fraud Rate (%)',
                    titlefont=dict(color='#2ecc71'),
                    tickfont=dict(color='#2ecc71'),
                    anchor="x",
                    overlaying="y",
                    side="right",
                    range=[0, max(hour_analysis['Fraud Rate']) * 1.2]  # Give some headroom
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            # Ensure hours are in order from 0-23
            fig.update_xaxes(
                categoryorder='array',
                categoryarray=list(range(24))
            )
            
            result = json.dumps(fig.to_dict(), cls=NpEncoder)
            return result
            
        except Exception as e:
            logger.error(f"Error creating hour distribution: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({})
