/**
 * Utility functions for creating and updating charts in the fraud detection system
 */

// Create a plot from JSON data
function createPlotFromJson(containerId, jsonData) {
    try {
        const plotData = JSON.parse(jsonData);
        Plotly.newPlot(containerId, plotData.data, plotData.layout, { responsive: true });
    } catch (error) {
        console.error(`Error creating plot for ${containerId}:`, error);
        document.getElementById(containerId).innerHTML = 
            `<div class="alert alert-danger">Error creating visualization: ${error.message}</div>`;
    }
}

// Create a confusion matrix plot
function createConfusionMatrix(containerId, data) {
    try {
        const matrix = data.matrix;
        const labels = data.labels;
        
        const heatmapData = [{
            z: matrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true,
            text: matrix.map(row => row.map(String)),
            texttemplate: '%{text}',
            textfont: { color: 'white' }
        }];
        
        const layout = {
            title: 'Confusion Matrix',
            xaxis: {
                title: 'Predicted Label',
                titlefont: { size: 14 }
            },
            yaxis: {
                title: 'True Label',
                titlefont: { size: 14 }
            },
            annotations: [],
            margin: { t: 50, l: 80, r: 20, b: 80 }
        };
        
        // Add text annotations
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                const result = {
                    x: labels[j],
                    y: labels[i],
                    text: matrix[i][j],
                    font: { color: 'white' },
                    showarrow: false
                };
                layout.annotations.push(result);
            }
        }
        
        Plotly.newPlot(containerId, heatmapData, layout, { responsive: true });
    } catch (error) {
        console.error(`Error creating confusion matrix for ${containerId}:`, error);
        document.getElementById(containerId).innerHTML = 
            `<div class="alert alert-danger">Error creating confusion matrix: ${error.message}</div>`;
    }
}

// Create ROC curve
function createRocCurve(containerId, data) {
    try {
        const plotData = [{
            x: data.fpr,
            y: data.tpr,
            type: 'scatter',
            mode: 'lines',
            name: `ROC Curve (AUC = ${data.auc.toFixed(3)})`,
            line: { color: '#2980b9', width: 2 }
        }, {
            x: [0, 1],
            y: [0, 1],
            type: 'scatter',
            mode: 'lines',
            name: 'Random Classifier',
            line: { color: '#7f7f7f', width: 2, dash: 'dash' }
        }];
        
        const layout = {
            title: 'ROC Curve',
            xaxis: {
                title: 'False Positive Rate',
                range: [0, 1]
            },
            yaxis: {
                title: 'True Positive Rate',
                range: [0, 1]
            },
            legend: {
                x: 0.7,
                y: 0.1
            },
            margin: { t: 50, l: 80, r: 20, b: 80 }
        };
        
        Plotly.newPlot(containerId, plotData, layout, { responsive: true });
    } catch (error) {
        console.error(`Error creating ROC curve for ${containerId}:`, error);
        document.getElementById(containerId).innerHTML = 
            `<div class="alert alert-danger">Error creating ROC curve: ${error.message}</div>`;
    }
}

// Create feature importance bar chart
function createFeatureImportance(containerId, data) {
    try {
        // Sort data by importance
        const sortedData = [...data].sort((a, b) => b.importance - a.importance);
        
        // Take top 15 features for better visualization
        const topFeatures = sortedData.slice(0, 15);
        
        const plotData = [{
            x: topFeatures.map(d => d.importance),
            y: topFeatures.map(d => d.feature),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: '#3498db',
                opacity: 0.8
            }
        }];
        
        const layout = {
            title: 'Feature Importance (Top 15)',
            xaxis: {
                title: 'Importance'
            },
            yaxis: {
                title: '',
                automargin: true
            },
            margin: { l: 200, r: 20, t: 50, b: 80 }
        };
        
        Plotly.newPlot(containerId, plotData, layout, { responsive: true });
    } catch (error) {
        console.error(`Error creating feature importance for ${containerId}:`, error);
        document.getElementById(containerId).innerHTML = 
            `<div class="alert alert-danger">Error creating feature importance: ${error.message}</div>`;
    }
}

// Create classification report table
function createClassificationReport(containerId, data) {
    try {
        const container = document.getElementById(containerId);
        
        // Extract data
        const classReport = data;
        
        // Create table
        let tableHTML = `
            <table class="table table-bordered table-hover">
                <thead class="table-dark">
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Add rows for each class
        for (const [className, metrics] of Object.entries(classReport)) {
            if (className !== 'accuracy' && className !== 'macro avg' && className !== 'weighted avg') {
                const rowClass = className === '1' ? 'table-danger' : 'table-success';
                const displayName = className === '0' ? 'Legitimate (0)' : 'Fraudulent (1)';
                
                tableHTML += `
                    <tr class="${rowClass}">
                        <td><strong>${displayName}</strong></td>
                        <td>${metrics.precision.toFixed(4)}</td>
                        <td>${metrics.recall.toFixed(4)}</td>
                        <td>${metrics.f1score ? metrics.f1score.toFixed(4) : metrics['f1-score'].toFixed(4)}</td>
                        <td>${metrics.support}</td>
                    </tr>
                `;
            }
        }
        
        // Add summary rows
        if (classReport['macro avg']) {
            tableHTML += `
                <tr class="table-secondary">
                    <td><strong>Macro Avg</strong></td>
                    <td>${classReport['macro avg'].precision.toFixed(4)}</td>
                    <td>${classReport['macro avg'].recall.toFixed(4)}</td>
                    <td>${classReport['macro avg'].f1score ? 
                          classReport['macro avg'].f1score.toFixed(4) : 
                          classReport['macro avg']['f1-score'].toFixed(4)}</td>
                    <td>${classReport['macro avg'].support}</td>
                </tr>
            `;
        }
        
        if (classReport['weighted avg']) {
            tableHTML += `
                <tr class="table-secondary">
                    <td><strong>Weighted Avg</strong></td>
                    <td>${classReport['weighted avg'].precision.toFixed(4)}</td>
                    <td>${classReport['weighted avg'].recall.toFixed(4)}</td>
                    <td>${classReport['weighted avg'].f1score ? 
                          classReport['weighted avg'].f1score.toFixed(4) : 
                          classReport['weighted avg']['f1-score'].toFixed(4)}</td>
                    <td>${classReport['weighted avg'].support}</td>
                </tr>
            `;
        }
        
        if (classReport['accuracy']) {
            tableHTML += `
                <tr class="table-primary">
                    <td colspan="3"><strong>Accuracy</strong></td>
                    <td colspan="2">${classReport.accuracy.toFixed(4)}</td>
                </tr>
            `;
        }
        
        tableHTML += `
                </tbody>
            </table>
        `;
        
        container.innerHTML = tableHTML;
    } catch (error) {
        console.error(`Error creating classification report for ${containerId}:`, error);
        document.getElementById(containerId).innerHTML = 
            `<div class="alert alert-danger">Error creating classification report: ${error.message}</div>`;
    }
}
