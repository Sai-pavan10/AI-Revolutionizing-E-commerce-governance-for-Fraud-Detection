/**
 * Dashboard functionality for the fraud detection system
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
    
    // Initialize visualizations if on dashboard page
    if (document.getElementById('fraud-distribution-chart')) {
        initDashboardCharts();
    }
    
    // Initialize model results page if on compare page
    if (document.querySelector('.model-results-container')) {
        initModelResultsPage();
    }
    
    // Initialize model selection form if on models page
    if (document.getElementById('model-selection-form')) {
        initModelSelectionForm();
    }
});

// Initialize dashboard charts
function initDashboardCharts() {
    // Load charts if data is available
    const chartElements = document.querySelectorAll('[data-chart-json]');
    chartElements.forEach(el => {
        const chartData = el.getAttribute('data-chart-json');
        if (chartData) {
            createPlotFromJson(el.id, chartData);
        }
    });
}

// Initialize model results page
function initModelResultsPage() {
    // Set up tabs for model results
    const modelTabs = document.querySelectorAll('.model-tab');
    modelTabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Hide all model results
            document.querySelectorAll('.model-result').forEach(result => {
                result.classList.remove('active');
            });
            
            // Remove active class from all tabs
            modelTabs.forEach(t => {
                t.classList.remove('active');
            });
            
            // Show selected model result and activate tab
            const modelId = this.getAttribute('data-model-id');
            document.getElementById(`model-result-${modelId}`).classList.add('active');
            this.classList.add('active');
            
            // Load model visualizations
            loadModelVisualizations(modelId);
        });
    });
    
    // Activate first tab by default
    if (modelTabs.length > 0) {
        modelTabs[0].click();
    }
}

// Load visualizations for a model
function loadModelVisualizations(modelId) {
    // Load confusion matrix
    fetch(`/api/confusion_matrix/${modelId}`)
        .then(response => response.json())
        .then(data => {
            createConfusionMatrix(`confusion-matrix-${modelId}`, data);
        })
        .catch(error => {
            console.error(`Error loading confusion matrix for ${modelId}:`, error);
            document.getElementById(`confusion-matrix-${modelId}`).innerHTML = 
                '<div class="alert alert-danger">Error loading confusion matrix.</div>';
        });
    
    // Load ROC curve
    fetch(`/api/roc_curve/${modelId}`)
        .then(response => response.json())
        .then(data => {
            createRocCurve(`roc-curve-${modelId}`, data);
        })
        .catch(error => {
            console.error(`Error loading ROC curve for ${modelId}:`, error);
            document.getElementById(`roc-curve-${modelId}`).innerHTML = 
                '<div class="alert alert-danger">Error loading ROC curve.</div>';
        });
    
    // Load feature importance if supported
    if (document.getElementById(`feature-importance-${modelId}`)) {
        fetch(`/api/feature_importance/${modelId}`)
            .then(response => response.json())
            .then(data => {
                createFeatureImportance(`feature-importance-${modelId}`, data);
            })
            .catch(error => {
                console.error(`Error loading feature importance for ${modelId}:`, error);
                document.getElementById(`feature-importance-${modelId}`).innerHTML = 
                    '<div class="alert alert-info">Feature importance not available for this model type.</div>';
            });
    }
    
    // Load classification report
    if (document.getElementById(`classification-report-${modelId}`)) {
        fetch(`/api/classification_report/${modelId}`)
            .then(response => response.json())
            .then(data => {
                createClassificationReport(`classification-report-${modelId}`, data);
            })
            .catch(error => {
                console.error(`Error loading classification report for ${modelId}:`, error);
                document.getElementById(`classification-report-${modelId}`).innerHTML = 
                    '<div class="alert alert-danger">Error loading classification report.</div>';
            });
    }
}

// Initialize model selection form
function initModelSelectionForm() {
    const form = document.getElementById('model-selection-form');
    
    if (form) {
        // Validate form submission
        form.addEventListener('submit', function(e) {
            const checkboxes = form.querySelectorAll('input[type="checkbox"]:checked');
            
            if (checkboxes.length === 0) {
                e.preventDefault();
                alert('Please select at least one model to train.');
            } else {
                const submitBtn = form.querySelector('button[type="submit"]');
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...';
            }
        });
    }
}
