# Import necessary libraries for evaluation
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, precision_recall_curve, f1_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def evaluate_model(classifier, X, y, save_plots=False, plot_path_prefix=''):
    """
    Evaluates a trained classifier on the given dataset (X, y).
    
    Prints out key metrics:
    - Accuracy
    - Classification report (precision, recall, f1-score)
    - Weighted F1-score
    - Matthews Correlation Coefficient (MCC)
    - ROC AUC Score
    - Confusion Matrix

    Parameters:
    -----------
    classifier : sklearn classifier or pipeline
        The trained model to evaluate.
    X : pd.DataFrame or array-like
        The input features.
    y : pd.Series or array-like
        The true labels.

    Returns:
    --------
    None
        Prints the evaluation results directly.
    """
    
    # Make predictions on the validation set
    predictions = classifier.predict(X)
    y_probs = classifier.predict_proba(X)[:, 1]

    # Evaluate accuracy
    accuracy = accuracy_score(y, predictions)
    print(f'Accuracy: {accuracy:.4f}')

    # Detailed classification report
    print('\nClassification Report:')
    print(classification_report(y, predictions))

    # Calculate F1-Score
    f1 = f1_score(y, predictions, average='weighted')
    print(f'Weighted F1-Score: {f1:.4f}')

    # Calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y, predictions)
    print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')

    # Calculate ROC AUC Score
    roc_auc = roc_auc_score(y, y_probs)
    print(f'ROC AUC Score: {roc_auc:.4f}')

    # Plot Confusion Matrix with custom colors
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#faf3f3', '#363636'])
    cm = confusion_matrix(y, predictions)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['Actual Negative', 'Actual Positive']
    )

    # Set labels and title with #363636 color
    plt.tight_layout()

    fig = ax.figure

    if save_plots:
        # Ensure the saved figure retains the background color
        plt.savefig(f'{plot_path_prefix}_confusion_matrix.png', facecolor='#faf3f3')

    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)  # Diagonal line representing random chance
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{plot_path_prefix}_roc_curve.png')
    plt.show()

    # Plot Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve', color='green', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{plot_path_prefix}_precision_recall_curve.png')
    plt.show()


    print("\nEvaluation completed.")