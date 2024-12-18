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
    Evaluates the performance of a classifier on validation data.

    Parameters:
    - classifier: Trained classifier with predict and predict_proba methods.
    - X_val: Feature matrix for validation.
    - y_val: True labels for validation.
    - save_plots: Boolean indicating whether to save the plots.
    - plot_path_prefix: Prefix path for saving plots.
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

    # Feature Importance (Logistic Regression Coefficients)
    if hasattr(classifier, 'coef_'):
        try:
            # Assuming X_val is a sparse matrix and vectorizer was used
            feature_names = []
            if hasattr(X, 'columns'):
                feature_names = X.columns
            elif hasattr(classifier, 'feature_names_in_'):
                feature_names = classifier.feature_names_in_
            else:
                print("Feature names not available for coefficient plotting.")
            
            if not feature_names and hasattr(classifier, 'steps'):
                # If classifier is part of a pipeline
                vectorizer = classifier.steps[0][1]
                feature_names = vectorizer.get_feature_names_out()
            
            if feature_names:
                coefficients = classifier.coef_[0]
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                })
                coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
                coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=False).head(20)

                plt.figure(figsize=(10, 8))
                sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
                plt.title('Top 20 Feature Coefficients')
                plt.xlabel('Coefficient Value')
                plt.ylabel('Feature')
                plt.tight_layout()
                if save_plots:
                    plt.savefig(f'{plot_path_prefix}_feature_importance.png')
                plt.show()
            else:
                print("Skipping feature importance plot due to unavailable feature names.")
        except Exception as e:
            print(f"An error occurred while plotting feature importance: {e}")

    else:
        print("The classifier does not have a 'coef_' attribute. Skipping feature importance plot.")

    print("\nEvaluation completed.")