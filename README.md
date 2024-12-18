# Amazon Reviews Sentiment Analysis

This project demonstrates how to build, tune, and evaluate a sentiment analysis model using Amazon product reviews. The goal is to classify each review as positive or negative, providing insights that can help businesses understand customer opinions and improve their products.

**Key Features:**

- **Large-Scale Data:** We use an Amazon Reviews dataset with millions of entries, ensuring the solution is scalable and robust.
- **End-to-End Pipeline:** From data cleaning and preprocessing to vectorization, modeling, and hyper-parameter tuning.
- **Practical NLP Techniques:** Leverages TF-IDF for feature extraction, a balanced choice between simplicity and effectiveness.
- **Hyper-Parameter Tuning:** Uses GridSearchCV to systematically find improved model configurations.
- **Comprehensive Documentation:** Aligns with a Medium article (linked below) that explains the reasoning behind each step.

---

## Project Structure

```
.
├─ data_cleaner.py                 # Contains the TextPreprocessor class and clean_dataset function
├─ model_evaluator.py              # Contains evaluation functions to assess model performance
├─ main.py (or main.ipynb)         # The main entry point with code cells and Markdown documentation
├─ preprocessed_data.csv           # Example preprocessed dataset saved for future use
├─ requirements.txt                # Python dependencies
├─ README.md                       # This README documentation
└─ figures/                        # Directory for saving plots (e.g., confusion matrix, PR, ROC curves)
```

*Note:*  
- `main.py` (or `main.ipynb`) contains the step-by-step workflow from raw data to final evaluation.
- `data_cleaner.py` defines custom preprocessing steps.
- `model_evaluator.py` provides utilities for computing metrics and visualizing performance.

---

## Dataset Overview

We use the Amazon Reviews dataset from Kaggle, containing millions of reviews labeled as positive or negative. This richness and diversity allow for robust, real-world model training and evaluation.

**Dataset Highlights:**

- **Size:** ~3.6 million reviews
- **Features:** `title`, `text`, and `polarity` (later mapped to `sentiment`)
- **Balance:** Approximately 50% positive and 50% negative sentiment

**Download Instructions:**

If you’re using `kagglehub`, you can download the dataset as follows:

```python
import kagglehub
path = kagglehub.dataset_download("kritanjalijain/amazon-reviews")
print("Path to dataset files:", path)
```

Adjust or use your own method if you prefer.

**Cleaning and Preprocessing:**
- Convert polarity (1 or 2) to binary sentiment (0 or 1).
- Remove duplicates, empty entries, and unnecessary columns.

**Figures and Examples:**
- Refer to `figures/` for illustrations (e.g., before and after preprocessing samples).

---

## Preprocessing and Text Normalization

The `TextPreprocessor` class (in `data_cleaner.py`) applies standard NLP preprocessing:
- Expanding contractions
- Lowercasing
- Removing punctuation
- Handling negations
- Tokenizing and removing stopwords
- POS-tagging and lemmatizing tokens
- Rejoining tokens into cleaned strings
- Filtering out empty entries

This ensures the classifier focuses on meaningful, standardized language cues.

---

## Splitting Data and Vectorization

After cleaning, we split into training, validation, and test sets:
- **Training/Validation:** For parameter tuning and avoiding overfitting.
- **Test:** For final unbiased evaluation.

**Feature Extraction with TF-IDF:**
- We treat `processed_title` and `processed_text` separately.
- Assign different weights to prioritize title signals if desired.

This approach yields a balanced, sentiment-focused representation.

---

## Baseline Modeling and Improvements

Start with a baseline Logistic Regression classifier for:
- Simplicity and speed
- Strong performance on large-scale text classification

**Baseline Accuracy:** ~92% on validation data.

Though good, we seek incremental improvements through parameter tuning.

---

## Hyper-Parameter Tuning with GridSearchCV

We define a parameter grid to adjust:
- TF-IDF vocabulary size (`max_features`, `min_df`)
- Weight ratios between title and text
- Logistic Regression parameters (`C`, `solver`, `max_iter`)

Using GridSearchCV:
- Systematically tests parameter combinations
- Employs cross-validation
- Optimizes accuracy

**Result:**  
Slight improvement in accuracy and better-calibrated model, proving the value of incremental refinement.

---

## Final Evaluation on the Test Set

Evaluate the tuned model on unseen test data:
- Achieves ~92% accuracy on test set
- Balanced precision/recall for both classes (~0.92 F1-scores)
- High MCC (~0.84) and excellent ROC AUC (~0.9750)

**Interpretation:**  
The model generalizes well and is robust, fair, and effective at distinguishing sentiment in real-world data.

**Figures:**
- **Confusion Matrix & Metrics:** Show balanced misclassifications and strong correlation with true labels.
- **Precision-Recall & ROC Curves:** Confirm high reliability and discriminative power.

---

## Conclusions and Future Work

**Key Takeaways:**
- Thorough preprocessing and data integrity checks set a solid foundation.
- TF-IDF and a simple Logistic Regression baseline yield strong initial performance.
- Hyper-parameter tuning with GridSearchCV refines the model, even if improvements are incremental.
- The final model generalizes effectively, handling new reviews with high accuracy and balanced metrics.

**Future Directions:**
- Experiment with word embeddings or contextual models like BERT if resources allow.
- Try different algorithms (SVM, Random Forest, XGBoost) to see if performance improves.
- Explore more extensive hyper-parameter grids or advanced tuning methods.

---

## Related Links

- **Medium Article:** [From Reviews to Sentiment: Building a Classifier with Amazon Data](https://medium.com/@mert_arcan/from-reviews-to-sentiment-building-a-classifier-with-amazon-data-abd811a91941)
- **Kaggle Dataset:** [Amazon Reviews Dataset](https://www.kaggle.com/kritanjalijain/amazon-reviews)

---

## Requirements and Setup

**Dependencies:**
- Python 3.13
- pandas, numpy
- scikit-learn
- nltk
- seaborn, matplotlib
- contractions, kagglehub

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Running the Code:**
Open `main.ipynb` to follow the code-and-commentary format.

**Model Saving and Loading:**
- Use `joblib` or `pickle` to save your trained pipeline.
- Once saved, load it to run predictions on new datasets without re-running the entire pipeline.

---

## Contact and Contributions

For issues, improvements, or contributions:
- Open an issue or pull request on this GitHub repository.
