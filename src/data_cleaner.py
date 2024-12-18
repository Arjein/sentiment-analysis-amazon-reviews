from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from nltk.stem import WordNetLemmatizer
from contractions import fix  # pip install contractions
from nltk.corpus import wordnet

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A text preprocessing transformer that:
    - Expands contractions
    - Converts text to lowercase
    - Removes punctuation
    - Handles negations
    - Tokenizes text
    - Removes stopwords
    - POS-tags and lemmatizes tokens
    - Removes empty rows and renames processed columns

    Intended for use on a pandas DataFrame with 'title' and 'text' columns.
    
    Parameters:
    -----------
    columns : list
        List of column names to preprocess.
    """
    def __init__(self, columns=None):
        self.columns = columns
        self.stop_words = set(stopwords.words('english'))  # **Step 6: Initialize Stopwords**
        self.lemmatizer = WordNetLemmatizer()  # **Step 7: Initialize Lemmatizer**
        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.dropna(subset=self.columns, inplace=True)

        for col in self.columns:
            X[col] = X[col].apply(self.expand_contractions)  # Step 1: Expand Contractions**
            X[col] = X[col].apply(self.lowercase_text)       # Step 2: Convert to Lowercase**
            X[col] = X[col].apply(self.remove_punctuation)   # Step 3: Remove Punctuation**
            X[col] = X[col].apply(self.handle_negations)     # Step 4: Handle Negations**
            X[col] = X[col].apply(self.tokenize_text)        # Step 5: Tokenize Text**
            X[col] = X[col].apply(self.remove_stopwords)     # Step 6: Remove Stopwords**
            
            X[col] = X[col].apply(self.pos_tag_tokens)       # Step 7: POS Tagging**
            X[col] = X[col].apply(self.lemmatize_with_pos)   # Step 7: POS Tag and Lemmatize**
            X[col] = X[col].apply(self.combine_tokens)       # Step 8: Recombine Tokens**
            
        # **Step 9: Final Validation - Remove Rows with Empty Processed Fields**
        for col in self.columns:
            X = X[X[col].str.strip() != ""]
        
        # Rename the processed columns for clarity
        X.rename(columns={col: f"processed_{col}" for col in self.columns}, inplace=True)
        
        return X

    def lowercase_text(self, text):
        return text.lower()
    
    def expand_contractions(self, text):
        return fix(text)

    def handle_negations(self, text):
        # After contractions are expanded, we mostly deal with words like "not", "no", "never", etc.
        negation_words = "not|never|no|none|nothing|nowhere|nobody|neither|nor"
        pattern = rf"\b({negation_words})\b\s+(\w+)"
        return re.sub(pattern, r"\1_\2", text)

    def remove_punctuation(self, text):
        return re.sub(r'[^a-z\s]', '', text)

    def tokenize_text(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def pos_tag_tokens(self, tokens):
        return nltk.pos_tag(tokens)  # Returns list of tuples: [(token, pos_tag), ...]

    def get_wordnet_pos(self, treebank_tag):
        # Convert Penn Treebank tag to a WordNet POS
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_with_pos(self, pos_tagged_tokens):
        # pos_tagged_tokens is a list of (word, pos) tuples
        lemmatized = []
        for word, tag in pos_tagged_tokens:
            wn_tag = self.get_wordnet_pos(tag)
            lemmatized.append(self.lemmatizer.lemmatize(word, wn_tag))
        return lemmatized

    def combine_tokens(self, tokens):
        return ' '.join(tokens)



def clean_dataset(data):
    """
    Cleans the raw dataset by:
    1. Mapping polarity values to sentiment labels (0 for negative, 1 for positive).
    2. Removing rows with missing or empty title/text fields.
    3. Dropping duplicate reviews.
    4. Removing unnecessary columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw DataFrame with 'polarity', 'title', and 'text' columns.
    
    Returns:
    --------
    pd.DataFrame
        A cleaned and focused DataFrame with 'title', 'text', and 'sentiment' columns ready for preprocessing.
    """

    # Map polarity values to sentiment labels
    data['sentiment'] = data['polarity'].map({1: 0, 2: 1})
    
    # Drop rows where 'title' or 'text' is missing or empty
    data = data.dropna(subset=['title', 'text'])  # Drop rows where title or text is NaN
    data = data[data['title'].str.strip() != ""]  # Remove rows with empty title
    data = data[data['text'].str.strip() != ""]   # Remove rows with empty text

    data = data.drop_duplicates(subset=['title', 'text'])

    # Drop unnecessary columns
    data = data.drop(columns=["polarity"])
    
    return data

def saveOutputDataset(dataset,output_path='preprocessed_data.csv'):
    dataset.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")