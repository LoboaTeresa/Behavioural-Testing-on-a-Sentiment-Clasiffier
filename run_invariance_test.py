import pickle
import json
import random

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score

from src.pre_process_data import pre_process_data

def apply_typo(text:str, typo_rate:float = 0.1) -> str:
    """Apply a typo to the input text.

    Args:
        input_text (str): The input text to apply the typo.

    Returns:
        str: The text with a typo applied.
    """
    # List of characters that can be used for substitutions
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    # Introduce typos with the specified rate
    typo_text = ''
    for char in text:
        if char.isalpha() and random.random() < typo_rate:
            # Introduce a typo by randomly selecting a character from the alphabet
            typo_text += random.choice(alphabet.replace(char.lower(), ''))
        else:
            typo_text += char
    return typo_text

def load_model(model_path: str):
    """Load a machine learning model from a file.

    Args:
        model_path (str): The file path to the model.

    Returns:
        object: The loaded machine learning model.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def get_model(model_name: str):
    """Get a machine learning model.

    Args:
        model_name (str): The name of the model.Options: RandomForestClassifier, LinearSVC, MultinomialNB, LogisticRegression, KMeans.

    Returns:
        object: The machine learning model.
    """
    main_directory = str(Path().resolve())
    train_data = pd.read_csv(main_directory + '/data/train.csv')
    __, r_train_X, r_train_y = pre_process_data(train_data , data_type = 'train')

    if model_name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(max_depth = 15, n_estimators = 3000, random_state = 2)
    elif model_name == "LinearSVC":
        from sklearn.svm import LinearSVC
        model = LinearSVC()
    elif model_name == "MultinomialNB":
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
    elif model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state = 2, max_iter = 500)
    elif model_name == "KMeans":
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters = 4, init = 'k-means++')
    else:
        raise ValueError("Invalid model name.")
    return model.fit(r_train_X, r_train_y)


def write_failure_modes(diff_df: pd.DataFrame, file_path: str):
    """Write out interesting cases where predictions differ to a file.

    Args:
        diff_df (pd.DataFrame): DataFrame containing interesting cases where predictions differ.
        file_path (str): The file path to write the data.
    """
    with open(file_path, "w") as outfile:
        outfile.write(diff_df.head(4).to_markdown(index=False))

def write_test_results(accuracy_orig: float, accuracy_typo: float, diff_rate: float, file_path: str):
    """Write test results to a JSON file.

    Args:
        accuracy_orig (float): Accuracy of the model on the original dataset.
        accuracy_typo (float): Accuracy of the model on the dataset with typos.
        diff_rate (float): Percentage of tweets that were classified differently.
        file_path (str): The file path to write the JSON data.
    """
    with open(file_path, 'w') as outfile:
        json.dump({
            "accuracy_original_dataset": accuracy_orig,
            "accuracy_typo_dataset": accuracy_typo,
            "percentage_tweets_classified_differently": "{:.2f}%".format(diff_rate * 100)
        }, outfile)

# Paths
test_path = './data/test.csv'
model_name = 'RandomForestClassifier' # Options: LinearSVC, LogisticRegression

# Read test data
test_df = pd.read_csv(test_path)

# Load model
model = get_model(model_name = "RandomForestClassifier")

# Pre-process the data
clean_df, x, gt_y = pre_process_data(test_df, data_type='test')

# Apply typos
typo_df = clean_df.copy()
typo_df['message'] = typo_df['message'].apply(apply_typo)
typo_clean_df, typo_x, __ = pre_process_data(typo_df, data_type='test')

# Run inference
y_pred = model.predict(x)
y_pred_typo = model.predict(typo_x)

# Create DataFrame for comparison
comparison_df = pd.concat([clean_df['message'], typo_clean_df['message'], clean_df['target']], axis=1)
comparison_df.columns = ['original_tweet', 'perturbed_tweet', 'gt']
comparison_df['pred'] = y_pred
comparison_df['pred_typo'] = y_pred_typo

# Select rows where predictions differ
diff_df = comparison_df[comparison_df['pred'] != comparison_df['pred_typo']]

# Write out interesting cases
write_failure_modes(diff_df, "failure_modes.txt")

# Calculate the percentage of tweets that are classified differently
diff_rate = len(diff_df) / len(comparison_df)

# Calculate the accuracy
gt = clean_df["target"]
accuracy_orig = accuracy_score(gt, y_pred)
accuracy_typo = accuracy_score(gt, y_pred_typo)

# Write results to file
write_test_results(accuracy_orig, accuracy_typo, diff_rate, "test_score.json")
