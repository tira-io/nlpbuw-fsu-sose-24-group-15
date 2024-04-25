from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
# Assuming tira.pd.inputs() and tira.pd.truths() return pandas DataFrames
# Assuming tira.rest_api_client.Client() is set up correctly

if __name__ == "__main__":
    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    # Feature Extraction
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(text_train["text"])
    X_validation = vectorizer.transform(text_validation["text"])
    
    # Model Training
    model = LogisticRegression()
    model.fit(X_train, targets_train['generated'])

    # Model Evaluation
    predictions = model.predict(X_validation)
    accuracy = accuracy_score(targets_validation['generated'], predictions)
    print("Validation Accuracy:", accuracy)

    # Classifying the Data
    prediction = model.predict(X_validation)

    # Converting the prediction to the required format
    prediction_df = text_validation[["id"]].copy()
    prediction_df["generated"] = prediction

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction_df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
