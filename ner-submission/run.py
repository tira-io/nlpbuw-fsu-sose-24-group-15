from pathlib import Path
import json
import spacy
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def predict_labels(sentences, nlp):
    predictions = []
    for sentence in sentences:
        doc = nlp(sentence['sentence'])
        tokens = [token.text for token in doc]
        labels = ['O'] * len(tokens)
        
        for ent in doc.ents:
            ent_tokens = [token.text for token in nlp(ent.text)]
            start_idx = None
            for i in range(len(tokens) - len(ent_tokens) + 1):
                if tokens[i:i+len(ent_tokens)] == ent_tokens:
                    start_idx = i
                    break
            if start_idx is not None:
                labels[start_idx] = f"B-{ent.label_}"
                for i in range(1, len(ent_tokens)):
                    labels[start_idx + i] = f"I-{ent.label_}"
        
        predictions.append({"id": sentence['id'], "tags": labels})
    return predictions

if __name__ == "__main__":
    tira = Client()

    # Loading validation data (automatically replaced by test data when run on TIRA)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    sentences = text_validation.to_dict(orient="records")

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Predicting labels for each sentence
    predictions = predict_labels(sentences, nlp)

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w') as outfile:
        for prediction in predictions:
            json.dump(prediction, outfile)
            outfile.write('\n')
