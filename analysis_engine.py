from transformers import AutoTokenizer, AutoModel
import torch
from functools import lru_cache 

@lru_cache(maxsize=4) # Cache up to 4 different models
def get_model_and_tokenizer(model_name):
    """Loads a model and tokenizer from Hugging Face, caching the result."""
    print(f"Loading model: {model_name}") 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

#  Main analysis function 
def analyze_sentence(model_name, sentence):
    """
    Analyzes a sentence using a specified transformer model and extracts internal states.
    """
    if not sentence or not model_name:
        return None 

    model, tokenizer = get_model_and_tokenizer(model_name)
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attention_weights = outputs.attentions
    hidden_states = outputs.hidden_states

    analysis_data = {
        "tokens": tokens,
        "attention": attention_weights,
        "hidden_states": hidden_states
    }
    return analysis_data

# Testing block 
if __name__ == "__main__":
    test_model = "dbmdz/bert-base-turkish-cased"
    test_sentence = "Yapay zeka geleceğin teknolojisidir."

    print(f"--- Running analysis for: '{test_sentence}' ---")
    data = analyze_sentence(test_model, test_sentence)

    if data:
        print(f"\nTokens: {data['tokens']}")
        print(f"Number of layers (for attention): {len(data['attention'])}")
        print(f"Shape of attention tensor in Layer 0: {data['attention'][0].shape}")