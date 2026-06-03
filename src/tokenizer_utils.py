from transformers import DistilBertTokenizerFast

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        padding=False
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Token is a special token (e.g., [CLS], [SEP]) -> Ignore in loss
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subtoken of the current word -> Assign the actual ground truth tag
                label_ids.append(label[word_idx])
            else:
                # Subsequent subtokens within the same word -> Use -100 to ignore, or set to 2 (I-PAR)
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == "__main__":
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    # Quick test check
    sample = {"tokens": [["Connection", "refused", "from", "192.168.1.5"]], "ner_tags": [[0, 0, 0, 1]]}
    aligned = tokenize_and_align_labels(sample, tokenizer)
    print("Tokens:", tokenizer.convert_ids_to_tokens(aligned["input_ids"][0]))
    print("Labels:", aligned["labels"][0])