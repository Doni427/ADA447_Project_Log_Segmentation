import os
import re
import json
import pandas as pd

def parse_template_to_regex(template):
    # Escape regex special characters except for our target placeholder
    escaped = re.escape(template)
    # Replace escaped placeholder '\<\*\>' with a wildcard capture group
    pattern = escaped.replace(r'\<\*\>', r'([^\s]+)')
    return f"^{pattern}$"

def generate_word_labels(content, template):
    # Normalize whitespaces
    content_tokens = content.strip().split()
    template_tokens = template.strip().split()
    
    labels = []
    # If lengths match exactly, we can do a direct token-by-token comparison
    if len(content_tokens) == len(template_tokens):
        for c_tok, t_tok in zip(content_tokens, template_tokens):
            labels.append(1 if t_tok == "<*>" else 0)
    else:
        # Fallback: simple heuristic matching via regex structural mapping if token lengths differ
        # Due to variable token splits, if a word doesn't match the hardcoded template token position, mark as parameter
        t_idx = 0
        for c_tok in content_tokens:
            if t_idx < len(template_tokens) and template_tokens[t_idx] == "<*>":
                labels.append(1)
                # Advance template pointer if next token matches or we run out of tokens
                if t_idx + 1 < len(template_tokens) and c_tok == template_tokens[t_idx + 1]:
                    t_idx += 2
            elif t_idx < len(template_tokens) and c_tok == template_tokens[t_idx]:
                labels.append(0)
                t_idx += 1
            else:
                labels.append(1) # Default to variable if out of sync
                
    return content_tokens, labels[:len(content_tokens)]

def process_log_dataset(csv_path, output_json_path):
    if not os.path.exists(csv_path):
        # Generate dummy simulation data based on requirements mockups
        print(f"Creating mock log data at {csv_path}...")
        mock_data = {
            "LineId": [1, 2, 3],
            "Month": ["Jun", "Jun", "Jun"],
            "DateTime": ["14 15:16:01", "15 02:04:59", "16 10:01:28"],
            "Level": ["INFO", "ERROR", "ERROR"],
            "Component": ["mcelog", "sshd", "sshd"],
            "Content": ["mcelog start", "Connection refused from 192.168.1.5 port 22", "Connection refused from 10.43.0.10 port 9090"],
            "EventTemplate": ["mcelog start", "Connection refused from <*> port <*>", "Connection refused from <*> port <*>"]
        }
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(mock_data).to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    processed_records = []

    for _, row in df.iterrows():
        tokens, ner_tags = generate_word_labels(str(row['Content']), str(row['EventTemplate']))
        processed_records.append({
            "tokens": tokens,
            "ner_tags": ner_tags
        })

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(processed_records, f, indent=2)
    print(f"Successfully processed {len(processed_records)} log records -> {output_json_path}")

if __name__ == "__main__":
    process_log_dataset("data/Linux_2k.log_structured.csv", "data/processed_logs.json")