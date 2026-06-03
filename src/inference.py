import re
import torch
import json
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

class LogParserPipeline:
    def __init__(self, model_dir="./models/fine_tuned_logbert/"):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForTokenClassification.from_pretrained(model_dir)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Standart YYYY-MM-DD HH:MM:SS formatındaki zaman damgalarını yakalayan regex
        self.timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(.*)$")

    def parse_log(self, raw_log_string):
        raw_log_string = raw_log_string.strip()
        match = self.timestamp_pattern.match(raw_log_string)
        
        timestamp_prefix = ""
        log_body = raw_log_string
        
        # Eğer log satırı zaman damgası ile başlıyorsa ayır
        if match:
            timestamp_prefix = match.group(1)  # Örn: "2026-06-03 10:01:28"
            log_body = match.group(2)          # Örn: "ERROR connection refused to 10.43.0.10:9090"
            
        words = log_body.split()
        
        inputs = self.tokenizer(
            [words], 
            is_split_into_words=True, 
            return_tensors="pt", 
            truncation=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()
        
        token_word_ids = self.tokenizer([words], is_split_into_words=True).word_ids(batch_index=0)
        
        # Subtoken tahminlerini kelime seviyesine indirge
        word_level_predictions = {}
        previous_word_idx = None
        for idx, word_idx in enumerate(token_word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                word_level_predictions[word_idx] = predictions[idx]
            previous_word_idx = word_idx

        template_tokens = []
        parameters = []
        
        for idx, word in enumerate(words):
            pred_label = word_level_predictions.get(idx, 0)
            if pred_label == 1:
                template_tokens.append("<*>")
                parameters.append(word)
            else:
                template_tokens.append(word)
                
        # İçerik şablonunu oluştur
        body_template = " ".join(template_tokens)
        
        # Eğer zaman damgası ayırdıysak, başına koruyarak geri ekle
        final_template = f"{timestamp_prefix} {body_template}".strip() if timestamp_prefix else body_template
        
        return {
            "template": final_template,
            "parameters": parameters
        }

if __name__ == "__main__":
    try:
        parser = LogParserPipeline()
        sample_log = "2026-06-03 10:01:28 ERROR connection refused to 10.43.0.10:9090"
        result = parser.parse_log(sample_log)
        print("\n🔮 Timestamp Protected Inference Output:")
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Hata oluştu: {e}")