import os
import re
import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

# Terminal Renk Kodları (ANSI Escape Sequences)
CLR_RAW = "\033[91m"       # Açık Kırmızı
CLR_TMP = "\033[92m"       # Açık Yeşil
CLR_PRM = "\033[96m"       # Açık Camgöbeği
CLR_RST = "\033[0m"        # Reset
CLR_BLD = "\033[1m"        # Bold
CLR_LINE = "\033[90m"      # Gri Çizgi

class LiveProductionParser:
    def __init__(self, model_dir="./models/fine_tuned_logbert/"):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Fine-tuned model bulunamadı: {model_dir}. Önce train.py çalıştırılmalı.")
            
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForTokenClassification.from_pretrained(model_dir)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Log başındaki standart ISO ve sistem zaman damgalarını yakalayan regex
        self.timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)\s+(.*)$")

    def parse(self, raw_log):
        raw_log = raw_log.strip()
        match = self.timestamp_pattern.match(raw_log)
        
        timestamp_prefix = ""
        log_body = raw_log
        
        if match:
            timestamp_prefix = match.group(1)
            log_body = match.group(2)
            
        words = log_body.split()
        if not words:
            return {"template": raw_log, "parameters": []}
            
        inputs = self.tokenizer(
            [words], 
            is_split_into_words=True, 
            return_tensors="pt", 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        if isinstance(predictions, int):  # Tek token durumu koruması
            predictions = [predictions]
            
        token_word_ids = self.tokenizer([words], is_split_into_words=True).word_ids(batch_index=0)
        
        # Subtoken kararlarını kelimelere indirgeme
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
                
        body_template = " ".join(template_tokens)
        final_template = f"{timestamp_prefix} {body_template}".strip() if timestamp_prefix else body_template
        
        return {
            "raw": raw_log,
            "template": final_template,
            "parameters": parameters
        }

def run_live_demo():
    # Gerçek dünya karmaşık altyapı log simülasyonları (K8s, RKE2, Prometheus, Vector)
    real_world_logs = [
        "2026-06-03 11:24:02 ERROR rke2-etcd connection refused to 10.43.0.15:2379",
        "2026-06-03 11:25:15 WARN kubelet-failed pod eviction triggered on node master-node-02",
        "2026-06-03 11:26:40 INFO prometheus-operator scraping failed for target metrics-server-uuid-8892f",
        "2026-06-03 11:27:01 ERROR vector-sink failed to flush 512 events to example.container.registry.com.tr",
        "2026-06-03 11:24:02 ERROR sshd connection refused from 10.43.0.15 port 2379",
        "2026-06-03 11:25:15 WARN CRON pam_unix session opened for user root by (uid=0)",
        "2026-06-03 11:26:40 INFO kernel pci 0000:00:1c.0: Signaling PME via PCIe slot",
        "2026-06-03 11:27:01 ERROR su pam_unix authentication failure logname= uid=0 euid=0 tty=tty1 ruser= rhost=192.168.1.100"
    ]
    
    print(f"\n{CLR_BLD}🚀 STARTING LIVE PRODUCTION PARSING DEMONSTRATION 🚀{CLR_RST}")
    print(f"{CLR_LINE}================================================================================{CLR_RST}")
    
    try:
        parser = LiveProductionParser()
        for log in real_world_logs:
            res = parser.parse(log)
            print(f"{CLR_BLD}[RAW LOG]:{CLR_RST}             {CLR_RAW}{res['raw']}{CLR_RST}")
            print(f"{CLR_BLD}[PARSED TEMPLATE]:{CLR_RST}    {CLR_TMP}{res['template']}{CLR_RST}")
            print(f"{CLR_BLD}[EXTRACTED PARAMS]:{CLR_RST}  {CLR_PRM}{res['parameters']}{CLR_RST}")
            print(f"{CLR_LINE}--------------------------------------------------------------------------------{CLR_RST}")
    except Exception as e:
        print(f"\033[91m❌ Gösterim hatası: {e}\033[0m")

if __name__ == "__main__":
    run_live_demo()