import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def find_latest_trainer_state(results_dir="./results"):
    """En güncel checkpoint içindeki trainer_state.json dosyasını bulur."""
    state_files = glob.glob(os.path.join(results_dir, "checkpoint-*", "trainer_state.json"))
    if not state_files:
        # Eğer checkpoint klasörleri silindiyse doğrudan results altına da bakabiliriz
        state_files = glob.glob(os.path.join(results_dir, "trainer_state.json"))
        
    if not state_files:
        raise FileNotFoundError(
            f"'{results_dir}' dizini altında 'trainer_state.json' bulunamadı. "
            "Lütfen train.py scriptinin başarıyla çalıştığından ve sonuçları ürettiğinden emin olun."
        )
    # En son modifiye edilen dosyayı seç
    latest_file = max(state_files, key=os.path.getmtime)
    print(f"📖 Okunan log dosyası: {latest_file}")
    return latest_file

def parse_training_logs(state_file_path):
    """Log dosyasını okuyarak loss ve eval metriklerini ayrıştırır."""
    with open(state_file_path, "r") as f:
        data = json.load(f)
        
    log_history = data.get("log_history", [])
    
    steps = []
    losses = []
    epoch_metrics = {}
    
    for log in log_history:
        # Training Loss takibi
        if "loss" in log and "step" in log:
            steps.append(log["step"])
            losses.append(float(log["loss"]))
            
        # Evaluation Metrikleri takibi (Tam epoch değerlerine göre filtreleme)
        if "eval_loss" in log and "epoch" in log:
            epoch_num = int(round(log["epoch"]))
            epoch_metrics[epoch_num] = {
                "Precision": log.get("eval_precision", 0.0),
                "Recall": log.get("eval_recall", 0.0),
                "F1-Score": log.get("eval_f1", 0.0)
            }
            
    return steps, losses, epoch_metrics

def plot_and_save_results():
    # Klasör kontrolü ve kurulumu
    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="darkgrid") # Görsel kalite için seaborn teması
    
    try:
        log_file = find_latest_trainer_state()
        steps, losses, epoch_metrics = parse_training_logs(log_file)
    except Exception as e:
        print(e)
        return

    # -------------------------------------------------------------
    # CHART 1: Loss Curve (Eğitim Kayıp Eğrisi)
    # -------------------------------------------------------------
    if steps and losses:
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, label="Training Loss", color="#1f77b4", linewidth=2)
        plt.title("DistilBERT Log Parsing - Training Loss Curve", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        loss_plot_path = "plots/loss_curve.png"
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        print(f"📉 Loss eğrisi kaydedildi: {loss_plot_path}")
    else:
        print("⚠️ Log geçmişinde çizilecek uygun loss verisi bulunamadı.")

    # -------------------------------------------------------------
    # CHART 2: Evaluation Metrics for Epoch 1
    # -------------------------------------------------------------
    target_epoch = 1
    if target_epoch in epoch_metrics:
        metrics_data = epoch_metrics[target_epoch]
        metric_names = list(metrics_data.keys())
        metric_values = list(metrics_data.values())
        
        plt.figure(figsize=(8, 5))
        colors = ["#2ca02c", "#ff7f0e", "#9467bd"] # Precision, Recall, F1 için belirgin renkler
        
        bars = plt.bar(metric_names, metric_values, color=colors, width=0.5, edgecolor='black', linewidth=0.7)
        
        # Barların üzerine tam sayı değerlerini/yüzdelerini yazma
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.0, 
                height - 0.08 if height > 0.1 else height + 0.01, 
                f"{height:.4f}", 
                ha='center', 
                va='bottom', 
                fontsize=11, 
                fontweight='bold',
                color='white' if height > 0.5 else 'black'
            )
            
        plt.title(f"Model Performance Evaluation - Epoch {target_epoch}", fontsize=14, fontweight='bold', pad=15)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1.1) # Skor aralığını sabitleme
        plt.tight_layout()
        
        metrics_plot_path = f"plots/epoch_{target_epoch}_metrics.png"
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        print(f"📊 Epoch {target_epoch} performans grafiği kaydedildi: {metrics_plot_path}")
    else:
        print(f"⚠️ Loglarda Epoch {target_epoch} değerine ait doğrulama metriği (eval) bulunamadı.")

if __name__ == "__main__":
    plot_and_save_results()