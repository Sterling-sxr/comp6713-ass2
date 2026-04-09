import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    "Model 1\nTF-IDF+LR",
    "Model 2\nRoBERTa public",
    "Model 3\nBERT public",
    "Model 4\nBERT+WordNet",
    "Model 5\nRoBERTa+Ling"
]

# Test metrics
accuracy = [0.8962, 0.8333, 0.6664, 0.9482, 0.9757]
precision = [0.8150, 0.7776, 0.4260, 0.8450, 0.9233]
recall = [0.7860, 0.5175, 0.7533, 0.9845, 0.9905]
f1_score = [0.8002, 0.6215, 0.5442, 0.9094, 0.9695]

x = np.arange(len(models))
width = 0.35

# Figure 1: Accuracy and F1-score
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width / 2, accuracy, width, label="Accuracy")
bars2 = plt.bar(x + width / 2, f1_score, width, label="F1-score")

plt.xticks(x, models)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Comparison of Overall Performance Across Five Models")
plt.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.4f}",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison_accuracy_f1.png", dpi=300, bbox_inches="tight")
plt.show()

# Figure 2: Precision and Recall
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width / 2, precision, width, label="Precision")
bars2 = plt.bar(x + width / 2, recall, width, label="Recall")

plt.xticks(x, models)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Comparison of Precision and Recall Across Five Models")
plt.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.4f}",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison_precision_recall.png", dpi=300, bbox_inches="tight")
plt.show()