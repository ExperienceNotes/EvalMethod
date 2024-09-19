import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix


class EvalMethod:
    def __init__(self, true_label, pred_label) -> None:
        self.True_label = true_label
        self.Pred_label = pred_label
        self.Precision, self.Recall, self.F1 = None, None, None
        self.FPR, self.TPR, self.ROC_AUC = None, None, None
        self.Conf_Matrix = None
    
    def Cal_BaseEval(self) -> None:
        self.Precision = precision_score(self.True_label, self.Pred_label)
        self.Recall = recall_score(self.True_label, self.Pred_label)
        self.F1 = f1_score(self.True_label, self.Pred_label)

    def Cal_Roc_curve(self):
        self.FPR, self.TPR, _ = roc_curve(self.True_label, self.Pred_label)
        self.ROC_AUC = auc(self.FPR, self.TPR)

    def Cal_Confusion_matrix(self):
        self.Conf_Matrix = confusion_matrix(self.True_label, self.Pred_label)

    def Plt_Result(self):
        data = [['Precision', f'{self.Precision:.2f}'], 
                ['Recall', f'{self.Recall:.2f}'],
                ['F1_score', f'{self.F1:.2f}']]
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        ax[0].plot(self.FPR, self.TPR, color='darkorange', lw=2, label=f'ROC curve (AUC = {self.ROC_AUC:.2f})')
        ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax[0].set_xlim([0.0, 1.0])
        ax[0].set_ylim([0.0, 1.05])
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_title('ROC Curve')
        ax[0].legend(loc="lower right")

        sns.heatmap(self.Conf_Matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[1])
        ax[1].set_title("Confusion Matrix")
        ax[1].set_xlabel('Predicted Label')
        ax[1].set_ylabel('True Label')

        plt.table(cellText=data, colLabels=["Metric", "Value"], cellLoc='center', loc='bottom', bbox=[0, -0.3, 1, 0.2])
        
        plt.tight_layout()
        plt.savefig('table.jpg')
    
if __name__ == "__main__":
    true_label = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]  # 0 表示正常, 1 表示異常
    pred_label = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]  # 模型預測的標籤
    EvalClass = EvalMethod(true_label, pred_label)
    EvalClass.Cal_BaseEval()
    EvalClass.Cal_Roc_curve()
    EvalClass.Cal_Confusion_matrix()
    EvalClass.Plt_Result()
