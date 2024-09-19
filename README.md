### 1. **ROC (Receiver Operating Characteristic) 曲線**
   - **ROC 曲線** 用於可視化分類模型的性能，通過繪製 **True Positive Rate (TPR)** 和 **False Positive Rate (FPR)** 隨著閾值變化的圖表來顯示模型在各個閾值下的表現。
   - **TPR (True Positive Rate) / Recall**:
    $TPR = \frac{TP}{TP + FN}$
   - **FPR (False Positive Rate)**:
    $FPR = \frac{FP}{FP + TN}$
   - 在程式中，我們使用 `roc_curve()` 函數計算了不同閾值下的 **TPR** 和 **FPR**，並繪製了 ROC 曲線。這能幫助我們判斷模型在各個閾值下的區分能力。


### 2. **AUC (Area Under Curve)**
   - **AUC** 是 ROC 曲線下的面積，它代表了模型區分正負樣本的整體能力。AUC 的值在 0 到 1 之間，值越接近 1，表示模型性能越好。
   - 在程式中，`auc(FPR, TPR)` 計算了 AUC 值，結果顯示在 ROC 曲線的標籤中。

### 3. **混淆矩陣中的 TP, TN, FP, FN**
   - **TP (True Positive)**：實際為異常（1），且被正確檢測為異常（1）的數量。在例子中，這是 **5**。
   - **TN (True Negative)**：實際為正常（0），且被正確檢測為正常（0）的數量。在例子中，這是 **3**。
   - **FP (False Positive)**：實際為正常（0），但被錯誤檢測為異常（1）的數量。在例子中，這是 **1**。
   - **FN (False Negative)**：實際為異常（1），但被錯誤檢測為正常（0）的數量。在例子中，這是 **1**。

   混淆矩陣示例：
   ```lua
   [[TN  FP]
    [FN  TP]]
   [[3   1]
    [1   5]]
   ```

### 4. **Precision（精確率）**
   - **定義**：精確率衡量被檢測為異常的數據中，有多少是實際異常的。公式為：
     $text{Precision} = \frac{TP}{TP + FP}$
   - 在程式中，`precision_score()` 函數計算了精確率，結果為：
     $\frac{5}{5 + 1} = 0.83$

### 5. **Recall（召回率）**
   - **定義**：召回率衡量所有實際異常的數據中，有多少被正確檢測為異常。公式為：
     $\text{Recall} = \frac{TP}{TP + FN}$
   - 在程式中，`recall_score()` 函數計算了召回率，結果為：
     $\frac{5}{5 + 1} = 0.83$

### 6. **F1-score**
   - **定義**：F1-score 是 Precision 和 Recall 的調和平均數，平衡兩者的影響。公式為：
     $F1\text{-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
   - 在程式中，`f1_score()` 函數計算了 F1-score，結果為：
     $2 \times \frac{0.83 \times 0.83}{0.83 + 0.83} = 0.83$

### 程式的具體數據分析：
1. **混淆矩陣**：
   ```
   [[3  1]
    [1  5]]
   ```
   - TP = 5
   - TN = 3
   - FP = 1
   - FN = 1

2. **Precision** = 0.83
3. **Recall** = 0.83
4. **F1-score** = 0.83
5. **AUC** 通過計算 ROC 曲線的面積來得到，範例中的結果是接近 **0.83**。
