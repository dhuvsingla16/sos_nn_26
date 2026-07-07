import random

def train_val_test_split(dataset, train_ratio=0.8, val_ratio=0.1):
    # uniform distribution
    random.shuffle(dataset)
    
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = dataset[:train_end]
    val_set = dataset[train_end:val_end]
    test_set = dataset[val_end:]
    
    return train_set, val_set, test_set

class ClassificationEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.tp, self.tn, self.fp, self.fn = self._build_confusion_matrix()

    def _build_confusion_matrix(self):
        tp = tn = fp = fn = 0
        for true_val, pred_val in zip(self.y_true, self.y_pred):
            if true_val == 1 and pred_val == 1:
                tp += 1
            elif true_val == 0 and pred_val == 0:
                tn += 1
            elif true_val == 0 and pred_val == 1:
                fp += 1
            elif true_val == 1 and pred_val == 0:
                fn += 1
        return tp, tn, fp, fn

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def precision(self):
        # Out of all predicted positives, how many were actually positive?
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def recall(self):
        # Out of all actual positives, how many did we find?
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def f1_score(self):
        # Harmonic mean of Precision and Recall
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

if __name__ == "__main__":
    print("--- Executing Week 3 Data Splitting & Evaluation ---")
    
    # 1. Simulate a Dummy Dataset of 100 items 
    dummy_data = [([random.uniform(0, 1), random.uniform(0, 1)], random.choice([0, 1])) for _ in range(100)]
    
    # 2. Split the Data (80% Train, 10% Val, 10% Test)
    train, val, test = train_val_test_split(dummy_data, train_ratio=0.8, val_ratio=0.1)
    print(f"Data Split -> Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    # 3. Simulate "Predictions" on the Test Set 
    y_true_test = [item[1] for item in test]
    
    # Simulating a slightly decent model by occasionally copying the true label
    y_pred_test = [label if random.random() > 0.3 else 1 - label for label in y_true_test]
    
    # 4. Evaluate the simulated model
    evaluator = ClassificationEvaluator(y_true_test, y_pred_test)
    
    print("\n--- Confusion Matrix ---")
    print(f"True Positives (TP):  {evaluator.tp}")
    print(f"True Negatives (TN):  {evaluator.tn}")
    print(f"False Positives (FP): {evaluator.fp}")
    print(f"False Negatives (FN): {evaluator.fn}")
    
    print("\n--- Core Metrics ---")
    print(f"Accuracy:  {evaluator.accuracy():.4f}")
    print(f"Precision: {evaluator.precision():.4f}")
    print(f"Recall:    {evaluator.recall():.4f}")
    print(f"F1-Score:  {evaluator.f1_score():.4f}")
