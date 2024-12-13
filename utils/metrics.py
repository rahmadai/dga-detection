from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }