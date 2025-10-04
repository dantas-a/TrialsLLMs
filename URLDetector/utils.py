import evaluate
import numpy as np

def compute_metrics(pred_res):
    accuracy = evaluate.load("accuracy")
    auc_score = evaluate.load("roc_auc")

    predictions, labels = pred_res
    
    probabilities = np.exp(predictions)/np.exp(predictions).sum(-1,keepdims=True)
    
    positive_class_probs = probabilities[:,1]
    
    auc = np.round(auc_score.compute(predictions_scores=positive_class_probs, references=labels)['roc_auc'],3)
    
    predicted_classes = np.argmax(predictions, axis=1)
    acc = np.round(accuracy.compute(predictions=predicted_classes,references=labels)['accuracy'],3)
    
    return {"Accuracy" : acc, "AUC": auc}