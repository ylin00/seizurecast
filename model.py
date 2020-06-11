"""Credit: https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76"""
from sklearn.metrics import precision_score, recall_score, roc_auc_score, \
    roc_curve


def evaluate_model(predictions, probs, train_predictions, train_probs,
                   train_labels, test_labels):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = {}
    baseline['recall'] = recall_score(test_labels,
                                      [1 for _ in range(len(test_labels))])
    baseline['precis'] = precision_score(test_labels,
                                         [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5

    results = {}
    results['recall'] = recall_score(test_labels, predictions)
    results['precis'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precis'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)

    for metric in ['recall', 'precis', 'roc']:
        print(
            f'{metric.capitalize()} \t Base: {round(baseline[metric], 2)} \t Test: {round(results[metric], 2)}\t Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels,
                                      [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    return base_fpr, base_tpr, model_fpr, model_tpr
