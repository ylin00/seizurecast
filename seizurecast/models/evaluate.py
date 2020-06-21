"""Credit: https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76"""
from sklearn.metrics import precision_score, recall_score, roc_auc_score, \
    roc_curve


def evaluate_model(test_pred, test_prob, train_pred, train_prob,
                   train_labels, test_labels, verbose=True):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve.

    Args:
        test_pred: test y_predicted
        test_prob: test y_scores
        train_pred: train y_predicted
        train_prob: train y_scores
        train_labels: train labels
        test_labels: test labels
        verbose: verbose mode

    Returns: tuple(list, list, list, list):
        - base_fpr
        - base_tpr
        - model_fpr
        - model_tpr
    """

    baseline = {}
    baseline['recall'] = recall_score(test_labels,
                                      [1 for _ in range(len(test_labels))])
    baseline['precis'] = precision_score(test_labels,
                                         [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5

    results = {}
    results['recall'] = recall_score(test_labels, test_pred)
    results['precis'] = precision_score(test_labels, test_pred)
    results['roc'] = roc_auc_score(test_labels, test_prob)

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_pred)
    train_results['precis'] = precision_score(train_labels, train_pred)
    train_results['roc'] = roc_auc_score(train_labels, train_prob)

    if verbose:
        for metric in ['recall', 'precis', 'roc']:
            print(f'{metric.capitalize()}\t '
                  f'Base: {round(baseline[metric], 2)}\t '
                  f'Test: {round(results[metric], 2)}\t '
                  f'Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels,
                                      [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, test_prob)

    return base_fpr, base_tpr, model_fpr, model_tpr
