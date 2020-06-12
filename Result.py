from plots import plot_roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt


class Result:
    model_name = ''
    model = None
    base_tpr=None
    base_fpr=None
    model_tpr=None
    model_fpr=None
    roc=0.5
    cm=None
    """Confusion Matrix. ndarray of shape (n_classes, n_classes)"""
    def __init__(self):
        pass

    def plot_roc_curve(self):
        plot_roc_curve(self.base_fpr, self.base_tpr, self.model_fpr,
                       self.model_tpr)

    def plot_confusion_matrix(self,classes=['Background', 'Pre-Seizure'],
                              title='Pre-Seizure Confusion Matrix'):
        plot_confusion_matrix(self.cm, classes=classes, title=title)


class Results:
    data_size = 0
    test_size = 0
    cross_val_fold = 0

    def __init__(self):
        """Object for handling multiple Result's"""
        self.results = {}

    def append(self, key:str, result:Result):
        """

        Args:
            key: key of the result
            result: Result object
        """
        self.results[key] = result

    def plot_roc_curve(self):
        # Plot formatting
        plt.figure(figsize=(8, 6))
        plt.rcParams['font.size'] = 16
        plt.style.use('fivethirtyeight')

        # Plot all curves
        for model, result in self.results.items():
            plt.plot(result.base_fpr, result.base_tpr, 'k',
                     label=model+' baseline')
            plt.plot(result.model_fpr, result.model_tpr,
                     label=model+' model')
        plt.legend();
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate');
        plt.title('ROC Curves');
        plt.show();
