"""From raw data to evaluated model"""
import pickle

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from src.models.Result import Result, Results
from src.features.dataset_funcs import balance_ds, get_power, get_power_freq
from src.data.file_io import listdir_edfs
from src.data.make_dataset import make_dataset
import numpy as np

from src.models.model import evaluate_model
from src.models.par import LABEL_BKG, LABEL_PRE
from src.utils import dataset2Xy


class TrainError(Exception):
    pass


class Config:
    len_pre = 300
    len_pos = 600
    sec_gap = 600
    sampl_r = 256
    pass


class Pipeline:
    def __init__(self, config:Config):
        self.token_paths = None
        self.LEN_PRE, self.LEN_POS, self.SEC_GAP, self.SAMPLING_RATE = \
            config.len_pre, config.len_pos, config.sec_gap, config.sampl_r

        self.__verbose = True

        self.__ncv = 5
        """Fold of cross validation"""
        self.__test_size = 0.2
        """Fraction of test dataset"""

        self.scores_CV = 0
        self.scores_Test = 0
        self.results = Results()

    def dump_xy(self):
        with open('../../data/processed/xy.pkl', 'wb') as fp:
            for ipath, token_path in enumerate(self.token_paths):
                if self.__verbose:
                    print(f'dumping: {token_path}')
                _X, _y = self.__Xy_from_one(token_path)
                pickle.dump((_X, _y), fp)

    def load_xy(self):
        X, y = [], []
        with open('../../data/processed/xy.pkl', 'rb') as fp:
            for i in range(0, 1000000):
                try:
                    _X, _y = pickle.load(fp)
                    X.extend(_X), y.extend(_y)
                except EOFError:
                    break
        return X, y

    def pipe(self):
        X, y = self.load_xy()
        self.scores_CV, self.scores_Test, eval_result, models = \
            self.__eval_many(X, y)

        # convert to results
        for key, evls in eval_result.items():
            res = Result()
            res.model_name = key
            res.model = models[key]
            res.model_tpr = evls['model_tpr']
            res.model_fpr = evls['model_fpr']
            res.base_tpr = evls['base_tpr']
            res.base_fpr = evls['base_fpr']
            self.results.append(key, res)
        self.results.data_size = len(X)
        self.results.cross_val_fold = self.__ncv
        self.results.test_size = self.__test_size

    def __Xy_from_one(self, token_path):
        # load dataset
        dataset, labels = make_dataset([token_path],
                                       len_pre=self.LEN_PRE,
                                       len_post=self.LEN_POS,
                                       sec_gap=self.SEC_GAP,
                                       fsamp=self.SAMPLING_RATE)

        # balance data
        dataset, labels = balance_ds(dataset, labels, seed=100)

        print(f"Collected {len(labels)} data points") if self.__verbose else None

        # feature extraction
        dataset_power = get_power(dataset, fsamp=self.SAMPLING_RATE)

        # convert to Xy
        ds_pwd = get_power_freq(dataset_power)
        X, y = dataset2Xy(ds_pwd, labels)

        return X, y

    def __post_process(self, X, y):
        # filtered out classes
        id_bkg_pre = [any([yi == lbl for lbl in [LABEL_BKG, LABEL_PRE]]) for
                      yi in y]
        X = np.array(X)[id_bkg_pre, :]
        y = np.array(y)[id_bkg_pre]

        # balance data again
        X, y = balance_ds(X, y, seed=100)
        return X, y

    def __eval_many(self, X, y):
        """Evaluate X and y"""
        X, y = self.__post_process(X, y)

        print(f"Collected {len(y)} data points") if self.__verbose else None

        if len(np.unique(y)) < 2:
            raise TrainError("# of unique values of y must >= 2")

        # Binarize the label
        y_b = preprocessing.label_binarize(y, classes=[LABEL_BKG, LABEL_PRE])
        y_b = np.reshape(y_b, (len(y_b),))

        # Train test split
        train_X, test_X, train_y, test_y = \
            train_test_split(X, y_b, test_size=self.__test_size,random_state=41)

        # Models
        clf = {}
        # Linear Model
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf['lda'] = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf['lg'] = LogisticRegression(random_state=0, max_iter=2000)
        clf['rf'] = RandomForestClassifier(n_estimators=40, max_depth=None,
                                           min_samples_split=3, random_state=0)
        # TODO: CNN+LSTM model

        # cv
        cvscores = {}
        for k in clf.keys():
            cvscores[k] = cross_val_score(clf[k], train_X, train_y,
                                         cv=self.__ncv)

        # fit
        scores = {}
        for k, v in clf.items():
            v.fit(train_X, train_y)
            scores[k] = v.score(test_X, test_y)

        # evaluate
        evalres = {}
        for k, model in clf.items():
            # Training predictions (to demonstrate overfitting)
            train_rf_predictions = model.predict(train_X)
            train_rf_probs = model.predict_proba(train_X)[:, 1]

            # Testing predictions (to determine performance)
            rf_predictions = model.predict(test_X)
            rf_probs = model.predict_proba(test_X)[:, 1]

            base_fpr, base_tpr, model_fpr, model_tpr = evaluate_model(
                rf_predictions, rf_probs, train_rf_predictions, train_rf_probs,
                train_y, test_y, verbose=self.__verbose)

            evalres[k]={'base_fpr':base_fpr,
                        'base_tpr':base_tpr,
                        'model_fpr':model_fpr,
                        'model_tpr':model_tpr}

        return cvscores, scores, evalres, clf


if __name__ == '__main__':
    edfs = listdir_edfs()
    conf = Config()
    pipe = Pipeline(conf)

    """Dump edf files to Xy (~ 6 hours)"""
    # pipe.token_paths = edfs['token_path'].to_numpy()
    # dump0 = time()
    # print(f'Dumping {len(pipe.token_paths)} files')
    # pipe.dump_xy()
    # print(f'Dump cost = {round(time()-dump0)} s')

    pipe.pipe()

    # save the RF classifier
    with open('../../models/model.pkl', 'wb') as f:
        pickle.dump(pipe.results.results['rf'].model, f)

    pipe.results.plot_roc_curve()
