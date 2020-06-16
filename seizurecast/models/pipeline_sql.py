from seizurecast.models.par import LABEL_PRE, LABEL_BKG, LABEL_SEZ
from seizurecast.models.pipeline import Pipeline, Config
import pandas as pd


class Pipeline_sql(Pipeline):
    def __init__(self, conf=Config()):
        """Pipeline for SQL based training"""
        super(Pipeline_sql, self).__init__(conf)

    def dump_xy(self):
        """Dump X y to SQL"""
        raise NotImplementedError

    def load_xy(self):
        """Load X, y from SQL"""
        from seizurecast.features.to_sql import SQLengine
        df = pd.read_sql_table('features', SQLengine)
        X, y_ = df.iloc[:, 0:24*8].to_numpy(), df.loc[:,['post','pres']].to_numpy()

        # y_ has two columns. Convert to y with labels
        y = list(y_)
        for i, y_i in enumerate(y_):
            (post, pres) = y_i
            if post > self.LEN_POS and pres > self.SEC_GAP + self.LEN_PRE:
                y[i] = LABEL_BKG
            elif post > self.LEN_POS and pres > self.SEC_GAP:
                y[i] = LABEL_PRE
            else:
                y[i] = LABEL_SEZ

        return X, y


if __name__ == '__main__':
    psql = Pipeline_sql()
    psql.pipe()
    psql.results.plot_roc_curve()
