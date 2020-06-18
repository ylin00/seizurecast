from seizurecast.models.par import LABEL_PRE, LABEL_BKG, LABEL_SEZ
from seizurecast.models.pipeline import Pipeline, Config
import pandas as pd


class Pipeline_sql(Pipeline):
    def __init__(self, conf=Config()):
        """Pipeline for SQL based training"""
        super(Pipeline_sql, self).__init__(conf)
        from seizurecast.features.to_sql import SQLengine
        self.engine = SQLengine

    def dump_xy(self):
        """Dump X y to SQL"""
        raise NotImplementedError

    # def load_xy_sql(self, query, engine=None, col_X=None, col_y=None):
    #     import pandas
    #     df = pandas.read_sql(query, engine)
    #     self.X = df.iloc[:, col_X].to_numpy()
    #     self.y = df.iloc[:, col_y].to_numpy()

    def load_xy_default(self,  query=None):
        """Load X, y from SQL"""
        df = pd.read_sql(query, self.engine)
        # SELECT COUNT(*) FROM {table};
        #   SELECT *
        #     FROM {table}
        #    WHERE frozen_rand BETWEEN %(rand_low)s AND %(rand_high)s
        # ORDER BY RAND() LIMIT {int(limit)}
        self.X, self.y = df.iloc[:, 0:24*8].to_numpy(), df.loc[:,['post','pres']].to_numpy()

    def load_xy_random(self, limit = 1):
        pass

    def _postpres2labels(self):
        # y_ has two columns. Convert to y with labels
        y = []
        for i, y_i in enumerate(self.y):
            (post, pres) = y_i
            if post > self.LEN_POS and pres > self.SEC_GAP + self.LEN_PRE:
                y.append(LABEL_BKG)
            elif post > self.LEN_POS and pres > self.SEC_GAP:
                y.append(LABEL_PRE)
            else:
                y.append(LABEL_SEZ)
        self.y = y


if __name__ == '__main__':
    psql = Pipeline_sql()
    psql.load_xy_default(1000)
    psql.pipe()
    psql.results.plot_roc_curve()
