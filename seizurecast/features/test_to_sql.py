import pandas as pd
from seizurecast.features.to_sql import SQLengine


def disabled_test_write_tables_to_sql():
    print(pd.read_sql_table('features', SQLengine, index_col='index').head().to_csv("../../tmp/tmp.csv"))
    assert True
