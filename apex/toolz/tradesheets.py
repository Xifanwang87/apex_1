from apex.toolz.bloomberg import fix_security_name_index
import pandas as pd
import numpy as np


def generate_bloomberg_weight_sheet(weights):
    weights = weights.copy()
    weights = weights[weights.abs() > 1e-5]
    df = pd.DataFrame({'FIXED WEIGHT': weights})
    df['PORTFOLIO NAME'] = 'APEX'
    df['SECURITY_ID'] = [' '.join(x.split(' ')[:2]) for x in fix_security_name_index(df).index.tolist()]
    return df.sort_values(by='FIXED WEIGHT', ascending=False)
