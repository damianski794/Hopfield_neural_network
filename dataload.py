import pandas as pd
import numpy as np

def load(path):
    df = pd.read_csv(path, header=None)
    return df.to_numpy()

