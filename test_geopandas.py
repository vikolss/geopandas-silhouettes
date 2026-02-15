import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np
import multiprocessing as mp
import pandas as pd
from geopandas_functions import *
import geodatasets

from extract_data import Event


def main():
    print("---FIRST---")
    
    print(Event.__init__.__doc__)
    
    df = pd.DataFrame({
        "data": [1, 23]
    })

    event = Event(df)

if __name__ == "__main__":
    main()
