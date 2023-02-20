'''
    "dataset.csv" dosyasından veri setini okur.
'''

import pandas as pd


def ReadCsv():
    data = pd.read_csv('C:\\Users\\Elif\\Desktop\\bitirmeProjesi\\dataset.csv')
    df = pd.DataFrame(data)
    print(df)












ReadCsv()
