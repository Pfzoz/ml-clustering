
import pandas as pd

data = pd.read_csv("data.csv")
new_data = data.iloc[0:0].copy()


for i in range(2000):
    ex_row = data.sample().iloc[0]
    new_data.loc[len(new_data)] = ex_row

new_data.to_csv("new_data.csv")
