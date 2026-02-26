import pandas as pd

data = pd.read_pickle(r"D:\清华\毕业设计\test\datareader_new\jqrd\outcome\jqrd燃机1_1min.pkl")

df = data[["余热锅炉出口烟气流量", "1#燃气流量", "APPARENT POWER"]].copy()



df.to_csv(r"D:\清华\毕业设计\test\temp\massflow.csv")



