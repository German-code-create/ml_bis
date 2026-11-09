import pandas as pd
from config import PATH

df = pd.read_csv(PATH)
df = df.drop(columns=["instant", "dteday", "casual", "registered", "mnth", "temp"])
