import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_excel("./enrollment_dataset.xlsx")
plt.scatter(df['Age'], df['Gender'], color='blue')