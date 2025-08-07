import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

iris_bunch = load_iris(as_frame=True)
df = iris_bunch.frame
df.rename(columns={'target': 'species_id'}, inplace=True)
df['species'] = df['species_id'].map(dict(enumerate(iris_bunch.target_names)))

print("Current working directory:", os.getcwd())
print("Dataframe shape:", df.shape)
print(df.head())
def plot_distribution(df, max_cols=5, cols_per_row=3):
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = num_cols[:max_cols]
    n = len(num_cols)
    rows = (n + cols_per_row - 1) // cols_per_row
    plt.figure(figsize=(5*cols_per_row, 4*rows))
    for idx, col in enumerate(num_cols, 1):
        plt.subplot(rows, cols_per_row, idx)
        df[col].hist(bins=15)
        plt.title(col)
        plt.xlabel('')
    plt.tight_layout()
    plt.show()

plot_distribution(df)
plt.figure(figsize=(6, 6))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
sns.pairplot(df, vars=iris_bunch.feature_names, hue='species', diag_kind='kde', corner=True)
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
features = iris_bunch.feature_names
xs = df[features[0]]
ys = df[features[1]]
zs = df[features[2]]
species = df['species']
for sp in species.unique():
    idx = species == sp

    ax.scatter(xs[idx], ys[idx], zs[idx], label=sp)
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_zlabel(features[2])
ax.legend()
plt.title("3D scatter of Iris features")
plt.show()
