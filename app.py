import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from tabulate import tabulate

# Load Dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(tabulate(df.head(), headers="keys", tablefmt="grid"))

# Explore structure
print("\nData types present include:")
print(tabulate(pd.DataFrame(df.dtypes, columns=["Data Type"]), headers="keys", tablefmt="grid"))

# Clean data after showing missing values
print("\nMissing Values:")
print(tabulate(pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]), headers="keys", tablefmt="grid"))
df = df.dropna()

# Basic statistics
print("\nDescriptive Statistics:")
print(tabulate(df.describe(), headers="keys", tablefmt="grid"))

# Grouping by species
print("\nAverage measurements per species:")
print(tabulate(df.groupby("target").mean(), headers="keys", tablefmt="grid"))

# Map target numbers to species names
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

# --- Visualizations ---

sns.set(style="whitegrid")

# 1. Line Chart – Average sepal length per sample index
plt.figure(figsize=(8, 4))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Sepal Length over Index")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart – Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x="species", y="petal length (cm)", hue="species", data=df, estimator="mean", palette="pastel", legend=False)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram – Sepal Width Distribution
plt.figure(figsize=(6, 4))
plt.hist(df["sepal width (cm)"], bins=10, color="skyblue", edgecolor="black")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot – Sepal Length vs Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()

# Observations
print("\nFindings:")
print("- Setosa generally has shorter petal lengths than other species.")
print("- There’s a positive correlation between sepal length and petal length.")
print("- Sepal width has a relatively normal distribution.")
