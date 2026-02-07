import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("data\FINAL_PROJECT_DATASET.csv")

"""EDA"""

print(df["Strength"].describe())

#Outlier Detection
sns.boxplot(x=df["Strength"])
plt.title("Outlier detection for Compression strength occupied in N days")
plt.show()

#Correlation matrixes
attributes = ["Cement","GGBS", "FlyAsh", "Water", "CoarseAggregate", "Sand", "Admixture","age"]
strength_cols = ["Strength"]

for attr in attributes:
    for i, col in enumerate(strength_cols):
        corr_value = df[attr].corr(df[col])
        corr_matrix = pd.DataFrame([[corr_value]], index=[attr], columns=[col])
        
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()

"""Feature Engineering"""
#Creating features DataFrame
df2=df.copy()

#WBRatio
df2["WBRatio"]=df["Water"]/(df["Cement"]+df["FlyAsh"]+df["GGBS"])

# Save df_copy to a CSV file
df2.to_csv("data/features.csv", index=False)