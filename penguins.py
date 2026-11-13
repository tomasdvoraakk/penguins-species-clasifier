#%%
import seaborn as sns

penguins = sns.load_dataset("penguins")
# %%
print(penguins[:5])
    
print(len(penguins))
print(len(penguins.head()))
print(penguins.head())
print(penguins.dtypes)
# %%
print(penguins["species"].value_counts())
# %%
for i in penguins.columns:
    print(f"{i}: {penguins[i].isna().sum()}")
# %%
model_data = penguins[[
    
        "species",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    
]]
model_data = model_data.dropna()
print(model_data.shape)
# %%
x = model_data.drop(columns=["species"])
y = model_data["species"]

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# %%
# %%
