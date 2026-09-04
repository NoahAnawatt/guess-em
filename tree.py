from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.tree import export_text

MAX_COMBINATORIC_DEPTH = 3
MAX_DEPTH = 5

dataset = pd.read_csv("Guess_Who_Dataset.csv", header=None)
dataset = dataset.set_index([0]).transpose()

x_raw = dataset.drop(columns="Name")
y_raw = dataset["Name"]

x_encoded = pd.get_dummies(x_raw)

col_to_orig = {
    oh_col: orig_col
    for orig_col in x_raw.columns
    for oh_col in pd.get_dummies(x_raw[[orig_col]]).columns
}

x_dict = {}

for depth in range(1, MAX_COMBINATORIC_DEPTH + 1):
    combo_list = list(combinations(x_encoded.columns, depth))
    for combo in tqdm(combo_list, desc=f"Depth {depth} AND"):
        orig_features = [col_to_orig[c] for c in combo]
        if len(orig_features) != len(set(orig_features)):
            continue
        feature_name = " AND ".join(combo)
        x_dict[feature_name] = x_encoded[list(combo)].all(axis=1).astype(int)

for depth in range(1, MAX_COMBINATORIC_DEPTH + 1):
    combo_list = list(combinations(x_encoded.columns, depth))
    for combo in tqdm(combo_list, desc=f"Depth {depth} OR"):
        feature_name = " OR ".join(combo)
        x_dict[feature_name] = x_encoded[list(combo)].any(axis=1).astype(int)

x = pd.DataFrame(x_dict, index=x_encoded.index)

label = LabelEncoder()
y = label.fit_transform(y_raw)

clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=MAX_DEPTH)
clf = clf.fit(x, y)

print("Data fed to model:")
print(x)

rules = export_text(clf,
                    feature_names=x.columns.tolist(),
                    show_weights=True
                    )
print(rules)
tree.plot_tree(
    clf,
    feature_names=x.columns.tolist(),
    class_names=label.classes_.tolist(),
    filled=True,
    fontsize=3,
)
plt.tight_layout()
plt.show()

mt = clf.tree_
node = 0
while mt.feature[node] != -2:
  q = x.columns[mt.feature[node]]
  ans = input(f"Does the character have [{q}]? (y/n): ").strip().lower()
  node = (
      mt.children_right[node]
      if ans in ["y", "yes"]
      else mt.children_left[node]
  )

guess = label.classes_[mt.value[node].argmax()]
print(f"Solution solved: {guess}")
