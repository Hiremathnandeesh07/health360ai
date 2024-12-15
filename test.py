import pickle
from sklearn import tree
import matplotlib.pyplot as plt

# Load the model
with open('models/svc.pkl', 'rb') as file:
    model = pickle.load(file)

# Check if it's a decision tree model and visualize
if isinstance(model, tree.DecisionTreeClassifier):
    plt.figure(figsize=(20, 10))  # Control the size of the plot
    tree.plot_tree(model, filled=True)
    plt.show()
