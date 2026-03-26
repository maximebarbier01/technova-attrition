import matplotlib.pyplot as plt
import seaborn as sns

def get_prediction_type(prediction, target):
    if prediction == 1 and target == 1:
        return "true_positive"
    elif prediction == 0 and target == 0:
        return "true_negative"
    elif prediction == 1 and target == 0:
        return "false_positive"
    elif prediction == 0 and target == 1:
        return "false_negative"
    else:
        return "unknow"
    

color_palette = {
    "true_positive" : "green",
    "true_negative": "red",
    "false_positive": "blue",
    "false_negative": "orange"
}

def plot_probability_disrtib_per_pred_type(
        X,
        color_palette=color_palette,
        categories_to_exclude=["true_negative"]
    ):
    X_filtered = X[~X["prediction_type"].isin(categories_to_exclude)]
    sns.displot(data=X_filtered, x="probability_score", hue="prediction_type",palette=color_palette)
    plt.legend(loc="upper right")
    plt.show()