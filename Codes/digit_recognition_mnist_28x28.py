import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


TRAIN_CSV = "mnist_train.csv"
TEST_CSV = "mnist_test.csv"
RESULTS_CSV = "mnist_28x28_model_results.csv"
CONFUSION_PNG = "mnist_28x28_confusion_matrix.png"
K_BEST = 200


def load_official_mnist(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values
    X_test = test_df.drop("label", axis=1).values
    y_test = test_df["label"].values

    return X_train, y_train, X_test, y_test


def build_models(k_best: int):
    return {
        "Decision Tree": Pipeline([
            ("scaler", MinMaxScaler()),
            ("select", SelectKBest(score_func=chi2, k=k_best)),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", MinMaxScaler()),
            ("select", SelectKBest(score_func=chi2, k=k_best)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", MinMaxScaler()),
            ("select", SelectKBest(score_func=chi2, k=k_best)),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "SVM": Pipeline([
            ("scaler", MinMaxScaler()),
            ("select", SelectKBest(score_func=chi2, k=k_best)),
            ("model", SVC(kernel="rbf", gamma="scale")),
        ]),
        "ANN (MLP)": Pipeline([
            ("scaler", MinMaxScaler()),
            ("select", SelectKBest(score_func=chi2, k=k_best)),
            ("model", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", MinMaxScaler()),
            ("select", SelectKBest(score_func=chi2, k=k_best)),
            ("model", GaussianNB()),
        ]),
    }


def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = []
    best_model_name = None
    best_accuracy = 0.0
    best_predictions = None

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append({"Model": name, "Accuracy": acc})
        print(f"{name} Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_predictions = y_pred

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    return results_df, best_model_name, best_accuracy, best_predictions


def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_sample_predictions(model, X_test, y_test, count=10):
    preds = model.predict(X_test[:count])
    plt.figure(figsize=(10, 4))
    for i in range(count):
        img = X_test[i].reshape(28, 28)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"T:{y_test[i]} P:{preds[i]}")
        plt.axis("off")
    plt.suptitle("Test Predictions")
    plt.show()


def main():
    X_train, y_train, X_test, y_test = load_official_mnist(TRAIN_CSV, TEST_CSV)
    print("Training samples:", X_train.shape)
    print("Testing samples:", X_test.shape)

    models = build_models(K_BEST)
    results_df, best_name, best_acc, best_preds = evaluate_models(
        models, X_train, y_train, X_test, y_test
    )

    print("\nFinal Results:")
    print(results_df)
    results_df.to_csv(RESULTS_CSV, index=False)

    labels = list(range(10))
    plot_confusion_matrix(
        y_test,
        best_preds,
        labels,
        f"Confusion Matrix for {best_name}",
        CONFUSION_PNG,
    )

    print(f"\nBest Model: {best_name}")
    print(f"Best Accuracy: {best_acc:.4f}")

    best_model = models[best_name]
    plot_sample_predictions(best_model, X_test, y_test, count=10)


if __name__ == "__main__":
    main()
