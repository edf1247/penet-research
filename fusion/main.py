import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import argparse
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.metrics import precision_recall_curve

def create_parser():
    parser = argparse.ArgumentParser(prog="Fusion model")
    parser.add_argument("--penet-val", help="Path to penet validation predictions")
    parser.add_argument("--penet-test", help="Path to penet test predictions")
    parser.add_argument("--text-val", help="Path to text clf val predictions")
    parser.add_argument("--text-test", help="Path to text clf test predictions")
    return parser

def train_classifier(args):
    try:
        penet_val = pd.read_pickle(args.penet_val)
    except:
        print("Please enter a valid file path for penet_val.")
        exit()
    
    try:
        text_val = pd.read_csv(args.text_val)
    except:
        print("Please enter a valid file path for text_val.")
        exit()

    for idx in penet_val:
        penet_val[idx]["text_val"] = text_val.loc[text_val["idx"] == idx]["text_preds"].to_numpy()[0]

    training_df = pd.DataFrame.from_dict(penet_val, orient="index")
    x_train = training_df[["pred", "text_val"]].to_numpy()
    y_train = training_df["label"].to_numpy()

    clf = LogisticRegression().fit(x_train, y_train)

    y_prob_val = clf.predict_proba(x_train)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_prob_val)
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]

    return clf, optimal_threshold

def test_classifier(args, clf, optimal_threshold):
    try:
        penet_test = pd.read_pickle(args.penet_test)
    except:
        print("Please enter a valid file path for penet_test.")
        exit()
    
    try:
        text_test = pd.read_csv(args.text_test)
    except:
        print("Please enter a valid file path for text_test.")
        exit()
    
    for idx in penet_test:
        penet_test[idx]["text_test"] = text_test.loc[text_test["idx"] == idx]["text_preds"].to_numpy()[0]
    
    training_df = pd.DataFrame.from_dict(penet_test, orient="index")

    x_test = training_df[["pred", "text_test"]].to_numpy()
    y_test = training_df["label"].to_numpy()

    accuracy = clf.score(x_test, y_test)
    y_prob = clf.predict_proba(x_test)[:, 1]

    print(f"AUC-ROC: {metrics.roc_auc_score(y_test, y_prob):.3f}")
    print(f"Accuracy with default threshold: {accuracy}")
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    print(f"Accuracy with validation-derived optimal threshold ({optimal_threshold:.3f}): {metrics.accuracy_score(y_test, y_pred_optimal)}")
    create_roc_plot(y_test, y_prob)

def create_roc_plot(y_test, y_prob):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("./auroc.png")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if len(sys.argv) <= 1:
        parser.print_help()
    else:
        clf, optimal_threshold = train_classifier(args)
        test_classifier(args, clf, optimal_threshold)