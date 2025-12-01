import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import argparse
from sklearn.linear_model import LogisticRegression

pd.options.mode.chained_assignment = None

def create_parser():
    parser = argparse.ArgumentParser(prog="Text Classifier")
    parser.add_argument("-d", help="Path to csv data")
    parser.add_argument("-p", help="Phase: (val, test)")
    return parser

def create_dataframe(args):
    df = pd.read_csv(args.d)
    df = df.drop("pe_type", axis=1)
    return df

def train(df):
    train_df = df.loc[df["split"] == "train"]
    x_train = train_df.drop(["idx", "label", "split"], axis=1).to_numpy()
    y_train = train_df["label"].to_numpy()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    clf = LogisticRegression(random_state=42, solver="liblinear", penalty="l1", C=0.03, max_iter=500)
    clf.fit(x_train, y_train)
    return clf, scaler

def validation(clf, scaler, df):
    val_df = df.loc[df["split"] == "val"]
    x_test = val_df.drop(["idx", "label", "split"], axis=1).to_numpy()
    y_test = val_df["label"].to_numpy() 
    x_test = scaler.transform(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1]
    print(f"Validation AUC-ROC: {metrics.roc_auc_score(y_test, y_prob):.3f}")
    exported_df = val_df[["idx", "label"]]
    exported_df["text_preds"] = y_prob
    exported_df.to_csv("./validation_proba.csv")

def test(clf, scaler, df):
    test_df = df.loc[df["split"] == "test"]
    x_test = test_df.drop(["idx", "label", "split"], axis=1).to_numpy()
    y_test = test_df["label"].to_numpy() 
    x_test = scaler.transform(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1]
    print(f"Test AUC-ROC: {metrics.roc_auc_score(y_test, y_prob):.3f}")
    exported_df = test_df[["idx", "label"]]
    exported_df["text_preds"] = y_prob
    exported_df.to_csv("./test_proba.csv")

if __name__ == "__main__":
    args = create_parser().parse_args()
    dataframe = create_dataframe(args)
    clf, scaler = train(dataframe)
    if(args.p == "val"):
        validation(clf, scaler, dataframe)
    elif(args.p == "test"):
        test(clf, scaler, dataframe)
