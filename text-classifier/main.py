import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import argparse
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

def create_parser():
    parser = argparse.ArgumentParser(prog="Text Classifier")
    parser.add_argument("-d", help="Path to csv data")
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
    x_train, x_train_val, y_train, y_train_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=123,
        eval_metric='auc',
        early_stopping_rounds=10,
        alpha=10,
        objective="binary:logistic",
        colsample_bytree=0.3
    )
    clf.fit(x_train, y_train, eval_set=[(x_train_val, y_train_val)], verbose=True)
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
    validation(clf, scaler, dataframe)
    test(clf, scaler, dataframe)
