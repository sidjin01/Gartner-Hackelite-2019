#!/usr/bin/env python3
"""Predict client retention."""
import pandas as pd
import category_encoders as ce
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

FIGSIZE = (10, 30)  # the figure size for the feature importance graph


def preprocess(data, train=False):
    """Preprocess input data before training.

    Args:
        data (`pd.DataFrame`): The raw input dataframe
        train (bool): Whether the input contains the client retention flag

    Returns:
        `pd.DataFrame`: The preprocessed data for the model
        `pd.Series`: The corresponding preprocessed labels. This is only
            returned if the `train` argument is set to True.

    """
    # Client ID and Company ID are unnecessary
    data = data.drop(columns=["Client ID", "Company ID"])

    label = "Client Retention Flag"  # the field that is to be predicted
    if train:
        # The output field has ONLY two values: "Yes" and "No". Its value is
        # set to 1 (int form of bool True) if it is "Yes". If it is not "Yes",
        # it is "No", and thus it is set to 0 (int form of bool False).
        data[label] = data[label].apply(lambda x: int(x == "Yes"))

    # Replace categorical columns by one-hot encoded columns.
    # Here, each categorical column is dropped, and multiple columns - each
    # denoting a single feature of the one-hot encoded vector - are added
    # in-place.
    ce_col = list(data.select_dtypes(include=["object"]).columns)
    encoder = ce.OneHotEncoder(cols=ce_col)
    data = encoder.fit_transform(data)

    if train:
        # Return X, y
        return data.drop(columns=[label]), data[label]
    else:
        # Return only X
        return data


def postprocess(pred, test_data):
    """Postprocess the model's outputs before exporting.

    Args:
        pred (`numpy.ndarray`): The model's outputs
        test_data (`pd.DataFrame`): The corresponding test data given to the
            model

    Returns:
        `pd.DataFrame`: The processed outputs for exporting to a CSV

    """
    label = "Client Retention Flag"
    # There is only one unnamed column, so its name is 0
    pred = pred.rename(columns={0: label})
    # Map the model's outputs (which are float values) to their meaning
    pred[label] = pred[label].apply(lambda x: "Yes" if x == 1.0 else "No")

    # This is required as by default pandas will insert the "Client ID"
    # column at the end of the DataFrame.
    # NOTE: This assumes that the order of the clients is not altered anywhere.
    pred.insert(loc=0, column="Client ID", value=test_data["Client ID"])
    return pred


def main(args):
    """Run the main program.

    Arguments:
        args (`argparse.Namespace`): The object containing the commandline
            arguments

    """
    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)
    X_train, y_train = preprocess(train_data, train=True)
    X_test = preprocess(test_data)

    # Hyperparams obtained using Hyperopt's hyperparam tuning methods
    model = XGBClassifier(
        base_score=0.5315979758590659,
        colsample_bytree=0.763171792007325,
        gamma=4.8396534022477855,
        learning_rate=0.26738381031741343,
        max_depth=20,
        n_estimators=200,
        random_state=3,  # for reproducibility
        reg_alpha=1.1573581164451705,
        reg_lambda=0.763295630042223,
        scale_pos_weight=1.6706777186920017,
        n_jobs=args.jobs,  # should be < no. of CPU cores, for efficiency
    )
    model.fit(X_train, y_train)

    if args.plot is not None:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        plot_importance(model, ax=ax)
        fig.savefig(args.plot, bbox_inches="tight")

    y_pred = pd.DataFrame(model.predict(X_test))
    y_pred = postprocess(y_pred, test_data)
    y_pred.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Predict client retention",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "train", metavar="TRAIN", type=str, help="path to the training dataset"
    )
    parser.add_argument(
        "test", metavar="TEST", type=str, help="path to the test dataset"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./Submission.csv",
        help="path where to save the output CSV",
    )
    parser.add_argument(
        "-p",
        "--plot",
        type=str,
        help="path where to save the feature importance graph",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=6,
        help="number of parallel jobs for XGBoost",
    )
    main(parser.parse_args())
