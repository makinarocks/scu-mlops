import argparse

parser = argparse.ArgumentParser(description="Pipeline Parameter")
parser.add_argument(
    "--modelpath",
    type=str,
    default="model.pkl",
    nargs="?",
    help='type=str, default="model.pkl"',
)
parser.add_argument(
    "--datapath",
    type=str,
    default="data_dumped.csv",
    nargs="?",
    help='type=str, default="data_dumped.csv"',
)

args = parser.parse_args()


if __name__ == "__main__":

    modelpath = args.modelpath
    datapath = args.datapath

    # =============================================================
    # Component: Load data
    # Parents: []
    import pandas as pd

    dat = pd.read_csv(datapath, index_col=0)

    # =============================================================
    # Component: Create model
    # Parents: []
    from sklearn.linear_model import LinearRegression

    regr = LinearRegression()

    # =============================================================
    # Component: Fit model
    # Parents: ['Create model', 'Load data']
    regr.fit(dat["x"].values[:, None], dat["y"])

    # =============================================================
    # Component: Save model
    # Parents: ['Fit model']
    import pickle

    with open(modelpath, "wb") as fp:
        pickle.dump(regr, fp)
