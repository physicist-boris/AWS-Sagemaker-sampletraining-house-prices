from lightgbm import LGBMRegressor
import joblib
import argparse
import os
import pandas as pd
import re

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--numleaves', type=int, default=4)
    parser.add_argument('--learningrate', type=float, default=0.01)
    parser.add_argument('--nestimators', type=int, default=5000)

    # Data, model, and output directories
    #parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--modeldir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    args, _ = parser.parse_known_args()
    
    model = LGBMRegressor(objective='regression',
                      num_leaves=args.numleaves,
                      learning_rate=args.learningrate,
                      n_estimators=args.nestimators,
                      max_bin=200,
                      bagging_fraction=0.75,
                      bagging_freq=5,
                      bagging_seed=7,
                      feature_fraction=0.2,
                      feature_fraction_seed=7,
                      verbose=-1,
                      )
    df_train = pd.read_csv(os.path.join(args.train, 'Train_House_Prices.csv'))
    df_train_2 = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X_train = df_train_2.drop(columns = df_train.columns[-1])
    y_train = df_train_2[[df_train.columns[-1]]]
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(args.modeldir, "model.joblib"))
