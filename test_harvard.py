from skmultiflow import drift_detection as dd
from river import drift
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


detectors = [
    dd.DDM(min_num_instances=100),
    dd.EDDM(),
    dd.KSWIN(),
    dd.ADWIN(),
]


detectors2 = [
    drift.DDM(),
    drift.EDDM(),
    drift.KSWIN(),
    drift.ADWIN(),
]


def drift_detect_skmultiflow(data, method):
    drifts = []
    for i in range(len(data)):
        method.add_element(data[i])
        if method.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(data[i]) + ' - of index: ' + str(i))
        if method.detected_change():
            print('Change has been detected in data: ' + str(data[i]) + ' - of index: ' + str(i))
            drifts.append(i)

    return drifts


def drift_detect_river(data, method):
    drifts = []
    for i, val in enumerate(data):
        method.update(val)   # Data is processed one sample at a time
        if method.change_detected:
            # The drift detector indicates after each sample if there is a drift in the data
            print(f'Change detected at index {i}')
            drifts.append(i)
            method.reset()   # As a best practice, we reset the detector

    return drifts


def test_drift(modelpath, datapath, colnames):
    df = pd.read_csv(datapath)[:20000]
    df['class'] = df['class'].astype(int)
    cols = df.columns
    label = 'class'
    X, y = df[[c for c in cols if c != label]], df[label]
    model = pickle.load(open(modelpath, 'rb'))
    df['predict'] = model.predict(X)
    # FIND OTHER METRICS:
    # misclassification
    df['misclassified'] = df['predict'] != df[label]
    df['misclassified'] = df['misclassified'].apply(lambda x: 1 if x is True else 0)
    df['measure'] = df['misclassified'].rolling(100, min_periods=1).mean()

    # F1 score

    # Accuracy (same as misclassification)
    
    for d in detectors2:
        sns.lineplot(data=df, x=df.index, y='measure')
        print('Run with drift detecting method: ', d.__class__.__name__)
        drifts = drift_detect_river(df['measure'], d)
        #if drifts is not None and len(drifts) > 0:
        #    plot_data(data, dist_a, dist_b, dist_c, drifts)
        if drifts is not None:
            for drift_detected in drifts:
                plt.axvline(drift_detected, color='red')
        plt.show()


if __name__ == '__main__':
    colnames = ['X1', 'X2', 'class']
    modelpath = 'XGBClassifier.pkl'
    #datapath = 'data/harvard_synthetic_data/sine_0123_abrupto.csv'
    datapath = 'data/harvard_synthetic_data/sine_0123_gradual.csv'
    test_drift(modelpath, datapath, colnames)


