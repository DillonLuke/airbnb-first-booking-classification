import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score


def extract_datetime_units(data: pd.DataFrame, features, formatting, units):
    data = data.copy()
    
    for feat, fmt in zip(features, formatting):
        dt_feat = pd.to_datetime(data[feat], format=fmt)
    
        for unit in units:
            data[feat + "_" + unit] = getattr(dt_feat.dt, unit).astype("object")
    
    data = data.drop(features, axis=1)
            
    return data


def univariate_visualization(data: pd.DataFrame, data_type: str, orientation: str,
                             ncols: int = 3, figsize: tuple = (3, 3)):
    plot_dict = {"categorical": sns.countplot, "numeric": sns.histplot}
    orient_dict ={"v": {"x": "value"}, "h": {"y": "value"}}
    
    data_for_plot = (data
                     .stack()
                     .reset_index(level=1)
                     .set_axis(["feature", "value"], axis=1))
    
    w, h = figsize
    fg = sns.FacetGrid(data=data_for_plot, col="feature", col_wrap=ncols, height=h,
                       aspect=w/h, sharex=False, sharey=False)
    
    fg.map_dataframe(plot_dict[data_type], **orient_dict[orientation])
    

def make_heatmap(x, y, *args, **kwargs):
    data = kwargs["data"]
    
    kwargs["data"] = pd.crosstab(index=data[y],
                                 columns=data[x],
                                 normalize="columns")
    
    sns.heatmap(**kwargs)
    
                                                                                
def bivariate_visualization(X: pd.DataFrame, y: pd.Series, data_type: str, 
                            ncols: int = 3, figsize: tuple = (3, 3), **kwargs):
    plot_type = {"categorical": make_heatmap, "numeric": sns.violinplot}
    
    data = pd.concat((X, y), axis=1)
    
    features, target = data.columns[:-1].tolist(), data.columns[-1]
    
    data_for_plot = (data.melt(id_vars=(target), 
                               value_vars=features, 
                               var_name="feature"))
    
    w, h = figsize
    fg = sns.FacetGrid(data=data_for_plot, col="feature", col_wrap=ncols, height=h,
                       aspect=w/h, sharex=False, sharey=False)
    
    fg.map_dataframe(plot_type[data_type], x=target, y="value", **kwargs)
        
        
def get_NDCG(y_true, y_pred, classes, k: int):
    y_true, classes = np.asarray(y_true), np.asarray(classes)
    
    y_true_by_class = (y_true.reshape(-1, 1) == classes.reshape(-1)).astype(int)
    
    return ndcg_score(y_true_by_class, y_pred, k=k)


def get_top_5_predictions(predict_proba_array: np.ndarray, classes: np.ndarray):
    ranks = predict_proba_array.argsort(axis=1)
    
    return np.array([classes[rank][:-6:-1] for rank in ranks])


def svm_ndcg_scorer(estimator, X, y):
    y_pred = estimator.decision_function(X)
    
    return get_NDCG(y, y_pred, classes=np.arange(0, 12), k=5)

