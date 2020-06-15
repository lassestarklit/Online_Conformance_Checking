import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def plot(path_to_csv,logs,performance_metric_to_plot):

    df = pd.read_csv(path_to_csv)
    #alpha 2 and sigmoid activation function
    df = df.loc[(df["alpha"] == 2) &  (df["activation function"] == 'sigmoid')]


    #Choose results from 500 log
    #df = df.loc[(df['errors_pr_mil'] == "['500']")]

    feature_process = df['feature process'].unique()
    fig = plt.figure()


    dif_logs = logs

    x=np.arange(len(dif_logs))
    width = 0.35

    fig,ax = plt.subplots()


    for index,feature in enumerate(feature_process):
        new_df = df.loc[(df["feature process"] == feature)]
        performance=new_df[performance_metric_to_plot]
        ax.bar(x + (-1)**index * width / 2, performance, width/2,label=feature)



    ax.set_ylabel(performance_metric_to_plot)
    ax.set_xticks(x)
    ax.set_xticklabels(dif_logs)
    ax.legend()

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    graph_name = path_to_csv[:-4]

    fig.tight_layout()

    plt.savefig(graph_name + '_' + performance_metric_to_plot + '_graph', bbox_inches='tight')

    plt.show()

