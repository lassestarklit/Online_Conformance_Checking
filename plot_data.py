import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def plot(path_to_csv,logs,performance_metric_to_plot):
    df = pd.read_csv(path_to_csv)

    fig = plt.figure()

    loss_functions = ['binary_crossentropy','MSE','categorical_crossentropy']
    dif_logs = logs

    x=np.arange(len(dif_logs))
    width = 0.35

    fig,ax = plt.subplots()

    offset=[x - width / 3, x, x + width / 3 ]
    for index,loss in enumerate(loss_functions):
        new_df = df.loc[(df["loss function"] == loss)]
        performance=new_df[performance_metric_to_plot]
        ax.bar(offset[index], performance, width/3,label=loss)


    ax.set_ylabel(performance_metric_to_plot)
    ax.set_xticks(x)
    ax.set_xticklabels(dif_logs)
    ax.legend()

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    graph_name = path_to_csv[:-4]
    fig.tight_layout()

    plt.savefig(graph_name + '_' + performance_metric_to_plot + '_graph', bbox_inches='tight')

    plt.show()

