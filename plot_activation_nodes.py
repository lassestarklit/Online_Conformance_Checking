import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def plot_activation_hidden(path_to_csv, feature_process):

    df = pd.read_csv(path_to_csv)
    #Choose results from selected feature process technique
    df = df.loc[(df["feature process"] == feature_process)]
    #Choose results from 500 log
    df = df.loc[(df['errors_pr_mil'] == "['500']")]
    df['hidden layers'] = df['hidden neurons'].astype(str)
    fig = plt.figure()
    activation_functions = df['activation function'].unique()
    num_hidden_layers = df['hidden layers'].unique()


    x=np.arange(len(num_hidden_layers))

    width = 0.35

    fig,ax = plt.subplots()

    offset=[x - width / 2, x + width / 2 ]

    for index,activation in enumerate(activation_functions):
        new_df = df.loc[(df["activation function"] == activation)]
        performance=new_df['F1']
        ax.bar(x + (-1)**index * width / 2, performance, width/2,label=activation)


    ax.set_ylabel('F1')
    ax.set_xticks(x)
    ax.set_xticklabels(num_hidden_layers)
    ax.legend()

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    graph_name = path_to_csv[:-4]

    fig.tight_layout()

    plt.savefig(graph_name + '_graph', bbox_inches='tight')

    plt.show()

