import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot(path_to_csv,training_data,testing_data):
    df = pd.read_csv(path_to_csv)



    fig = plt.figure()
    data_processing=['Embedded','One hot encoding','Freq one hot encoding']
    with_faulty = [True,False]

    for proc in data_processing:
        for faulty in with_faulty:

            newdf=df.loc[(df["data_processing"]==proc) & (df["move/state discriminating"]==faulty)]
            plt.plot(newdf['errors_pr_mil'], newdf['accuracy'],marker='o',
                     label= '{0}(with faulty)'.format(proc) if faulty
                     else '{0}(no faulty)'.format(proc))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.title("Trained on: {0}\nTested on: {1}".format(training_data,testing_data))
    graph_name = path_to_csv[:-4]

    plt.savefig(graph_name + "_graph", bbox_inches='tight')

    #plt.show()

