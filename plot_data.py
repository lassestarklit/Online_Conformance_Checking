import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot(path_to_csv):
    df = pd.read_csv(path_to_csv)

    #print(df["data_processing"]=="Embedded" )
    #print(df["move/state discriminating"]==False)

    fig = plt.figure()
    data_processing=['Embedded','One hot encoding']
    with_faulty = [True,False]

    for proc in data_processing:
        for faulty in with_faulty:
            print(proc,faulty)
            newdf=df.loc[(df["data_processing"]==proc) & (df["move/state discriminating"]==faulty)]
            plt.plot(newdf['errors_pr_mil'], newdf['accuracy'],marker='o',
                     label= '{0}(with faulty)'.format(proc) if faulty
                     else '{0}(no faulty)'.format(proc))

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
