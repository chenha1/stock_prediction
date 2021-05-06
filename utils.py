import seaborn as sns 
import matplotlib.pyplot as plt

def plot(original, predict):
    sns.set_style("darkgrid")    

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 1, 1)
    ax = sns.lineplot(x = original.index, y = original[3], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[3], label="Training Prediction", color='tomato')
    ax.set_title('Stock price', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Cost (USD)", size = 14)
    ax.set_xticklabels('', size=10)
    plt.show()