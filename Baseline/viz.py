import matplotlib.pyplot as plt
def plot_predictions(true, pred, node_idx=0, savepath=None):
    plt.figure(figsize=(8,3))
    plt.plot(true, label='True')
    plt.plot(pred, label='Pred')
    plt.legend()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()