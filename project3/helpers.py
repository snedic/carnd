from matplotlib import pyplot as plt

def plotLossHistory(histObj, saveFileName='msePerEpoch.png', title='Model MSE Loss',
                    yLabel='MSE Loss', xLabel='Epoch', legend=['Train','Validate']):
    ### plot the training and validation loss for each epoch
    plt.plot(histObj.history['loss'])
    plt.plot(histObject.history['val_loss'])
    plt.title(title)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend(legend, loc='upper right')
    #plt.show()

    plt.savefig('LossPerEpoch.png')
