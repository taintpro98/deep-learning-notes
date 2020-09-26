import numpy as np

def cross_entropy(predictions, targets, epsilon=1e-10):
    # predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(targets * np.log(predictions + 1e-5))/N/4
    return ce_loss

# Define the loss function
def loss(y, t):
    return - np.mean(
            np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y))
        )

predictions = np.array([[0.25,0.25,0.25,0.25]])
targets = np.array([[0,0,0,1]])
floss = cross_entropy(predictions, targets)
sloss = loss(predictions, targets)
print ("First cross entropy loss is: " + str(floss))
print ("Second cross entropy loss is: " + str(sloss))

print("-------------------")

predictions = np.array([[0.01,0.01,0.01,0.96]])
targets = np.array([[0,0,0,1]])
floss = cross_entropy(predictions, targets)
sloss = loss(predictions, targets)
print ("First cross entropy loss is: " + str(floss))
print ("Second cross entropy loss is: " + str(sloss))