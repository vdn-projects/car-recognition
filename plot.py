# Refer to: https://www.pyimagesearch.com/2017/12/25/plot-accuracy-loss-mxnet/
import matplotlib.pyplot as plt
import numpy as np
import re

# define the paths to the training logs
logs = [
    (85, "./logs/training_50.log")
]

# initialize the list of train rank-1 and rank-5 accuracies, along
# with the training loss
(trainRank1, trainRank5, trainLoss) = ([], [], [])

# initialize the list of validation rank-1 and rank-5 accuracies,
# along with the validation loss
(valRank1, valRank5, valLoss) = ([], [], [])


# loop over the training logs
for (i, (endEpoch, p)) in enumerate(logs):
        # load the contents of the log file, then initialize the batch
        # lists for the training and validation data
    rows = open(p).read().strip()
    (bTrainRank1, bTrainRank5, bTrainLoss) = ([], [], [])
    (bValRank1, bValRank5, bValLoss) = ([], [], [])

    # grab the set of training epochs
    epochs = set(re.findall(r'Epoch\[(\d+)\]', rows))
    epochs = sorted([int(e) for e in epochs])

    # loop over the epochs
    for e in epochs:
        # find all rank-1 accuracies, rank-5 accuracies, and loss
        # values, then take the final entry in the list for each
        s = r'Epoch\[' + str(e) + '\].*accuracy=([0]*\.?[0-9]+)'
        rank1 = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(e) + '\].*top_k_accuracy_5=([0]*\.?[0-9]+)'
        rank5 = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(e) + '\].*cross-entropy=([0-9]*\.?[0-9]+)'
        loss = re.findall(s, rows)[-2]

        # update the batch training lists
        bTrainRank1.append(float(rank1))
        bTrainRank5.append(float(rank5))
        bTrainLoss.append(float(loss))

# extract the validation rank-1 and rank-5 accuracies for each
# epoch, followed by the loss
bValRank1 = re.findall(r'Validation-accuracy=(.*)', rows)
bValRank5 = re.findall(r'Validation-top_k_accuracy_5=(.*)', rows)
bValLoss = re.findall(r'Validation-cross-entropy=(.*)', rows)

# convert the validation rank-1, rank-5, and loss lists to floats
bValRank1 = [float(x) for x in bValRank1]
bValRank5 = [float(x) for x in bValRank5]
bValLoss = [float(x) for x in bValLoss]

# check to see if we are examining a log file other than the
# first one, and if so, use the number of the final epoch in
# the log file as our slice index
if i > 0 and endEpoch is not None:
    trainEnd = endEpoch - logs[i - 1][0]
    valEnd = endEpoch - logs[i - 1][0]

# otherwise, this is the first epoch so no subtraction needs
# to be done
else:
    trainEnd = endEpoch
    valEnd = endEpoch

# update the training lists
trainRank1.extend(bTrainRank1[0:trainEnd])
trainRank5.extend(bTrainRank5[0:trainEnd])
trainLoss.extend(bTrainLoss[0:trainEnd])

# update the validation lists
valRank1.extend(bValRank1[0:valEnd])
valRank5.extend(bValRank5[0:valEnd])
valLoss.extend(bValLoss[0:valEnd])

# plot the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(trainRank1)), trainRank1,
         label="train_rank1")
plt.plot(np.arange(0, len(trainRank5)), trainRank5,
         label="train_rank5")
plt.plot(np.arange(0, len(valRank1)), valRank1,
         label="val_rank1")
plt.plot(np.arange(0, len(valRank5)), valRank5,
         label="val_rank5")
plt.title("rank-1 and rank-5 accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

# plot the losses
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(trainLoss)), trainLoss,
         label="train_loss")
plt.plot(np.arange(0, len(valLoss)), valLoss,
         label="val_loss")
plt.title("cross-entropy loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()
