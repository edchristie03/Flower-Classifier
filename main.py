import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from randomclassifier import RandomClassifier
from nearestneighbour import NearestNeighbourClassifier

def read_dataset(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array.
               - x is a numpy array with shape (N, K),
                   where N is the number of instances
                   K is the number of features/attributes
               - y is a numpy array with shape (N, ), and each element should be
                   an integer from 0 to C-1 where C is the number of classes
               - classes : a numpy array with shape (C, ), which contains the
                   unique class labels corresponding to the integers in y
    """
    lines = []
    class_list = []
    for line in open(filepath, 'r'):
      if line.strip() != "": # Skip empty lines
        line = line.strip().split(',')
        lines.append(list(map(float, line[:-1])))
        class_list.append(line[4])

    [classes, y] = np.unique(class_list, return_inverse=True) # Unique elements and indicies

    # Convert to numpy arrays
    x = np.array(lines)
    y = np.array(y)

    return (x, y, classes)

# Pre process data into training and test data

def split_dataset(x, y, test_proportion, random_generator=default_rng()):
    """ Split dataset into training and test sets, according to the given
        test set proportion.

    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        test_proportion (float): the desired proportion of test examples
                                 (0.0-1.0)
        random_generator (np.random.Generator): A random generator

    Returns:
        tuple: returns a tuple of (x_train, x_test, y_train, y_test)
               - x_train (np.ndarray): Training instances shape (N_train, K)
               - x_test (np.ndarray): Test instances shape (N_test, K)
               - y_train (np.ndarray): Training labels, shape (N_train, )
               - y_test (np.ndarray): Test labels, shape (N_train, )
    """

    N_test = int(test_proportion * len(x))
    indicies = np.arange(len(x))
    random_generator.shuffle(indicies)
    test_indicies = indicies[:N_test]
    train_indicies = indicies[N_test:]

    x_train = x[train_indicies]
    x_test = x[test_indicies]
    y_train = y[train_indicies]
    y_test = y[test_indicies]

    return x_train, x_test, y_train, y_test

def compute_accuracy(y_gold, y_prediction):
    """ Compute the accuracy given the ground truth and predictions

    Args:
    y_gold (np.ndarray): the correct ground truth/gold standard labels
    y_prediction (np.ndarray): the predicted labels

    Returns:
    float : the accuracy
    """

    assert len(y_gold) == len(y_prediction)

    matches = y_gold == y_prediction          # returns array of True / False if matching
    correct_predictions = np.sum(matches)     # Counts cases of True

    return correct_predictions/(len(y_gold))

def compute_stats(x, y, classes):

    # Compute the min, max, mean, median and stdev for each feature
    print(x.min(axis=0))
    print(x.max(axis=0))
    print(np.round(x.mean(axis=0), 1))
    print(np.median(x, axis=0).round(1))
    print((x.std(axis=0)).round(1))

    # Compute the min, max, mean, median and stdev for each class
    for i in range(len(classes)):
        z = x[y == 1]  # Index x with the index values of y when == i
        print(z.min(axis=0))
        print(z.max(axis=0))
        print(np.round(z.mean(axis=0), 1))
        print(np.median(z, axis=0).round(1))
        print((z.std(axis=0)).round(1))

def plot_features(x, y, classes):
    # Plotting a scatter plot for two features

    feature_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.show()

    plt.figure()
    plt.scatter(x[:, 2], x[:, 3], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(feature_names[2])
    plt.ylabel(feature_names[3])
    plt.show()

    # Plotting Histograms

    fig, ax = plt.subplots(1, 3)  # plot subfigures in a 1x3 grid

    ax[0].hist(x[y == 0, 2], color='r')
    ax[0].set(title=classes[0])

    ax[1].hist(x[y == 1, 2], color='b')
    ax[1].set(title=classes[1])

    ax[2].hist(x[y == 2, 2], color='g')
    ax[2].set(title=classes[2])

    plt.show()

def run_classifiers(a, b, x_train, x_test, y_train, y_test):
    classifiers = [a, b]
    for classifier in classifiers:
        classifier.fit(x_train, y_train)
        random_predictions = classifier.predict(x_test)
        print(f"{classifier.__class__.__name__} predictions are:")
        print(random_predictions)

        accuracy = compute_accuracy(y_test, random_predictions)
        print(f"{classifier.__class__.__name__} accuracy is: {accuracy}\n")

def main():
    # Import data and sense check dimensions of arrays
    (x, y, classes) = read_dataset("iris.data.txt")
    print(f"Dimensions of feature data: {x.shape}")
    print(f"Dimensions of corresponding class data: {y.shape}")
    print(f"Classes are {classes}")

    # Compute stats for the data and plot
    compute_stats(x, y, classes)
    plot_features(x, y, classes)

    # Pre-process data into test and training data
    seed = 60012  # Set seed for repeatable experiments
    rg = default_rng(seed)
    x_train, x_test, y_train, y_test = split_dataset(x, y, test_proportion=0.2, random_generator=rg)

    # Initiate classifier instances
    random_classifier = RandomClassifier(rg)
    nn_classifier = NearestNeighbourClassifier()

    # Run classifiers and compute accuracy
    run_classifiers(random_classifier, nn_classifier, x_train, x_test, y_train, y_test)

main()






