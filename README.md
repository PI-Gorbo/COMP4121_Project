# COMP4121_Project 
This repo contains all the code and the test data used in my comp 4121 project on implementations of classification algorithms. I implemented 4 models, K-Nearest Neighbours, Decision Tree and Naive Bayes. 

### Dependencies

This program will work with any version of Python 3.X and uses Pandas, Numpy and Sklearn. (However it does not use the SKlean models.)

### Running the Algorithms

The program has an entrypoint at `./run.py`. The top of the file is shown in the snippet below;

```
if __name__ == '__main__':
    runner = LogisticRegression()
    runner.run(dataset=Dataset.Cancer)

    # runner = KNN()
    # runner.run(dataset=Dataset.Cancer)

    # runner = NaiveBayes()
    # runner.run(dataset=Dataset.Cancer)

    # runner = DecisionTree()
    # runner.run(dataset=Dataset.Cancer)
```
Simply comment and uncomment different models, and run them with `Python run.py`. 

### Fitting to other datasets.

In the `Models/Helpers.py` file, it is possible to import other datasets that are added to the directory. A Enum called `StdDistributionType` provides two options, Labeled or RealValued. For each feature in the dataset, label it one of these features and and split it  `ms.test_train_split(...)`. Use the `cancerData()` method as a template.
