import pandas as pd
from actionable_classification.actionable_classification import actionable_classification, UNACTIONABLE
from testing.chart import german_chart
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from statistics import mean

dataset_filename = r"german2_unfair.csv"
df = pd.read_csv(dataset_filename)

# Train the classifier
y = df.iloc[:, 0]  # classification is first column
X = df.iloc[:, 1:]

# Get costs from excel file
costs = pd.read_excel(r"input_costs_german.xlsx")
costs = costs.iloc[:X.shape[1], 1].fillna(UNACTIONABLE).to_list()  # Only get costs, replace NaNs with UNACTIONABLE

# Set AF parameters
sensitive_attr_index = 0  # Married attribute
proportion_strictness = 6
number_of_blocks = 25

fairness_ratio_org = []
fairness_ratio_retrained = []

accuracy_org = []
accuracy_retrained = []
for t in range(20):
    # Separate into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train, y_train)

    # Run Actionable Classification with Action Fairness
    AC = actionable_classification((X_train, y_train),
                                   costs,
                                   output_file=None,
                                   clf=clf,
                                   clf_type='LR',
                                   action_fairness=True,
                                   sensitive_attr_index=sensitive_attr_index,
                                   proportional_strictness=proportion_strictness,
                                   number_of_blocks=number_of_blocks
                                   )

    # Get average subgroup costs
    print(f"\nMean cost in initial data\n"
          f"Male: {AC.mean_group_costs[1]}, Female: {AC.mean_group_costs[2]}\n")

    # Show work of ActionFairness
    # plt1 = german_chart(AC.NegativeActionFairness.initial_data, AC.NegativeActionFairness,
    #                     "Before Fairness Improvements"
    #                     )
    # plt2 = german_chart(AC.NegativeActionFairness.final_data, AC.NegativeActionFairness,
    #                     "After Fairness Improvements"
    #                     )
    # plt1.show()
    # plt2.show()

    # Retrain classifier with new data
    retrained_clf = LogisticRegression(solver='liblinear')
    retrained_clf.fit(AC.modified_x, AC.modified_y)

    # Record accuracy of original & retrained classifiers
    accuracy_org.append(accuracy_score(y_test, clf.predict(X_test)))
    accuracy_retrained.append(accuracy_score(y_test, retrained_clf.predict(X_test)))

    # Run AC again, on testing data with retrained classifier
    AC2 = actionable_classification((X_test, y_test),
                                    costs,
                                    output_file=None,
                                    clf=retrained_clf,
                                    clf_type='LR',
                                    action_fairness=True,
                                    sensitive_attr_index=sensitive_attr_index,
                                    proportional_strictness=proportion_strictness,
                                    number_of_blocks=number_of_blocks
                                    )

    # Run AC again, on testing data with original classifier
    AC3 = actionable_classification((X_test, y_test),
                                    costs,
                                    output_file=None,
                                    clf=clf,
                                    clf_type='LR',
                                    action_fairness=True,
                                    sensitive_attr_index=sensitive_attr_index,
                                    proportional_strictness=proportion_strictness,
                                    number_of_blocks=number_of_blocks
                                    )
    # Record fairness ratio for test data with both original and retrained classifiers
    fairness_ratio_retrained.append(AC2.unfairness_ratio)
    fairness_ratio_org.append(AC3.unfairness_ratio)

print(f"Fairness ratio with original classifier was {mean(fairness_ratio_org)}.\n"
      f"Fairness ratio with retrained classifier was {mean(fairness_ratio_retrained)}")

print(f"Accuracy of original classifier was {mean(accuracy_org)}.\n"
      f"Accuracy of retrained classifier was {mean(accuracy_retrained)}")

print("Finished!")
