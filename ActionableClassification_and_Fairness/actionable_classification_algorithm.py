import csv
from statistics import mean
import numpy as np
import pandas as pd
from time import time
from gurobipy import *
from action_fairness_algorithm import ActionFairnessAlgorithm, \
    FailedToImproveDataPointException
from flipset_algorithm import FlipsetAlgorithm, UnrecoverableDataPointException
from scipy.spatial import distance


class ActionableClassificationAlgorithm:
    """Run actionable classification problem as an algorithm object."""

    def __init__(self, costs, clf, clf_type, output_file, x, y, attr_names, sensitive_attr_index,
                 proportion_strictness=None, number_of_blocks=None, run_af_alg=False):
        """Initialise variables needed."""
        self.costs = costs
        self.clf = clf
        self.clf_type = clf_type
        self.output_file = output_file  # None if don't want results written to CSV
        self.x = x
        self.y = y
        self.attr_names = attr_names
        # Sensitive attribute for fairness
        self.sensitive_attr_index = sensitive_attr_index
        self.positive_data_points = []  # Input datapoints that are positively classified
        self.negative_data_points = []  # Input datapoints that are negatively classified
        self.proportion_strictness = proportion_strictness
        self.number_of_blocks = number_of_blocks
        self.run_af_alg = run_af_alg  # Flag for whether to run ActionFairnessAlgorithm
        

    def run(self):
        """Run actionable classification problem with given inputs."""
        self._initial_setup()
        self._calculate_negative_flipsets()
        self._calculate_positive_flipsets()

        if self.output_file is not None and self.output_file != 'german' and self.output_file != 'credit':
            self._write_flipset_results()  # Write results to file only if specified

        if self.sensitive_attr_index is not None:
            # Evaluate Action Fairness for selected sensitive attribute
            self._eval_action_fairness()
            if self.run_af_alg:
                try:
                    self._run_action_fairness()  # Attempt to improve Action Fairness
                except AttributeError as e:  # Added to check new_list not being created
                    print(str(e))
                self._get_modified_data()

        self._remove_temp_vars()  # Added for demo
    def _initial_setup(self):
        """Setup for Optimiser Model."""
        self._calculate_variables()
        self.m = self._init_model()
        self.m = self._build_model_constraints(self.m)

    def _calculate_variables(self):
        """Calculate class variables needed for execution."""
        # Get max and min values per column in input data
        self.max_col_vals = self.x.max().to_list()  # expect x as a df
        self.min_col_vals = self.x.min().to_list()

        # Reformat x and y to numpy arrays
        self.y = np.array(self.y)
        self.x = np.array(self.x)

        # Only considering values where the model would predict them -1
        y_pred = self.clf.predict(self.x)
        # Get index of each predicted negative and positive outcome
        self.negatives = [idx for idx, val in enumerate(y_pred) if val == -1]
        self.positives = [idx for idx, val in enumerate(y_pred) if val == 1]

    def _init_model(self):
        """Initialise gurobipy model with variables and the decision boundary constraint."""
        m = Model("mip1")  # Create a new model
        m.setParam('OutputFlag', False)  # Suppress output to terminal

        # Create variables
        for j in range(self.x.shape[1]):
            m.addVar(lb=self.min_col_vals[j],
                     ub=self.max_col_vals[j],
                     vtype=GRB.INTEGER,
                     name='var' + str(j)
                     )
        m.update()
        return m

    def _build_model_constraints(self, m):
        """Create constraints for the optimisation problem."""
        # Make sure solution will be positively labelled for linear classifiers
        coefs = self.clf.coef_[0].tolist()
        self.classification = quicksum(coefs[k] * var for k, var in enumerate(m.getVars()))\
            + self.clf.intercept_
        m.addConstr(self.classification, GRB.GREATER_EQUAL, 0, "c0")

        m.update()
        return m

    def _calculate_negative_flipsets(self):
        """Calculate minimal set of actions to flip negative outcome to positive."""
        # Initialise storing Actionable Classification and Action Fairness evaluation results
        self.action_values = {}
        self.results_df = pd.DataFrame()

        self._calculate_flipset(self.negatives, negatives=True)

    def _calculate_positive_flipsets(self):
        """Calculate minimal set of actions to flip positive outcome to negative."""
        # Remove old constraint and add new constraint (must be negatively classified, not positively)
        self.m.remove(self.m.getConstrByName('c0'))  # Remove old constraint
        self.m.addConstr(self.classification, GRB.LESS_EQUAL, 0, "c0")
        self.m.update()

        self._calculate_flipset(self.positives, negatives=False)

    def _calculate_flipset(self, dataset_indices, negatives):
        dict_for_avg = {}
        negatives_dict = {}

        """Calculate flipsets for each data point in dataset_indices and store the information."""
        for i in dataset_indices:
        	try:
        		sample = {}
        		start_time = time()
        		flipset = FlipsetAlgorithm(self.clf, self.costs, self.x[i], self.m)
        		flipset.run()
        		time_taken = time() - start_time
        		#print("Time taken was", time_taken)
        		indd = [str(j) for j in range(len(flipset.old_values))]
        		fac_sample = dict(zip(indd, flipset.old_values))
        		fac_sample['y'] = negatives
        		indd = [str(j) for j in range(len(flipset.new_values))]
        		cfe_sample = dict(zip(indd, flipset.new_values))
        		cfe_sample['y'] = not negatives
        		dst = distance.euclidean(flipset.new_values, flipset.old_values)
        		sample['cfe_distance'] = dst
        		sample['cfe_time'] = time_taken
        		sample['cfe_sample'] = cfe_sample
        		sample['fac_sample'] = fac_sample
        		sample['id'] = i
        		dict_for_avg[str(i)] = sample
        		self._update_results(flipset, i)
        		if negatives:
        			negatives_dict[str(i)] = sample
        			self._update_df(flipset, i)
        			if self.sensitive_attr_index is not None:
        				self._update_action_values(flipset, i)
        			
        	except GurobiError as e:
        		print('Error code ' + str(e.errno) + ": " + str(e))
        	except AttributeError as e:
        		print('Encountered an attribute error: ')
        		print(str(e))
        	except UnrecoverableDataPointException as e:
        		print(str(e)) 
        if negatives:
        	#print(negatives_dict)
        	if self.output_file is not None:
        		with open("dictionaries/" + self.output_file + "ACdist.txt", 'w') as filee:
        			print(negatives_dict, file=filee)
		#for first part of project
        #with open('allpoint.txt', 'w') as f:
         #   print(dict_for_avg, file=f)

    def _update_df(self, flipset, row_index):
        """Update results dataframe with values from a particular flipset."""
        # Get index of input data for top level of dataframe
        data_row_index = [str(row_index)] * len(self.costs)

        # Pad smaller rows with NaNs
        padded_change_number = self._pad_df(flipset.total_changed)
        padded_cost = self._pad_df(flipset.total_cost)

        df_row_data = [data_row_index,
                       flipset.old_values,
                       flipset.new_values,
                       padded_change_number,
                       padded_cost
                       ]

        column_names = ["row", "old_values",
                        "new_values", "change_number", "total_cost"]
        transposed_data = zip(*df_row_data)  # Transpose data
        df_row = pd.DataFrame(transposed_data, columns=column_names)
        # Add result of this data row to the results_df of all results
        self.results_df = self.results_df.append(df_row)

    def _pad_df(self, value):
        """Pads list of a single element with NaNs, for the results dataframe."""
        return [value] + ([np.nan] * len(self.costs))

    def _update_results(self, flipset, data_row_index):
        """Update list containing all DataPoint objects, with results."""
        data_prediction = self.clf.predict_proba([flipset.old_values])[
            0]  # Used for ranking in action fairness
        data_point_vals = (
            data_row_index, flipset.old_values, flipset.new_values)
        if data_prediction[1] >= data_prediction[0]:  # Positively classified
            self.positive_data_points.append(
                DataPoint(*data_point_vals, data_prediction[1]))
            #print("POS", data_point_vals,  data_prediction[1])
        else:  # Negatively classified
            self.negative_data_points.append(
                DataPoint(*data_point_vals, data_prediction[0]))
            #print("NEG", data_point_vals,  data_prediction[0])

    def _update_action_values(self, flipset, data_row_index):
        """Update values for evaluating Action Fairness for a particular flipset."""
        sensitive_val = self.x[data_row_index][self.sensitive_attr_index]  # Get value of sensitive attribute
        # Get costs of all flipsets for that value
        costs_for_val = self.action_values.get(sensitive_val, [])
        # Add cost of this flipset to the list
        costs_for_val.append(flipset.total_cost)
        # Add appended list back to dict
        self.action_values.update({sensitive_val: costs_for_val})

    def _eval_action_fairness(self):
        """Use total cost of changes per sensitive attribute to evaluate action fairness of classifier."""
        mean_actions = [
            mean(val) for val in self.action_values.values()]  # Mean group costs
        self.mean_group_costs = {
            key: mean(val) for key, val in self.action_values.items()}
        # Find ratio between mean cost of max and min
        self.unfairness_ratio = min(mean_actions) / max(mean_actions)

    def _write_flipset_results(self):
        """Write original and modified values for each negative outcome to a file."""
        self._write_attr_names()  # Write names of the attribute at the top of the file

        # Get each unique value of row index
        for idx in self.results_df.row.unique():
            # Get that row's data as dataframe
            row_df = self.results_df[self.results_df.row == idx]

            # Get list of that row's old values and new values
            old_values, new_values = row_df.old_values.to_list(), row_df.new_values.to_list()
            # Get values for total changed and total cost of changes
            total_changed, total_cost = row_df.iloc[0].change_number, row_df.iloc[0].total_cost

            # Write values to output file
            # Append as is done per negative outcome
            with open(self.output_file, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["old values:"] + old_values)
                writer.writerow(["new values:"] + new_values)
                writer.writerow(["total number of value changes:"] + [total_changed] +
                                ["total cost of changes:"] + [total_cost])
                writer.writerow(["\n"])

    def _write_attr_names(self):
        """Write the names of each attribute at the top of the output file, for reference."""
        with open(self.output_file, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([""] + self.attr_names)  # Start with blank cell

    def _run_action_fairness(self):
        """Check Action Fairness of classifier, and return modified rows to ensure Action Fairness."""
        run_state = (pd.DataFrame(self.x), self.sensitive_attr_index,
                     self.number_of_blocks, self.proportion_strictness)
        try:
            self.NegativeActionFairness = ActionFairnessAlgorithm(clf=self.clf,
                                                                  costs=self.costs,
                                                                  data_points=self.negative_data_points,
                                                                  direction=0  # Shifting negative datapoints
                                                                  )
            self.PositiveActionFairness = ActionFairnessAlgorithm(clf=self.clf,
                                                                  costs=self.costs,
                                                                  data_points=self.positive_data_points,
                                                                  direction=1  # Shifting positive datapoints
                                                                  )
            # Run data perturbation on negatively classified data
            self.NegativeActionFairness.run(*run_state)
            # Run data perturbation on positively classified data
            self.PositiveActionFairness.run(*run_state)
            self.mean_group_costs_after_af = self.NegativeActionFairness.mean_group_costs_after
        except FailedToImproveDataPointException:
            print("Incomplete Ranked List Produced")

        except RecursionError:
            print("Reached recursion error")
            exit(1)

    def _get_modified_data(self):
        """Construct DataFrame of modified fairer data."""
        positive_data = [
            datapoint for block in self.PositiveActionFairness.new_list for datapoint in block]
        positive_data = [point.current_values for point in positive_data]
        negative_data = [
            datapoint for block in self.NegativeActionFairness.new_list for datapoint in block]
        negative_data = [point.current_values for point in negative_data]
        self.modified_y = [1 for i in range(
            len(positive_data))] + [-1 for j in range(len(negative_data))]
        self.modified_x = pd.DataFrame(
            positive_data + negative_data, columns=self.attr_names)

    def _remove_temp_vars(self):
        """Tidy variable space by removing superfluous variables."""
        del self.m
        del self.action_values
        del self.attr_names
        del self.max_col_vals
        del self.min_col_vals
        del self.negatives
        del self.positives
        del self.output_file
        del self.run_af_alg
        del self.clf_type


# Currently used as a struct, no associated methods
class DataPoint:
    """Store data points and their values as their own objects."""
    def __init__(self, data_row_index, data_values, flipset_values, score):
        """Initialise data point with current and flipset values, and its prediction probability value."""
        self.index = data_row_index  # Act as id
        self.current_values = data_values
        self.flipset_values = flipset_values
        self.attr_value = None  # Not set at initialisation
        self.score = score# max clf.predict_proba([current_values])
      #  self.flipset_cost = flipset_cost
