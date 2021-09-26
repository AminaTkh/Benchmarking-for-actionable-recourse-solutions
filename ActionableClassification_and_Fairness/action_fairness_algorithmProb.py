from copy import copy
import pandas as pd
import numpy as np
from bisect import bisect_left
from math import ceil
from statistics import mean


class FailedToImproveDataPointException(Exception):  # Data point couldn't be improved past previous data point
    pass                                             # (Means previous data point is beyond decision boundary)


class FailedToIntroduceFairness(Exception):  # Couldn't construct list in a way that is fair (in terms of proportions)
    pass


class ActionFairnessAlgorithm:
    """Perform action fairness on a given attribute, modifying data points so that there is a fair ranked list wrt
        each different value for that attribute, according to the global proportions with an threshold of acceptance

        (e.g. rank data points based on prediction probability, then check if proportions of Male / Female going down
        that list are similar enough to proportions in whole dataset, if not then improve values of some individuals,
        according to their action sets, until a fair proportion is reached)."""
    def __init__(self, clf, costs, data_points, direction=0, global_proportions=None):
        """Initialise Algorithm with necessary starting variables."""
        self.clf = clf
        self.costs = costs
        self.data_points = data_points
        self.global_proportions = global_proportions  # Allow proportions to be set (so can use sample of overall data)
        self.direction = direction  # 0 if improving negative points, 1 if 'worsening positive points'
        # Vars below are populated later
        self.initial_data = None
        self.final_data = None
        self.number_of_blocks = None
        self.proportion_strictness = None
        self.initially_unfair = None
        self.over_represented_subgroup = None

        # Check initialisation values
        assert (direction == 0 or direction == 1), "Direction improving datapoints should be 0 (improve negatives) or" \
                                                   "1 (worsen positives)"
                                               

    def run(self, data_df, attr_col_index, number_of_blocks, proportion_strictness, over_represented_subgroup=None):
        """Run the algorithm."""
        # Store running parameters
        self.number_of_blocks = number_of_blocks
        self.proportion_strictness = proportion_strictness
        self.over_represented_subgroup = over_represented_subgroup

        # Initialise ranked list
        self._get_attr_values(attr_col_index)
        self._create_ranked_list()
        #initialdf = self._store_data_point_values_as_df(self.ranked_list)
        #print(len(initialdf))
        #initialdf.to_csv('initialrankings/' +  + 'initial' + '.csv')
        if self.global_proportions is None:  # If proportions not been set, find them from data points given
            self._get_global_proportions(data_df, attr_col_index)
        self.initialise_group_totals()
        self._record_initial_blocks()

        # Run block-based data perturbation method
        self._block_based_fairness()

        # Store results
        self._record_final_blocks()
        self._eval_mean_group_costs()  # Get new mean costs per subgroup after the algorithm has taken place
        self._remove_temp_vars()

    def _get_attr_values(self, attr_col_index):
        """Store each DataPoint's value for the sensitive attribute being checked (e.g. the Gender value)."""
        for i, point in enumerate(self.data_points):
            self.data_points[i].attr_value = point.current_values[attr_col_index]

    def _create_ranked_list(self):
        """Create ranking of data points, to check against for fairness."""
        # For now, just get negatively classified data and rank it in accordance with predict_proba values
        # (Lower value is higher ranked)
        self.ranked_list = sorted(self.data_points, key=lambda x: abs(x.score))
        return self.ranked_list
    def _get_final_data(self):
    	return self.final_data
    def _get_initial_data(self):
    	return self.initial_data
    def get_data_points(self):
    	return len(self.data_points)

    def _record_initial_blocks(self):
        """Store how the initial data (in blocks) looks, and calculate which blocks were initially unfair."""
        self.initial_data = []

        # Get indices of block splits
        ranked_list_splits = np.array_split(range(len(self.ranked_list)), self.number_of_blocks)
        # Get next block
        for block_index, indices in enumerate(ranked_list_splits):
            block = self.ranked_list[indices[0]: indices[-1]]

            # Get blocks initially unfair
            self._get_initial_block_totals(block)
            self.initially_unfair = []
            if self._check_block_proportion_fairness(len(block)):  # If overrepresented values exist, block is unfair
                self.initially_unfair.append(block_index)

            # Record values for each data point in the block
            self.initial_data.append(self._store_data_point_values_as_df(block))

    def _record_final_blocks(self):
        """Store how the final data looks."""
        self.final_data = []
        for block in self.new_list:
            self.final_data.append(self._store_data_point_values_as_df(block))

    @staticmethod
    def _store_data_point_values_as_df(data_point_list):
        """Create copy of data point values, for comparison between start and end of algorithm."""
        data_list = []
        for point in data_point_list:
            data_list.append(
                {"datapoint": point.index,
                 "attr_val": point.attr_value,
                 "data_values": copy(point.current_values),
                 "flipset_values": point.flipset_values,
                 "score": copy(point.score)

                 })  # Get methods as reference value passed instead of actual value.
        data_df = pd.DataFrame(data_list)
        return data_df

    def _get_global_proportions(self, df, attr_col_index):
        """Calculate proportions of sensitive attribute in the dataset as a whole, to measure against."""
        self.global_proportions = df.iloc[:, attr_col_index].value_counts(normalize=True).to_dict()

    def initialise_group_totals(self):
        """Retrieve all possible values (all groups) the sensitive attribute could be."""
        # Initialise current totals to 0 for all possible attribute values
        self._block_totals = {attr_val: 0 for attr_val in self.global_proportions.keys()}

    def _choose_attr_to_improve(self, data_point, prediction_goal):  # Pass in DataPoint or store as class var
        """Improve attribute of the data row with the least cost."""
        # Assumes UNACTIONABLE attributes won't be changed in action set
        data_values = data_point.current_values
        data_changes = [act_val - data_val for (act_val, data_val) in zip(data_point.flipset_values, data_values)]

        # Try to only improve a single attribute
        cost_index_ordered = np.array(self.costs[:]).argsort()  # Order of attributes to attempt

        def _improve_data_values(clf, reset_values):
            """Improve values of a row of data so its closer to the decision boundary than the target value."""
            improved_values = data_values[:]
            for i in cost_index_ordered:

                # If only attempting to change a single attribute, then reset the values for each attribute
                if reset_values:
                    improved_values = data_values[:]
                for j in range(int(abs(data_changes[i]))):

                    if data_changes[i] > 0:
                        improved_values[i] += 1
                    else:
                        improved_values[i] -= 1
                    #  If data point moved enough, return
                    if clf.predict_proba([improved_values])[0][self.direction] < prediction_goal:
                        return improved_values

            return None  # No improvement is sufficient

        # Try to change just a single attribute
        new_values = _improve_data_values(self.clf, True)
        if new_values is not None:
            return new_values

        # If didn't work, allow multiple attributes to be changed
        new_values = _improve_data_values(self.clf, False)
        if new_values is None:
            raise FailedToImproveDataPointException

        # Try to decrease cost of values changed
        def _minimise_data_value_changes(clf):
            """Remove excess changes to the data row that aren't required to still be over the decision boundary."""
            minimised_values = new_values[:]
            for i in reversed(cost_index_ordered):

                for j in range(int(abs(data_changes[i]))):

                    attempted_change = minimised_values[:]
                    # Attempt to reduce changes by 1
                    if data_changes[i] > 0:
                        attempted_change[i] -= 1
                    else:
                        attempted_change[i] += 1

                    if clf.predict_proba([attempted_change])[0][self.direction] >= prediction_goal:
                        # If minimisation for this attribute failed, stop trying to reduce this attribute
                        break

                    else:  # Else keep changes
                        minimised_values = attempted_change

            return minimised_values

        return _minimise_data_value_changes(self.clf)

    def _block_based_fairness(self):
        """Improve action fairness by improving proportions of each block of data points."""
        block_count = 0
        self.new_list = []
        self.fair_blocks = []
        self.unfair_blocks = []

        # Get indices of block splits
        ranked_list_splits = np.array_split(range(len(self.ranked_list)), self.number_of_blocks)

        # Get next block
        for indices in ranked_list_splits:
            block_size = len(indices)
            block = self.ranked_list[:block_size]
            self.ranked_list = self.ranked_list[block_size:]

            block, is_fair = self._process_next_block(block)
            self.new_list.append(block)
            if is_fair:
                self.fair_blocks.append(block_count)
            else:
                self.unfair_blocks.append(block_count)
            print(f"Finished block {block_count} out of {self.number_of_blocks}")
            block_count += 1

    def _process_next_block(self, block):
        """Try to get the next block of n elements with fair proportions."""
        # Get initial attribute totals for block
        self._get_initial_block_totals(block)
        return self._make_block_fair(block)

    def _make_block_fair(self, block):
        """Attempt to insert modified data points into range of block, to make fair proportions."""
        # Check proportion of block
        over_represented_values = self._check_block_proportion_fairness(len(block))  # length of block varies

        # Proportions of block unfair (/ proportion of specified subgroup is unfair)
        if (self.over_represented_subgroup is None and over_represented_values) \
                or self.over_represented_subgroup in over_represented_values:
            return self._find_datapoint_to_improve(block, over_represented_values)
        else:  # Proportions of block acceptable
            return block, True  # Block is fair, get next block

    def _find_datapoint_to_improve(self, block, over_represented_values):
        """Find next data point with acceptable value that can be improved within the range of the block."""
        # Get range of prediction_proba values in block
        lb = block[0].score
        # ub_datapoint = round(len(block)*0.8)  # The data point 20% up the block
        # ub = block[ub_datapoint].predict_prob_value  # New data point must have predict_prob value in top 80% of dps

        ub = block[-1].score
        ub = ub - (ub-lb)/5  # New data point must have predict_prob value in top 80% of range

        # Find next datapoint with acceptable sensitive attribute value
        for i, dp in enumerate(self.ranked_list):
            if dp.attr_value not in over_represented_values:
                # Attempt to modify datapoint to within accepted prediction range
                improved_data_row, new_prediction = self._improve_data_point_to_within_range(dp, ub)

                if new_prediction > lb:
                    # Update data point with new values
                    dp.current_values = improved_data_row
                    dp.score = new_prediction

                    # Remove data point from old list
                    del self.ranked_list[i]
                    return self._insert_into_block(block, dp)

                # Else find next acceptable data point to try to improve

        # Run out of data points, so block cannot be made fair. Label it as such(?) and move on to next block
        return block, False  # return some flag that wasn't fair

    def _insert_into_block(self, block, datapoint):
        """Insert improved datapoint into correct position in block, remove last element in block."""
        # Get place in block to insert improved data point, insert it into block
        index = bisect_left([x.score for x in block], datapoint.score)
        block.insert(index, datapoint)

        # Pop last element out of block, insert as first element in old ranked list
        self.ranked_list.insert(0, block.pop())

        # Update sensitive totals for block
        self._update_block_totals(datapoint.attr_value, self.ranked_list[0].attr_value)

        # Check if block has been made fair. If not, continue process
        return self._make_block_fair(block)

    def _get_initial_block_totals(self, block):
        """Calculate initial totals in the block of each value for sensitive attribute."""
        # Initialise block totals to be 0 for each possible attribute value
        self._block_totals = {k: 0 for k in self._block_totals.keys()}
        for i in block:
            self._block_totals[i.attr_value] += 1  # Increment number of datapoints with that value by one

    def _check_block_proportion_fairness(self, n):
        """Return which values of the sensitive attribute are currently overrepresented in the block."""
        # Divide by number in block for proportions.
        block_proportions = {key: val / n for key, val in self._block_totals.items()}

        # Get proportions for each attribute value
        over_represented = []
        for attr_value, proportion in self.global_proportions.items():
            proportion_with_attr_value = block_proportions.get(attr_value)

            # Over represented if current_proportions > allowed_threshold
            if proportion_with_attr_value > proportion + (1 - proportion) / self.proportion_strictness:
                over_represented.append(attr_value)

        return over_represented  # Returns list of indices of classes over represented AS STRING
        # (So, select next object in ranklist that isn't one of those classes)

    def _improve_data_point_to_within_range(self, data_point, ub):
        """Attempt to improve data point so its prediction prob value is within the accepted range."""
        # same as improve_data_point as before, though this time prediction_goal = ub
        new_data_row = data_point.current_values
        while self.clf.predict_proba([new_data_row])[0][self.direction] >= ub:
            new_data_row = self._choose_attr_to_improve(data_point, ub)

        new_prediction = self.clf.predict_proba([new_data_row])[0][self.direction]
        return new_data_row, new_prediction

    def _update_block_totals(self, added_value, removed_value):
        """Update totals of sensitive attribute values in block."""
        self._block_totals[added_value] += 1  # Increment added attribute value
        self._block_totals[removed_value] -= 1  # Decrement removed attribute value

    def _eval_mean_group_costs(self):
        """Calculate mean cost for each subgroup after alg run."""
        # Combine list of dataframes into one dataframe
        initial = pd.concat(self.initial_data)
        final = pd.concat(self.final_data)

        def calculate_costs_per_group(df):
            """Calculate the costs of flipsets per subgroup on given dataframe."""
            costs_per_group = {}

            for idx, row in df.iterrows():
                value_changes = np.subtract(np.array(row.data_values), np.array(row.flipset_values))  # Get attr changes
                value_changes = np.power(value_changes, 2)  # Changes ** 2
                value_changes = np.multiply(self.costs, value_changes)  # Weighted by costs
                value_changes = np.sum(value_changes)  # Get total cost (sum)

                # Add to dict (of value sensitive attr val)
                current_costs = costs_per_group.get(row.attr_val, [])
                current_costs.append(value_changes)
                costs_per_group[row.attr_val] = current_costs

            return costs_per_group

        costs_per_group_before = calculate_costs_per_group(initial)
        costs_per_group_after = calculate_costs_per_group(final)

        # Calculate mean cost per dict slot
        self.mean_group_costs_before = {group: mean(costs) for group, costs in costs_per_group_before.items()}
        self.mean_group_costs_after = {group: mean(
            costs) for group, costs in costs_per_group_after.items()}

        return self.mean_group_costs_before, self.mean_group_costs_after

    def _remove_temp_vars(self):
        """Tidy variable space by removing superfluous variables."""
        del self._block_totals
        del self.ranked_list
