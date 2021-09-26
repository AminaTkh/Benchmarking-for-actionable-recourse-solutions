from gurobipy import *
UNACTIONABLE = 1e9  # constant declared for features that can't be changed (e.g. Gender, Age)
import math

class UnrecoverableDataPointException(Exception):
    pass


class FlipsetAlgorithm:
    """Find minimal flipset per row of input data with a predicted negative outcome."""
    def __init__(self, clf, costs, data_row, m=None, for_individual=False, min_max=None):
        """Initialise variables needed."""
        self.clf = clf
        self.costs = costs
        self.m = m
        self.data_row = data_row
        self.for_individual = for_individual  # If calculating a single flipset, instead of for entire dataset
        self.min_max = min_max

    def run(self):
        """Build the model, get the new values, and write them to an output file."""
        if self.for_individual:
            self._build_model_from_scratch()
        else:
            self._build_model()

        if self.m.Status == 2:  # Optimal solution found
            self._get_values(True)
        elif self.m.Status == 3:  # Model is infeasible
            self._get_values(False)
        else:
            print(f"Model status is: {self.m.Status}\n")
            print(f"For data row: {self.data_row}\n")

    def _build_model(self):
        """Sets objective of model, to find minimum cost of changes to get a positive outcome."""
        # Ensure unactionable features can't have their values changed
        for i, var in enumerate(self.m.getVars()):
            if self.costs[i] == UNACTIONABLE:  # If this variable is unactionable, set it to a constant value
                var.setAttr("ub", self.data_row[i])
                var.setAttr("lb", self.data_row[i])
        self.m.update()

        index_costs = [self.costs[k] * ((self.data_row[k] - var) * (self.data_row[k] - var))
                       for k, var in enumerate(self.m.getVars())  # Weighted Euclidean Distance
                       ]
        self.m.setObjective(quicksum(index_costs), GRB.MINIMIZE)
        self.m.optimize()

    def _get_values(self, solution_found):
        """Get the flipset values, and the original values, to compare the change."""
        if type(self.data_row) != list:
            self.data_row = self.data_row.tolist()
        self.old_values = self.data_row  # Get original values for datarow

        if not solution_found:
            raise UnrecoverableDataPointException(f"Couldn't find flipset for values: {self.old_values}")

        else:  # If model was feasible, get new values
            self.new_values = [v.x for v in self.m.getVars()]  # New values that will give a positive outcome

            # Get total number of attributes changed, and the total cost of changes
            self.total_cost = 0
            self.total_changed = 0
            for i, v in enumerate(self.m.getVars()):
                self.total_cost += self.costs[i] * (((self.data_row[i] - v.x) ** 2))
                self.total_changed += abs(self.old_values[i] - self.new_values[i])
            self.total_cost = math.sqrt(self.total_cost)

    def _build_model_from_scratch(self):
        """Build gurobi model if not already set (needed for individual flipsets)."""
        assert self.min_max is not None, "For an individual's flipset, must include upper and lower bounds of inputs"
        self.m = Model("mip1")  # Create a new model
        self.m.setParam('OutputFlag', False)  # Suppress output to terminal
        min_vals = [i for i, j in self.min_max]
        max_vals = [j for i, j in self.min_max]

        # Create variables
        for i in range(len(min_vals)):
            self.m.addVar(lb=min_vals[i],
                          ub=max_vals[i],
                          vtype=GRB.INTEGER,
                          name='var' + str(i)
                          )
        self.m.update()

        # Set constraint of positively classified
        coefs = self.clf.coef_[0].tolist()
        classification = quicksum(coefs[k] * var for k, var in enumerate(self.m.getVars())) + self.clf.intercept_
        self.m.addConstr(classification, GRB.GREATER_EQUAL, 0, "c0")
        self.m.update()
        self._build_model()
