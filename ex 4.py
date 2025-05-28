from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

model = BayesianModel([('Rain', 'Traffic'), ('Accident', 'Traffic')])

from pgmpy.factors.discrete import TabularCPD

cpd_rain = TabularCPD('Rain', 2, [[0.8], [0.2]])
cpd_accident = TabularCPD('Accident', 2, [[0.6], [0.4]])
cpd_traffic = TabularCPD('Traffic', 2, [[0.9, 0.6, 0.7, 0.1],
                                        [0.1, 0.4, 0.3, 0.9]],
                         evidence=['Rain', 'Accident'], evidence_card=[2, 2])

model.add_cpds(cpd_rain, cpd_accident, cpd_traffic)

inference = VariableElimination(model)
print(inference.query(variables=['Traffic'], evidence={'Rain': 1}))
