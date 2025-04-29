import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as Bayesianmodel
from pgmpy.inference import VariableElimination 
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
heartdisease=pd.dataframe({
    'age':[37,91,62,45,56],
    'chol':[150,160,180,200,250],
    'fbs':[0,1,0,1,0],
    'restecg':[0,1,1,0,1],
    'thalach':[160,190,200,210,220],
    'target':[0,1,0,1,0]
})
heartdisease=heartdisease.replace('?',np.nan)
model=Bayesianmodel([
    ('age','fbs'),
    ('fbs','target'),
    ('target','restecg'),
    ('target','thalach'),
    ('target','chol')
])
model.fit(heartdisease,estimator=MaximumLikelihoodEstimator)
heartdisease_infer=VariableElimination(model)
q=heartdisease_infer.querey(variables=['target'],evidence={'age':37})
print(q)
