from anfis_model import ANFIS
import membershipfunction
import numpy as np
import pandas as pd

ts = pd.read_csv("hp_train.csv").sort_values(by=['SalePrice'])
X = ts[['BedroomAbvGr', 'OverallQual','OverallCond']].to_numpy() # 3 Inputs
Y = ts['SalePrice'].to_numpy() # Output


mf = [] # Your MFs here

# Updating the model with MFs
mfc = membershipfunction.MemFuncs(mf)

# Creating the ANFIS Model Object
anf = ANFIS(X, Y, mfc)

# Plot the MFs pre-training
anf.plotMF(X[:,0],0)
anf.plotMF(X[:,1],1)
anf.plotMF(X[:,2],2)

# Fitting the ANFIS Model
anf.trainHybridJangOffLine(epochs=20, k=0.01) # Modify k learning rate here

# Print RMSE
rmse = np.sqrt((np.square(np.subtract(anf.fittedValues, anf.Y)).mean()))
print('RMSE:', rmse)

# Plotting Model performance
anf.plotErrors()
anf.plotResults()

# Plot the MFs post-training
anf.plotMF(X[:,0],0)
anf.plotMF(X[:,1],1)
anf.plotMF(X[:,2],2)

