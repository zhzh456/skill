# EigenPulse: Detecting Surges in Large Streaming Graphs with Row Augmentation

import spartan as st
import gzip

# Open the input file
f = gzip.open("./inputData/test_beer.tensor.gz")


# Create tensor stream
tensor_stream = st.TensorStream(f, col_idx = [0,1,2,3], col_types=[int,int,int,int], sep=',', mappers={}, hasvalue=True)

# Run EigenPulse as a single model
param_dict={'window':20, 'stride':10, 'l':20, 'b':10, 'item_idx':1, 'ts_idx':2}
eigenpulse = st.EigenPulse(tensor_stream, **param_dict)

# Run the model
res, densities = eigenpulse.run()
# Draw the densities
st.drawEigenPulse(densities, figpath='images/eigenDensities.png')

# Run EigenPulse from anomaly detection task
f = gzip.open("./inputData/test_beer.tensor.gz")
tensor_stream = st.TensorStream(f, col_idx = [0,1,2,3], col_types=[int,int,int,int], sep=',', mappers={}, hasvalue=True)

ad_model = st.AnomalyDetection.create(tensor_stream, st.ADPolicy.EigenPulse, 'eigenpulse', **param_dict)
res, densities = ad_model.run()