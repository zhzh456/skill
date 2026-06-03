# Fraudar_demo

import sys
import spartan as st

# load graph data
tensor = st.loadTensor(path = "inputData/plain_graph_small.zip", col_types = [int, int], sep=' ')

stensor = tensor.toSTensor(hasvalue=False)

# create a anomaly detection model
fd = st.Fraudar(stensor)

# run the model
fd.run(k=2, maxsize=[100, 100])