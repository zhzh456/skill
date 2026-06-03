# EagleMine: Beyond outliers and on to micro-clusters: Vision-guided Anomaly Detection

import spartan as st
import zipfile
from scipy.sparse import csr_matrix

# Run EagleMine as a single model
voctype = "dtmnorm"
mode, mix_comps = 2, 2
eaglemine = st.EagleMine(voctype=voctype, mode=mode, mix_comps=mix_comps)

# Load data for EagleMine
# 2.1 Using the off-the-shelf histogram
inpath = "./inputData/"
infn_histogram = inpath + "histogram.out"
infn_node2hcel = inpath + "node2hcel.out"
infn_hcel2avgfeat = inpath + "hcel2avgfeat.out"

# Extract the eaglemine data zip file
zFile = zipfile.ZipFile("./inputData/eaglemine_data.zip", "r")
for fileM in zFile.namelist(): 
    zFile.extract(fileM, "./inputData")
zFile.close()

# Load histogram data from file
histogram = st.loadFile2Dict(infn_histogram, 2, int, int)
node2hcel = st.loadFile2Dict(infn_node2hcel, 1, int, int)
hcel2avgfeat = st.loadFile2Dict(infn_hcel2avgfeat, 2, int, float)

# 2.2 Raw graph data for the EagleMine
# load graph data
inpath = "./inputData/"
in_data = inpath + "example_graph.tensor"
tensor_data = st.loadTensor(path = in_data, header=None)
stensor = tensor_data.toSTensor(hasvalue=True)
graph = st.Graph(stensor, bipartite=True, weighted=True, modet=2)

# 2.2.1 Extract example features
feature_type = 'outdegree2hubness'  #  "indegree2authority" #
degreeidx, feature = eaglemine.graph2feature(graph, feature_type)

# 2.2.2 Construct histogram based on the above feature
histogram, node2hcel, hcel2avgfeat = eaglemine.feature2histogram(feature, degreeidx, N_bins=80, base=10, mode=mode, verbose=True)

# [Optional] 2.2.3 Save generated histogram data
# note: make sure ./output dir has been manually created
outpath = "./output/"
outs_histogram = outpath + "histogram.out"
outs_node2hcel = outpath + "node2hcel.out"
outs_hcel2avgfeat = outpath + "hcel2avgfeat.out"
eaglemine.save_histogram(outs_histogram, outs_node2hcel, outs_hcel2avgfeat, comments="#", delimiter=",")

# Feed histogram data to EagleMine model
eaglemine.set_histdata(histogram, node2hcel, hcel2avgfeat, weighted_ftidx=0)

# 3. Run the EagleMine model
# note: make sure ./output dir has been manually created
outpath = "./output/"
eaglemine.run(outs=outpath, waterlevel_step=0.2, prune_alpha=0.80, min_pts=20, strictness=3, verbose=True)

# output model information
eaglemine.dump()

# [Optional] 4. Save results
outs_eaglemine = outpath + "eaglemine.out"
outs_leveltree = outpath + "waterleveltree.out"
outs_node2label = outpath + "node2label.out"
outs_hcel2label = outpath + "hcel2label.out"
eaglemine.save(outs_eaglemine, outs_leveltree, outs_node2label, outs_hcel2label, comments="#", delimiter=",")

# [Optional] 5. Result visualization
# Visualize the two-dimensional histogram
infn_histogram = outpath + "histogram.out"  # inpath + "histogram.out" # 
hist_shape, ticks_dims, hist_arr = st.loadHistogram(infn_histogram)
hist_spm = csr_matrix((hist_arr[:, -1], (hist_arr[:, 0], hist_arr[:, 1])), shape=hist_shape, dtype=int)

outfn_hist = outpath + "histogram.png"
x_label, y_label = "Hubness", "Out-degree"  #"Authority", "In-degree"  # "PageRank", "Degree"
hfig = st.histogram_viz(hist_spm, ticks_dims[1], ticks_dims[0], outfn_hist, x_label=x_label, y_label=y_label)

# Visualize clustering result of EagleMine
infn_hcel2label = inpath + "hcel2label.out"  #outpath + "hcel2label.out" # 
hcel2lab = st.loadFile2Dict(infn_hcel2label, 2, int, int)

outfn_hcls = outpath + "hcluster.png"
hcsl_fig = st.clusters_viz(hcel2lab, outfn_hcls)