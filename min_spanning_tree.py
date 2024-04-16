import cv2
import numpy as np
import networkx as nx

def compute_edge_weights(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_weights = np.sqrt(grad_x**2 + grad_y**2)
    return edge_weights

def build_graph(image):
    rows, cols = image.shape
    G = nx.grid_2d_graph(rows, cols)
    edge_weights = compute_edge_weights(image)
    for (u, v) in G.edges():
        G[u][v]['weight'] = edge_weights[u[0], u[1]]
    return G

def compute_similarity(pixel1, pixel2, graph, sigma=0.1):
    try:
        shortest_path = nx.shortest_path(graph, source=pixel1, target=pixel2, weight='weight')
        sum_edge_weights = sum(graph[shortest_path[i]][shortest_path[i+1]]['weight'] for i in range(len(shortest_path)-1))
        similarity = np.exp(-sum_edge_weights / sigma)
    except nx.NetworkXNoPath:
        similarity = 0.0
    return similarity

def cost_aggregation_mst(image, mst, graph, sigma=0.1):
    rows, cols = image.shape
    aggregated_costs = np.zeros_like(image, dtype=np.float32)
    
    #leaf to root
    for node in nx.dfs_postorder_nodes(mst):
        parent = next(mst.neighbors(node), None)
        if parent is not None:
            similarity = compute_similarity(tuple(parent), tuple(node), graph, sigma=sigma)
            aggregated_costs[node] = aggregated_costs[parent] + similarity * aggregated_costs[node]
    
    #root to leaf
    for node in nx.dfs_preorder_nodes(mst):
        for child in mst.neighbors(node):
            similarity = compute_similarity(tuple(node), tuple(child), graph, sigma=sigma)
            aggregated_costs[child] = aggregated_costs[child] + similarity * aggregated_costs[node]
    
    return aggregated_costs

def build_mst(image):
    rows, cols = image.shape
    G = nx.grid_2d_graph(rows, cols)
    edge_weights = compute_edge_weights(image)
    for (u, v) in G.edges():
        G[u][v]['weight'] = edge_weights[u[0], u[1]]
    mst = nx.minimum_spanning_tree(G)
    return mst

def generate_disparity_map(aggregated_costs_left, aggregated_costs_right):
    disparity_map = np.zeros_like(aggregated_costs_left, dtype=np.float32)
    return disparity_map

left_image = cv2.imread('/home/tharun/Data_extended/Baby1/view1.png', 0)
right_image = cv2.imread('/home/tharun/Data_extended/Baby1/view5.png', 0)

mst_left = build_mst(left_image)
mst_right = build_mst(right_image)

graph_left = build_graph(left_image)
graph_right = build_graph(right_image)

aggregated_costs_left = cost_aggregation_mst(left_image, mst_left, graph_left)
aggregated_costs_right = cost_aggregation_mst(right_image, mst_right, graph_right)
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=182, blockSize=13)

disparity_map = generate_disparity_map(aggregated_costs_left, aggregated_costs_right)
disparity_map= stereo.compute(left_image, right_image)

# Post-processing
disparity_map= cv2.medianBlur(disparity_map, 5)  
disparity_map= cv2.morphologyEx(disparity_map, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))  
disparity_map= cv2.GaussianBlur(disparity_map, (5, 5), 0)  

disp_min = disparity_map.min()
disp_max = disparity_map.max()
disparity_normalized = ((disparity_map - disp_min) / (disp_max)) * 255.0
disparity_normalized = np.uint8(disparity_normalized)

cv2.imshow('Disparity Map', disparity_normalized)
cv2.imwrite('spanning_tree.png',disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
