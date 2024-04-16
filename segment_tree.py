import cv2
import numpy as np
import networkx as nx

from collections import defaultdict

class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight

def compare_edges(e1, e2):
    return e1.weight <= e2.weight

def merge_subtrees(trees, tree_members, u, v):
    u_tree = -1
    v_tree = -1
    for i, tree in enumerate(trees):
        if u in tree:
            u_tree = i
        if v in tree:
            v_tree = i

    if u_tree != v_tree:
        trees[u_tree].update(trees[v_tree])
        tree_members[u_tree].update(tree_members[v_tree])
        del trees[v_tree]
        del tree_members[v_tree]

def construct_segment_tree(edges, n, k):
    # Initialize phase (one for each node)

    trees = [set([i]) for i in range(n)]
    edge_existence = [set() for _ in range(n)]

    # Grouping phase
    selected_edges = []
    for edge in edges:
        u, v = edge.u, edge.v
        if len(trees[u]) != len(trees[v]) and edge.weight <= min(
            k + max(trees[u], default=-float('inf')),
            k + max(trees[v], default=-float('inf')),
        ):
            merge_subtrees(trees, edge_existence, u, v)
            selected_edges.append(edge)
            edge_existence[u].add((v, edge.weight))
            edge_existence[v].add((u, edge.weight))

    # Linking phase (remove grouped edges and connect remaining)
    remaining_edges = []
    for edge in edges:
        u, v = edge.u, edge.v
        if (v, edge.weight) not in edge_existence[u] and (u, edge.weight) not in edge_existence[v]:
            remaining_edges.append(edge)

    for edge in remaining_edges:
        u, v = edge.u, edge.v
        if len(trees[u]) != len(trees[v]):
            merge_subtrees(trees, edge_existence, u, v)
            selected_edges.append(edge)
            if len(trees) == 1:
                break

    return selected_edges

def compute_similarity(pixel1, pixel2, segment_tree, sigma=0.1):
    try:
        path = nx.shortest_path(segment_tree, source=pixel1, target=pixel2)
        sum_edge_weights = sum(segment_tree[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        similarity = np.exp(-sum_edge_weights / sigma)
    except nx.NetworkXNoPath:
        similarity = 0.0
    return similarity


def cost_aggregation_segment_tree(image, segment_tree, sigma=0.1):
    rows, cols = image.shape
    aggregated_costs = np.zeros_like(image, dtype=np.float32)

    for edge in segment_tree:
        u, v = edge.u, edge.v
        similarity = compute_similarity(image[u], image[v], segment_tree, sigma=sigma)
        aggregated_costs[v // cols, v % cols] += similarity * aggregated_costs[u // cols, u % cols]

    return aggregated_costs

def generate_disparity_map(aggregated_costs_left, aggregated_costs_right):
    disparity_map = np.zeros_like(aggregated_costs_left, dtype=np.float32)
    return disparity_map

left_image = cv2.imread('/home/tharun/Data_extended/Baby1/view1.png', 0)
right_image = cv2.imread('/home/tharun/Data_extended/Baby1/view5.png', 0)

def compute_edges(image):
    edges = []
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                edges.append(Edge(i * cols + j, i * cols + j + 1, abs(int(image[i, j]) - int(image[i, j + 1]))))
            if i + 1 < rows:
                edges.append(Edge(i * cols + j, (i + 1) * cols + j, abs(int(image[i, j]) - int(image[i + 1, j]))))
    return edges

n, m = left_image.shape  
edges_left = compute_edges(left_image)
edges_right = compute_edges(right_image)

k = 1200  # Constant parameter ( same value as given in the base paper)

segment_tree_left = construct_segment_tree(edges_left, n * m, k)
segment_tree_right = construct_segment_tree(edges_right, n * m, k)

aggregated_costs_left = cost_aggregation_segment_tree(left_image, segment_tree_left)
aggregated_costs_right = cost_aggregation_segment_tree(right_image, segment_tree_right)

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=21)
disparity_map = stereo.compute(left_image, right_image)

# Post-processing
disparity_map= cv2.medianBlur(disparity_map, 5)  
disparity_map= cv2.morphologyEx(disparity_map, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))  
disparity_map= cv2.GaussianBlur(disparity_map, (5, 5), 0)  

disp_min = disparity_map.min()
disp_max = disparity_map.max()
disparity_normalized = ((disparity_map - disp_min) / (disp_max)) * 255.0
disparity_normalized = np.uint8(disparity_normalized)

cv2.imwrite('segment_tree.png',disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
