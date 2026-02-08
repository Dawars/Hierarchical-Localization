"""
Paper: Leveraging Camera Triplets for Efficient and Accurate Structure-from-Motion
https://ee.iisc.ac.in/cvlab/research/camtripsfm/
"""

from pathlib import Path
from collections import defaultdict
from typing import Dict

from tqdm import tqdm
import pycolmap
import networkx as nx
from . import logger

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = int((pair_id - image_id2) / 2147483647)
    return image_id1, image_id2

def enumerate_triangles_nx(graph):
    """Use NetworkX's optimized triangle enumeration"""
    triangles = set()
    # nx.triangles returns {node: count} dict
    # We need to enumerate actual triangles
    for edge in graph.edges():
        common_neighbors = set(graph.neighbors(edge[0])) & set(graph.neighbors(edge[1]))
        for cn in common_neighbors:
            triangle = tuple(sorted([edge[0], edge[1], cn]))
            triangles.add(triangle)
    return list(triangles)


def remove_non_tri_edges(G, verbose: bool = False):
    tri_edges = set()
    triangles = enumerate_triangles_nx(G)

    for a, b, c in triangles:
        tri_edges.add(tuple(sorted((a, b))))
        tri_edges.add(tuple(sorted((b, c))))
        tri_edges.add(tuple(sorted((a, c))))

    # Remove edges not in any triangle
    edges_to_remove = [tuple(sorted(e)) for e in G.edges() if tuple(sorted(e)) not in tri_edges]
    if verbose:
        logger.info(f"Triangle-supported edges: {len(tri_edges)}")
        logger.info(f"Edges to remove: {len(edges_to_remove)}")

    G.remove_edges_from(edges_to_remove)


def score_edges(graph, inlier_counts, verbose: bool = False):
    """
    For every triplet (a,b,c) in the graph:
    - compute relative edge score per edge in triplet
    - accumulate scores to get final per-edge score
    """
    # accumulate score sums & counts
    edge_scores_sum = defaultdict(float)
    edge_scores_cnt = defaultdict(int)
    triangles = enumerate_triangles_nx(graph)

    for (a, b, c) in tqdm(triangles, desc="Scoring triplets", disable=not verbose):

        eab, ebc, eac = image_ids_to_pair_id(a, b), image_ids_to_pair_id(b, c), image_ids_to_pair_id(a, c)
        # get inliers
        nab = inlier_counts.get(eab, 0)
        nbc = inlier_counts.get(ebc, 0)
        nac = inlier_counts.get(eac, 0)

        # max
        m = max(nab, nbc, nac, 1)
        # relative scores
        for e, n in [(eab, nab), (ebc, nbc), (eac, nac)]:
            score = float(n) / float(m)
            edge_scores_sum[e] += score
            edge_scores_cnt[e] += 1

    # final averages
    edge_score = {e: edge_scores_sum[e] / edge_scores_cnt[e] for e in inlier_counts.keys() if edge_scores_cnt[e] > 0}

    return edge_score


def adaptive_threshold(graph, min_score=0.5):
    """
    Compute adaptive threshold based on graph connectivity.

    τ = m * (1 - dmax/|V|) + dmax/|V|

    where:
    - m is the minimum score edges should satisfy
    - dmax is the maximum node degree
    - |V| is the number of nodes
    """
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return min_score

    # Maximum degree in the graph
    degrees = dict(graph.degree())
    dmax = max(degrees.values()) if degrees else 0

    # Adaptive threshold formula from paper
    tau = min_score * (1 - dmax / num_nodes) + (dmax / num_nodes)

    logger.info(f"Adaptive threshold: τ = {tau:.4f} (dmax={dmax}, |V|={num_nodes}, m={min_score})")

    return tau


def apply_camera_triplet_pruning(database_path: Path, image_ids: Dict[str, int], camera_triplet_threshold: float, verbose: bool = False):
    # hypterparams
    min_inlier_score = 15  # don't add edges (image pairs) to the graph with num_inliers below this number

    with pycolmap.Database.open(database_path) as db:
        inlier_counts = db.read_two_view_geometry_num_inliers()
    id_to_name = {image_id: image_name for image_name, image_id in image_ids.items()}

    G = nx.Graph()
    G.add_nodes_from(id_to_name.keys())  # image_ids
    for pair_id, num_inliers in zip(*inlier_counts):
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        if num_inliers >= min_inlier_score:
            G.add_edge(image_id1, image_id2)

    # Find all connected components
    components = list(nx.connected_components(G))
    if verbose:
        logger.info(f"Found {len(components)} connected components")
    # Sort by size (largest first)
    components_sorted = sorted(components, key=len, reverse=True)

    # Print component sizes
    if verbose:
        for i, comp in enumerate(components_sorted[:10]):  # Show top 10
            logger.info(f"  Component {i + 1}: {len(comp)} nodes")

    # Create subgraphs
    component_graphs = [G.subgraph(comp).copy() for comp in components_sorted]

    remove_non_tri_edges(component_graphs[0])
    inlier_dict = {pair_id: num_inliers for pair_id, num_inliers in zip(*inlier_counts)}
    edge_scores = score_edges(component_graphs[0], inlier_dict)

    tau = adaptive_threshold(component_graphs[0], min_score=camera_triplet_threshold)
    num_removed_edges = 0
    with pycolmap.Database.open(database_path) as db:
        for pair_id, score in tqdm(edge_scores.items(), disable=not verbose):
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            if score < tau:
                db.delete_inlier_matches(image_id1, image_id2)
                num_removed_edges += 1
    if verbose:
        logger.info(f"{num_removed_edges} edges with scores lower than {tau=} {camera_triplet_threshold=} removed")

