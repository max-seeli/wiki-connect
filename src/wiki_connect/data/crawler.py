from typing import List

import networkx as nx
import numpy as np
from tqdm import tqdm

from wiki_connect.data.util import draw_graph, save_graph
from wiki_connect.data.wikimedia_api import get_page_infos


def filter_titles(titles: List[str]) -> List[str]:
    """
    Filter out titles that start with a forbidden prefix. These prefixes are typically used for
    Wikipedia pages that are not actual articles.

    Parameters
    ----------
    titles : List[str]
        The titles to filter.

    Returns
    -------
    List[str]
        The filtered list of titles.
    """
    forbidden_prefixes = ["File:", "Category:",
                          "Wikipedia:", "Template:", "Portal:", "Template talk:"]
    return [title for title in titles if not any(title.startswith(prefix) for prefix in forbidden_prefixes)]


def build_graph(start_title, depth=2, layer_size=50):
    """
    Build a graph of Wikipedia pages by crawling through the links of the pages.

    Parameters
    ----------
    start_title : str
        The title of the Wikipedia page to start the crawl from.
    depth : int
        The depth of the crawl. This determines how many layers of pages are crawled.
    layer_size : int
        The number of pages to crawl in each layer.

    Returns
    -------
    nx.DiGraph
        The built graph.
    """
    graph = nx.DiGraph()
    graph.add_node(start_title)
    queue = set([start_title])
    visited = set()
    pending_edges = []

    for _ in tqdm(range(depth)):
        current_titles = [title for title in queue if title not in visited]

        if not current_titles:
            # No more pages to crawl
            break

        referenced_pages = get_page_infos(current_titles)

        # Remove pages that have no info text (can't be used for training)
        referenced_pages = {title: info for title,
                            info in referenced_pages.items() if info["info_text"]}

        # Add exactly those edges that connect two nodes, which we have already crawled
        visited.update(referenced_pages.keys())
        pending_edges = list(
            filter(lambda x: x[0] in visited and x[1] in visited, pending_edges))
        graph.add_edges_from(pending_edges)

        new_page_pool = set()
        for title, info in referenced_pages.items():
            if title not in graph.nodes:
                continue
            
            graph.nodes[title]["title"] = info["title"]
            graph.nodes[title]["info_text"] = info["info_text"]
            graph.nodes[title]["categories"] = info["categories"]

            for page in filter_titles(info["links"]):
                if page == title:
                    # Remove potential self loops, that occur with redirects
                    continue

                if page in visited:
                    graph.add_edge(title, page)
                else:
                    new_page_pool.add((title, page))

        # Randomly select a subset of size layer_size from the new pages to crawl
        new_page_pool = list(new_page_pool)
        selected_idx = np.random.choice(len(new_page_pool), size=min(
            len(new_page_pool), layer_size), replace=False)
        selected_pages = [new_page_pool[i] for i in selected_idx]

        pending_edges = selected_pages
        queue.update(page for _, page in selected_pages)

    return graph


def main(start_title, depth, layer_size, output_path):
    """
    Main function to build a graph of Wikipedia pages and save it to a file.

    Parameters
    ----------
    start_title : str
        The title of the Wikipedia page to start the crawl from.
    depth : int
        The depth of the crawl. This determines how many layers of pages are crawled.
    layer_size : int
        The number of pages to crawl in each layer.
    output_path : str
        The path to save the graph to.
    """
    graph = build_graph(start_title, depth=depth, layer_size=layer_size)
    save_graph(graph, output_path)
    draw_graph(graph)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--start-title", type=str, default="Computer science")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--layer-size", type=int, default=10)
    parser.add_argument("--output-path", type=str,
                        default="data/test_graph.json")

    args = parser.parse_args()
    main(args.start_title, args.depth, args.layer_size, args.output_path)
