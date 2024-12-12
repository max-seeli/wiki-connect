import json
import os
import pickle

import networkx as nx

from wiki_connect.data import util

script_dir = os.path.dirname(os.path.realpath(__file__))

G = nx.DiGraph()
G.add_node("Hello", 
           info_text="This is a test",
           categories=["Test", "Hello"], 
           page_name="Hello")
G.add_node("World", 
           info_text="This is a test",
           categories=["Test", "World"], 
           page_name="World")
G.add_edge("Hello", "World")


def test_json_loading():
    file_path = os.path.join(script_dir, "test_graph.json")

    with open(file_path, "w") as f:
        json.dump(nx.node_link_data(G, edges="edges"), f, indent=4)

    try:
        loaded_graph = util.load_graph(file_path)

        assert nx.is_directed(loaded_graph)
        assert nx.utils.graphs_equal(G, loaded_graph)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_pkl_loading():
    file_path = os.path.join(script_dir, "test_graph.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(G, f)

    try:
        loaded_graph = util.load_graph(file_path)

        assert nx.is_directed(loaded_graph)
        assert nx.utils.graphs_equal(G, loaded_graph)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_invalid_file_format():
    file_path = os.path.join(script_dir, "test_graph.txt")

    try:
        util.load_graph(file_path)
    except ValueError as e:
        assert str(e) == "Invalid file format. Must be .pkl or .json"
    else:
        assert False, "Should have raised an exception"

def test_json_saving():
    file_path = os.path.join(script_dir, "test_graph.json")

    try:
        util.save_graph(G, file_path)

        with open(file_path, "r") as f:
            loaded_data = json.load(f)

        loaded_graph = nx.node_link_graph(loaded_data, edges="edges")
        assert nx.is_directed(loaded_graph)
        assert nx.utils.graphs_equal(G, loaded_graph)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_pkl_saving():
    file_path = os.path.join(script_dir, "test_graph.pkl")

    try:
        util.save_graph(G, file_path)

        with open(file_path, "rb") as f:
            loaded_graph = pickle.load(f)

        assert nx.is_directed(loaded_graph)
        assert nx.utils.graphs_equal(G, loaded_graph)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_invalid_file_format_saving():
    file_path = os.path.join(script_dir, "test_graph.txt")

    try:
        util.save_graph(G, file_path)
    except ValueError as e:
        assert str(e) == "Invalid file format. Must be .pkl or .json"
    else:
        assert False, "Should have raised an exception"
