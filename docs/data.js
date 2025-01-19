export function getWikiDataJSON() {
    return fetch('./data/deep_learning_graph.json')
        .then(response => response.json())
        .then(data => {
            const startTime = new Date().getTime();

            data.nodes = [...new Map(data.nodes.map(node => [node.id, node])).values()];
            // Make all edges have a lexigraphically smaller source than target
            data.edges = data.edges.map(edges => {
                if (edges.source > edges.target) {
                    const temp = edges.source;
                    edges.source = edges.target;
                    edges.target = temp;
                }
                return edges;
            });
            data.edges = [...new Map(data.edges.map(edges => [`${edges.source}-${edges.target}`, edges])).values()];
            const endTime = new Date().getTime();
            console.log(`Time taken to process data: ${endTime - startTime} ms`);
            return data;
        });
}

export function getPredictionDataJSON(name) {
    return fetch('./data/deep_learning_graph_predictions.json')
        .then(response => response.json());
}
