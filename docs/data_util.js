const owner = "max-seeli"; // Replace with repository owner
const repo = "wiki-connect"; // Replace with repository name
    

async function downloadReleaseAsset(owner, repo, assetName) {
    try {
        console.log("Hi");  
        // Step 1: Get the latest release
        let releaseUrl = `https://api.github.com/repos/${owner}/${repo}/releases/latest`;
        let releaseResponse = await fetch(releaseUrl); /*, {
            headers: { 'Authorization': `token ${token}` }
        });*/

        if (!releaseResponse.ok) throw new Error(`Error fetching release: ${releaseResponse.statusText}`);
        let releaseData = await releaseResponse.json();

        console.log('Found latest release:', releaseData);

        // Step 2: Find the asset ID for the given asset name
        let asset = releaseData.assets.find(a => a.name === assetName);
        if (!asset) throw new Error(`Asset ${assetName} not found in the latest release`);

        let assetId = asset.id;
        console.log(`Found asset: ${assetName} (ID: ${assetId})`);

        // Step 3: Download the asset
        let assetUrl = asset.url;
        console.log(`Downloading asset from ${assetUrl}`);
        let assetResponse = await fetch(assetUrl, {
            headers: {
                'Accept': 'application/octet-stream',  
            }
        });

        if (!assetResponse.ok) throw new Error(`Error downloading asset: ${assetResponse.statusText}`);

        // Step 4: Convert response to JSON
        console.log(assetResponse)
        let assetData = await assetResponse.json();
        console.log(assetData);
        return assetData;
    } catch (error) {
        console.error(error);
    }
}




export async function getWikiDataJSON() {

    console.log("Hii");
    let results = await downloadReleaseAsset(owner, repo, "deep_learning_graph.json");
    console.log(results);
    return results;
    /*
    return fetch('https://github.com/max-seeli/wiki-connect/releases/download/v0.1.0/deep_learning_graph.json')
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
    */
}

export async function getPredictionDataJSON(name) {
    console.log("Hii");
    let results = await downloadReleaseAsset(owner, repo, "deep_learning_graph_predictions.json");
    console.log(results);
    return results;
}
