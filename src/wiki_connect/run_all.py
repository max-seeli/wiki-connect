import subprocess

# Create a graph of Wikipedia pages by crawling through the links of the pages.
command = "python src/wiki_connect/data/crawler.py --depth 10 --layer_size 1000 --output_path data/wiki_graph.pkl"
subprocess.run(command, shell=True, check=True)

# Load the graph and generate embeddings for each node.
command = "python src/wiki_connect/model/embed.py --graph_path data/wiki_graph.pkl --output_path data/wiki_graph.pt"
subprocess.run(command, shell=True, check=True)

# Hyperparameter optimization on link prediction model on the graph.
grid = {
    "hidden_channels": [32, 64, 128],
    "out_channels": [32, 64],
    "lr": [0.01, 0.001],
}
for hidden_channels in grid["hidden_channels"]:
    for out_channels in grid["out_channels"]:
        for lr in grid["lr"]:
            print(f"\n\n\nTraining with hidden_channels={
                  hidden_channels}, out_channels={out_channels}, lr={lr}", flush=True)
            command = f"python src/wiki_connect/model/trainer.py --data_path data/wiki_graph.pt --hidden_channels {
                hidden_channels} --out_channels {out_channels} --lr {lr}"
            subprocess.run(command, shell=True, check=True)
            print(f"Finished training with hidden_channels={
                  hidden_channels}, out_channels={out_channels}, lr={lr}", flush=True)


# Run inference on the trained model to predict missing links.
# command = "python src/wiki_connect/model/inference.py --graph_path data/wiki_graph.pt --encoder_path best_encoder.pth --predictor_path best_predictor.pth"
# subprocess.run(command, shell=True, check=True)
