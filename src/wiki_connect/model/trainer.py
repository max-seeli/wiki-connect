import argparse

import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, precision_score,
                             roc_auc_score)
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected

from wiki_connect.model.encoder import GCNEncoder
from wiki_connect.model.predictor import LinkPredictor


class LinkPredictionTrainer:
    def __init__(self, data_path, hidden_channels, out_channels, lr, epochs, device):
        self.data_path = data_path
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

        self._load_data()
        self._initialize_model()

    def _load_data(self):
        data = torch.load(self.data_path, weights_only=False)
        data.edge_index = to_undirected(data.edge_index)
        self.num_node_features = data.x.size(1)

        transform = RandomLinkSplit(
            num_val=0.15, num_test=0.15, split_labels=False, is_undirected=True)
        # new attributes are introduced:
        # edge_label_index contains positive and negative edges with respective labels in edge_label
        self.train_data, self.val_data, self.test_data = transform(data)

        self.train_data = self.train_data.to(self.device)
        self.val_data = self.val_data.to(self.device)
        self.test_data = self.test_data.to(self.device)

    def _initialize_model(self):
        self.encoder = GCNEncoder(
            self.num_node_features, self.hidden_channels, self.out_channels).to(self.device)
        self.predictor = LinkPredictor(self.out_channels).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.predictor.parameters()),
            lr=self.lr
        )

    def train(self):
        """Train the model for one epoch."""
        self.encoder.train()
        self.predictor.train()
        self.optimizer.zero_grad()

        x = self.encoder(self.train_data.x, self.train_data.edge_index)

        out = self.predictor(
            x[self.train_data.edge_label_index[0]], x[self.train_data.edge_label_index[1]])
        labels = self.train_data.edge_label.unsqueeze(1).to(self.device)

        loss = F.binary_cross_entropy(out, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data):
        """
        Evaluate the model on the given data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The data to evaluate on.

        Returns
        -------
        auc : float
            The ROC AUC score.
        ap : float
            The average precision score.
        p : float
            The precision score.
        """
        self.encoder.eval()
        self.predictor.eval()
        x = self.encoder(data.x, data.edge_index)

        y_true = data.edge_label.cpu().numpy()
        y_score = self.predictor(
            x[data.edge_label_index[0]], x[data.edge_label_index[1]]).cpu().numpy().flatten()

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        p = precision_score(y_true, y_score > 0.5)
        return auc, ap, p

    def run(self):
        """Train the model and evaluate on the test set."""
        best_val_auc, best_val_ap, best_val_p = 0, 0, 0

        for epoch in range(1, self.epochs + 1):
            loss = self.train()

            if epoch % 10 == 0:
                val_auc, val_ap, val_p = self.evaluate(self.val_data)
                print(f"Epoch: {epoch}, Loss: {loss:.4f}, Val AUC: {
                      val_auc:.4f}, Val AP: {val_ap:.4f}, Val P: {val_p:.4f}")

                if val_auc > best_val_auc and val_ap > best_val_ap and val_p > best_val_p:
                    best_val_auc = val_auc
                    best_val_ap = val_ap
                    best_val_p = val_p

                    torch.save(self.encoder.state_dict(), "best_encoder.pth")
                    torch.save(self.predictor.state_dict(),
                               "best_predictor.pth")

        test_auc, test_ap, test_p = self.evaluate(self.test_data)
        print(f"Test AUC: {test_auc:.4f}, Test AP: {
              test_ap:.4f}, Test P: {test_p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link Prediction Trainer")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the graph data file")
    parser.add_argument("--hidden_channels", type=int,
                        default=128, help="Number of hidden channels")
    parser.add_argument("--out_channels", type=int,
                        default=64, help="Number of output channels")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training (cuda or cpu)")

    args = parser.parse_args()

    trainer = LinkPredictionTrainer(
        data_path=args.data_path,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device
    )
    trainer.run()
