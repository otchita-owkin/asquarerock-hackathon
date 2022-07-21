from typing import List
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from ge_hack.data import OmicsDataset
from ge_hack.data.loading import load_rnaseq, load_mutations
from ge_hack.data.preprocessing import filter_by_freq
import matplotlib.pyplot as plt
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class AutoEncoder(torch.nn.Module):
    """
    Linear Auto Encoder
    Parameters
    ----------
    in_features: int
    hidden: List[int]
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        repr_dim: int,
        hidden: List[int] = [],
        bias: bool = True,
        num_epochs: int = 10,
        dropout: int = 0.5,
        batch_size: int = 16,
        learning_rate: float = 1.0e-3,
        linear: bool = True,
        device: str = "cuda:0",
        edge_index = torch.tensor( [[0, 1, 1, 2], 
                                    [1, 0, 2, 1]], dtype=torch.long ),
        ):

        super(AutoEncoder, self).__init__()

        self.criterion = torch.nn.MSELoss()
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.linear = linear
        self.train_loss, self.eval_loss = [], []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and device.startswith("cuda"):
            self.device = torch.device(device)
        
        self.edge_index = edge_index

        hidden.append(repr_dim)
        encoder_hidden = hidden
        decoder_hidden = hidden[::-1][1:] if len(hidden) > 1 else []
        decoder_hidden.append(in_features)

        encoder_layers = []
        encoder_layers.append((GCNConv(1, 1), 'x, edge_index -> x'))
        encoder_layers.append(torch.nn.ReLU(inplace=True))
        encoder_layers.append((GCNConv(1, 1), 'x, edge_index -> x'))
        encoder_layers.append(torch.nn.ReLU(inplace=True))
        for h in encoder_hidden:
            encoder_layers.append(torch.nn.Linear(in_features, h, bias=bias))
            if not self.linear:
                encoder_layers.append(torch.nn.ReLU())
            if self.dropout:
                encoder_layers.append(torch.nn.Dropout(self.dropout))
            in_features = h
        self.encoder = Sequential('x, edge_index', encoder_layers)

        decoder_layers = []
        for h in decoder_hidden:
            decoder_layers.append(torch.nn.Linear(in_features, h, bias=bias))
            if not self.linear:
                decoder_layers.append(torch.nn.ReLU())
            if self.dropout:
                decoder_layers.append(torch.nn.Dropout(self.dropout))
            in_features = h
        self.decoder = torch.nn.Sequential(*decoder_layers)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, self.edge_index)
        return self.decoder(z)

    def fit(self, X, split_data):
        # split data
        if split_data:
            X, X_eval = train_test_split(X)

        dataset = OmicsDataset(X)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.encoder.train()
        self.decoder.train()

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            epoch_loss = 0
            for x in dataloader:
                import ipdb; ipdb.set_trace();
                x = x.to(self.device)
                x_hat = self.forward(x)
                loss = self.criterion(x_hat, x)
                epoch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss /= len(dataloader)
            self.train_loss.append(epoch_loss.detach().cpu().numpy()) 

            if split_data:  
                self.evaluate(X_eval)

        return self
    
    def evaluate(self, X_eval):
        dataset = OmicsDataset(X_eval)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.encoder.eval()
        self.decoder.eval()

        epoch_loss = 0
        for x in dataloader:
            x = x.to(self.device)
            x_hat = self.forward(x)
            loss = self.criterion(x_hat, x)
            epoch_loss += loss

        loss /= len(dataloader)
        self.eval_loss.append(epoch_loss.detach().cpu().numpy())    

        self.encoder.train()
        self.decoder.train()


    def transform(self, X):
        
        self.encoder.eval()
        self.decoder.eval()
        
        dataset = OmicsDataset(X)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        features = []
        for x in tqdm(dataloader, total=len(dataloader)):
            x = x.to(self.device)
            f = self.encoder(x)
            features.append(f.detach())
        
        if self.device == "cpu":
            features = torch.cat(features, dim=0).numpy()
        else:
            features = torch.cat(features, dim=0).cpu().numpy()
        
        return features

    def fit_transform(self, X):
        _ = self.fit(X)
        return self.transform(X)


def data_type_ccle(is_ccle=False):
    if is_ccle:
        return 'ccle/'
    else:
        return 'tcga/'


if __name__ == "__main__":

    data_prefix_path = '/Data_hackathon/'
    data_directory = data_prefix_path + data_type_ccle()
    path_to_data = data_directory+"processed/"
    full_df_rna_log = pd.read_csv(path_to_data+ "tcga_expression_processed.csv", index_col=[0])
    X_rna = full_df_rna_log.values
    print(X_rna.shape)

    print(torch.cuda.is_available())

    edge_index = torch.tensor( [[0, 1, 1, 2],
                                [1, 0, 2, 1]], dtype=torch.long )

    autoencoder_rna = AutoEncoder(
                    in_features=X_rna.shape[1],
                    repr_dim = 128,
                    hidden = [ 1024 ],
                    bias = False,
                    num_epochs = 20,
                    dropout = True,
                    batch_size = 32,
                    learning_rate = 1e-4,
                    linear = False,
                    device = "cpu",
                    edge_index = edge_index,
                             )
    autoencoder_rna.fit(X=X_rna, split_data=True)
    plt.plot(autoencoder_rna.train_loss, color="red", label="train loss")
    plt.plot(autoencoder_rna.eval_loss, color="blue", label="validation loss")
    plt.grid(True)
    plt.legend()
    plt.title("Expression AE loss")
    plt.savefig("rna_ae_perf.png")
    