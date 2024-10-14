import torch 

from dig.threedgraph.dataset import QM93D
# from dig.threedgraph.method import ComENet
from comenet import ComENetModel
from mace import MACEModel
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

device = 'cuda'

# Load the dataset and split
dataset = QM93D(root='/root/workspace/data/')
target = 'U0'
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

# Define model, loss, and evaluation
model = MACEModel(num_layers=3, correlation=1, mlp_dim=128, emb_dim=32)
# model = ComENetModel()
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate
run3d = run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          epochs=1000, batch_size=128, vt_batch_size=128, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=100)