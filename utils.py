#%%
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import plotly.graph_objects as go
import plotly.figure_factory as ff
from ipywidgets import HBox
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

def viz_func(dataset, sample_no):
    labels = {0:"bathtub", 1:"bed", 2:"chair", 3:"desk", 4:"dresser", 5:"monitor", 6:"night_stand", 7:"sofa", 8:"table", 9:"toilet"}
    data = dataset[sample_no]
    x = data.pos[:, 0]
    y = data.pos[:, 1]
    z = data.pos[:, 2]
    mesh = ff.create_trisurf(z=z, x=x, y=y, simplices=data.face.t(), colormap="Portland", width=600, height=600, show_colorbar=False)
    mesh.update_layout(margin=dict(l=0, r=0), width=600, height=600,coloraxis_showscale=False)
    mesh.update_layout(title_text=f'Mesh Representation ({labels[data.y.item()]})', title_x=0.5)
    mesh.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )
    mesh=go.FigureWidget(mesh)

    new_ds = dataset.copy()
    new_ds.transform = T.SamplePoints(2048)
    new_dt = new_ds[sample_no]

    pointcloud = go.FigureWidget(data=[go.Scatter3d(
        x=new_dt.pos[:,0],
        y=new_dt.pos[:,1],
        z=new_dt.pos[:,2],
        mode='markers',
        marker=dict(color=z, colorscale='Portland', size=4)
    )])
    pointcloud.update_layout(margin=dict(l=0, r=0), width=600, height=600)
    pointcloud.update_layout(title_text=f'Point Cloud Representation ({labels[data.y.item()]})', title_x=0.5, showlegend=False)
    pointcloud.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )

    fig_subplots=  HBox([mesh, pointcloud])

    return fig_subplots


def get_data(batch_size=32):
    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2),
        T.SamplePoints(1024, include_normals=True)
    ])
    pre_transform = T.NormalizeScale()
    root_dir = 'data/ModelNet'
    dataset_name = '10'
    train_dataset = ModelNet(root=root_dir, name=dataset_name, train=True, transform=transform, pre_transform=pre_transform)
    test_dataset = ModelNet(root=root_dir, name=dataset_name, train=False, transform=transform, pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset.num_classes



# %%
def train_one_epoch(model, optimizer, loader, criterion, device):
    model.train()
    
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        logits = model(data.pos, data.normal, data.batch)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    total_loss = total_loss / len(loader.dataset)
    return total_loss


def test_model(model, loader, criterion, device):
    model.eval()
    test_preds = []
    test_truths = []

    total_loss = 0
    total_correct = 0
    for data in loader:
        data = data.to(device)
        logits = model(data.pos, data.normal, data.batch)
        pred = logits.argmax(dim=-1)
        loss = criterion(logits, data.y)
        total_loss += loss.item() * data.num_graphs
        total_correct += int((pred == data.y).sum())
        test_preds.extend(pred.cpu().detach().numpy())
        test_truths.extend(data.y.cpu().detach().numpy())
    test_truths = np.ndarray.flatten(np.array(test_truths))
    test_preds = np.ndarray.flatten(np.array(test_preds))
    ba = balanced_accuracy_score(test_truths, test_preds)
    acc = total_correct / len(loader.dataset)
    total_loss = total_loss / len(loader.dataset)
    return ba, acc, pred, total_loss


