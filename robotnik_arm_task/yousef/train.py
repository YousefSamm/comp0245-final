import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader

def make_loaders(D, net="b", batch_size=1024):
    """net = "b" (base) or "a" (arm). Returns train_loader, val_loader."""
    X, Y = f"X{net}", f"Y{net}"
    def norm(split):
        x = (D[f"{X}_{split}"] - D[f"{X}_mu"]) / D[f"{X}_sd"]
        y = (D[f"{Y}_{split}"] - D[f"{Y}_mu"]) / D[f"{Y}_sd"]
        return TensorDataset(torch.from_numpy(x.astype(np.float32)),
                             torch.from_numpy(y.astype(np.float32)))
    train_dl = DataLoader(norm("train"), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(norm("val"),   batch_size=batch_size * 4, shuffle=False)
    return train_dl, val_dl

def train_model (model, train_dl, val_dl, epochs=30, lr=1e-3, patience=5, device="cuda"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best_val, best_state, bad_epochs = float("inf"), None, 0
    hist = {"train": [], "val": []}

    for ep in range(epochs):
        model.train()
        tot = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        train_loss = tot / len(train_dl.dataset)

        model.eval()
        tot = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                tot += loss_fn(model(xb), yb).item() * len(xb)
        val_loss = tot / len(val_dl.dataset)

        hist["train"].append(train_loss); hist["val"].append(val_loss)
        print(f"epoch {ep+1:2d} train {train_loss:.4f} val{val_loss:.4f}")


        if val_loss < best_val:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k,v in model.state_dict().items()}
        else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("early stop"); break
    model.load_state_dict(best_state)
    return model, hist