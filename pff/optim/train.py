import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter




def train_pff(
        model,
        train_dset,
        batch_size,
        optimiser,
        epochs,
        stats=None,
):
    writer = SummaryWriter(f"logs/{model.__class__.__name__}")
    if stats is None:
        stats = [
            # Train
            {
            'step': 0,
            'rep_g_loss': [],
            'rep_c_loss': [],
            'gen_loss': [],
            'x_mse': [],
            'y_ce': [],
            },
            # Validation
            {
            'rep_g_loss': [],
            'rep_c_loss': [],
            'gen_loss': [],
            'x_mse': [],
            'y_ce': [],
            }
        ]
    
    device = train_dset.device

    model.train()
    for epoch in range(epochs):
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i, (x, y) in loop:
            if epoch > 0 or i > 0:
                loop.set_description(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"rep_g_loss={stats[0]['rep_g_loss'][-1]:.4f} - "
                    f"rep_c_loss={stats[0]['rep_c_loss'][-1]:.4f} - "
                    f"gen_loss={stats[0]['gen_loss'][-1]:.4f} - "
                    f"x_mse={stats[0]['x_mse'][-1]:.4f} - "
                    f"y_ce={stats[0]['y_ce'][-1]:.4f}"
                )
            x = torch.cat([x, x])
            y = torch.cat([y, (y + torch.randint(1, 10, y.shape).to(device)) % 10])
            y = F.one_hot(y, 10).float()
            label = torch.cat([torch.ones((x.shape[0] // 2)).to(device), torch.zeros((x.shape[0] // 2)).to(device)])

            x_hat, y_hat, (rep_g_loss, rep_c_loss), gen_loss = model.infer_and_generate(x, y, label, 10, optimiser)

            x_mse = torch.mean((x_hat - x).square())
            y_ce = F.cross_entropy(y_hat, torch.argmax(y, dim=1))

            stats[0]['step'] += x.shape[0]//2
            stats[0]['rep_g_loss'].append(rep_g_loss.item())
            stats[0]['rep_c_loss'].append(rep_c_loss.item())
            stats[0]['gen_loss'].append(gen_loss.item())
            stats[0]['x_mse'].append(x_mse.item())
            stats[0]['y_ce'].append(y_ce.item())

            writer.add_scalar("rep_g_loss", rep_g_loss.item(), stats[0]['step'])
            writer.add_scalar("rep_c_loss", rep_c_loss.item(), stats[0]['step'])
            writer.add_scalar("gen_loss", gen_loss.item(), stats[0]['step'])
            writer.add_scalar("x_mse", x_mse.item(), stats[0]['step'])
            writer.add_scalar("y_ce", y_ce.item(), stats[0]['step'])
    
    return stats

            

