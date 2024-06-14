import torch

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    
    def forward(self, x, out, z_mean, z_log_var):
        # Calculate the reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(x, out, reduction='sum')
        
        # Calculate the KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        
        return recon_loss + 0.1*kl_loss