import torch

class Loss(torch.nn.Module):
    def __init__(self, kl_factor = 1.):
        super(Loss, self).__init__()
        self.kl_factor = kl_factor
        
    def forward(self, x, out, z_mean, z_log_var):
        # Calculate the reconstruction loss
        x_expanded = x.unsqueeze(0).expand(*out.shape)
        recon_loss = torch.nn.functional.mse_loss(x_expanded, out, reduction='mean')
        print('mse',recon_loss.shape,recon_loss)
        # Calculate the KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        print('kl',kl_loss.shape,kl_loss)

        return recon_loss + self.kl_factor*kl_loss