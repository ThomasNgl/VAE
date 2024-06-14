import torch
from encoder import Encoder
from decoder import Decoder

class VAE(torch.nn.Module):
    def __init__(self, input_dim, sample_size = 10, encoder_layer_dimensions=[16, 8, 2], decoder_layer_dimensions=[2, 8, 16]):
        super(VAE, self).__init__()
        
        # Initialize the encoder and decoder
        self.encoder = Encoder(input_dim, )
        self.decoder = Decoder(input_dim, )
        self.sample_size = sample_size

    def sample(self, z_mean, z_log_var):
        # Generate a random sample
        epsilon = torch.randn(self.sample_size, *z_mean.shape)
        # Re-parameterization trick
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        
        return z
    
    def forward(self, x):
        # Pass the input through the encoder
        z_mean, z_log_var = self.encoder(x)
        
        # Sample from the distribution
        z = self.sample(z_mean, z_log_var)
        # Pass the sample through the decoder
        out = self.decoder(z)

        return out, z_mean, z_log_var    