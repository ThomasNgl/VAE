import torch



class Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_dimensions=[16, 8, 4]):
        super(Encoder, self).__init__()
        
        # Define the layer dimensions
        self.layer_dimensions = [input_dim] + layer_dimensions

        self.num_layers = len(self.layer_dimensions) - 1 
        # Initialize an empty list to hold the layers
        layers = []

        # Loop through the layer dimensions to create Linear and ReLU layers
        for i in range(len(self.layer_dimensions) - 1):
            # Add a Linear layer
            layers.append(torch.nn.Linear(self.layer_dimensions[i], self.layer_dimensions[i + 1]))
            
            # Add a ReLU layer, except after the last Linear layer
            if i < len(self.layer_dimensions) - 2:
                layers.append(torch.nn.ReLU())

        # Create the Sequential model with the layers
        self.model = torch.nn.Sequential(*layers)
        

    def forward(self, x):
        # Pass the input through the model
        a = self.model(x)

        z_mean, z_log_var = a[:,:2], a[:,2:] 
        return  z_mean, z_log_var