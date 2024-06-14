import torch 


def dataloader(data, batch_size):
    # Create a DataLoader
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    return loader


def train(data, model, optimizer, loss_fn, epochs, batch_size):
    # Create a DataLoader
    loader = dataloader(data, batch_size)
    
    # Loop through the epochs
    for epoch in range(epochs):
        # Initialize the loss
        loss = 0
        
        # Loop through the DataLoader
        for i, x in enumerate(loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Pass the input through the model
            out, z_mean, z_log_var = model(x)
            
            # Calculate the loss
            loss = loss_fn(x, out, z_mean, z_log_var)
            
            # Backpropagate
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Add the loss to the total loss
            loss += loss.item()
        
        # Print the loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return model


def test(data, model):
    # Pass the data through the model
    out, z_mean, z_log_var = model(data)
    
    return out, z_mean, z_log_var

