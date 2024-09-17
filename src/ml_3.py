"""
Feedforward Neural Network
"""

def feedforward_nn(X, y, hidden_layers, num_iterations, learning_rate):
    """
    Feedforward Neural Network
    """
    # Initialize weights and biases
    weights, biases = initialize_weights(X, hidden_layers)
    # Train the model
    for _ in range(num_iterations):
        # Forward pass
        activations = forward_pass(X, weights, biases)
        # Compute loss
        loss = compute_loss(y, activations)
        # Backward pass
        gradients = backward_pass(X, y, activations, weights, biases)
        # Update weights and biases
        weights, biases = update_weights(weights, biases, gradients, learning_rate)
    return weights, biases