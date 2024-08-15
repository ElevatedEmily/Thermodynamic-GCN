import jax.numpy as jnp
from thermox.sampler import sample
import networkx as nx
import jax

# Example: Create a graph and compute its Laplacian
G = nx.karate_club_graph()
L = nx.laplacian_matrix(G).todense()
theta = 0.1  # Mean-reversion rate
A = theta * jnp.array(L)

num_nodes = L.shape[0]
b = jnp.zeros(num_nodes)  # Assuming zero drift
D = jnp.eye(num_nodes)  # Identity diffusion matrix

def ou_graph_convolution(samples, W):
    """
    Performs a graph convolution using the samples from the OU process.
    
    Args:
    - samples: Output from Thermox's OU process simulation (shape: [num_time_steps, num_nodes]).
    - W: Weight matrix for the convolution (shape: [num_features, output_dim]).
    
    Returns:
    - convolved_features: Convolved features after applying the OU process and weights.
    """
    # Example: Use the final time step's samples as the feature input for convolution
    final_features = samples[-1]  # Take the last time step
    
    # Apply a linear transformation (equivalent to a learned weight matrix)
    convolved_features = jnp.dot(final_features, W)
    
    return convolved_features

# Example usage with a random weight matrix
num_features = samples.shape[1]
output_dim = 16  # Output dimension for the convolution
W = jax.random.normal(key, (num_features, output_dim))

convolved_features = ou_graph_convolution(samples, W)
