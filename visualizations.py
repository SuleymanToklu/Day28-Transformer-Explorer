import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np

def plot_attention_heatmaps(attention_data, tokens, layer_index):
    """
    Plots all attention heads for a given layer as heatmaps.
    """
    # Extract the attention tensor for the specified layer and remove the batch dimension
    attention_for_layer = attention_data[layer_index].squeeze(0)
    num_heads = attention_for_layer.shape[0]

    # Create a grid of subplots (e.g., 3 rows, 4 columns for 12 heads)
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    fig.suptitle(f"Dikkat Desenleri - Katman {layer_index+1}", fontsize=16)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i in range(num_heads):
        ax = axes[i]
        head_attention = attention_for_layer[i].detach().numpy() # Convert tensor to numpy array

        # Create the heatmap
        sns.heatmap(head_attention, xticklabels=tokens, yticklabels=tokens,
                    cmap='viridis', ax=ax, cbar=False)

        ax.set_title(f"Dikkat Kafası {i+1}")
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    # Hide any unused subplots
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_embedding_space(hidden_states, tokens):
    """
    Visualizes word embeddings in 2D space using PCA.
    """
    # Get the embeddings from the last layer and remove the batch dimension
    last_layer_embeddings = hidden_states[-1].squeeze(0).detach().numpy()

    # Use PCA to reduce dimensions from 768 to 2
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(last_layer_embeddings)

    # Create an interactive scatter plot with Plotly
    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        text=tokens,
        title="Kelime Gömülmeleri Uzayı (PCA ile 2D)"
    )

    # Update plot for better readability
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title="Ana Bileşen 1 (Principal Component 1)",
        yaxis_title="Ana Bileşen 2 (Principal Component 2)"
    )

    return fig