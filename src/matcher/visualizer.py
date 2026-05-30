import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ResearchVisualizer:
    """
    Microsoft-Level Visualization Suite for ID Matching.
    """
    @staticmethod
    def plot_cost_matrix(matrix, nodes_t, nodes_t1):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='viridis')
        
        ax.set_xticks(np.arange(len(nodes_t1)))
        ax.set_yticks(np.arange(len(nodes_t)))
        ax.set_xticklabels([n['label'].split('.')[-1] for n in nodes_t1], rotation=45)
        ax.set_yticklabels([n['label'].split('.')[-1] for n in nodes_t])
        
        plt.colorbar(im, ax=ax, label='Matching Cost (Lower is Better)')
        ax.set_title("Inter-Frame Identity Matching Cost Matrix")
        return fig

    @staticmethod
    def render_metrics(results_df):
        st.table(results_df)
