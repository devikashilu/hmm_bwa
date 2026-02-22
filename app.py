"""
HMM Baum-Welch Algorithm Calculator
A simple implementation for training Hidden Markov Models
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Page configuration
st.set_page_config(
    page_title="HMM Baum-Welch Calculator",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .matrix-box {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


class HiddenMarkovModel:
    """Hidden Markov Model with Baum-Welch Algorithm"""
    
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        self.N = n_states
        self.M = n_observations
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize with uniform probabilities"""
        self.pi = np.ones(self.N) / self.N
        self.A = np.ones((self.N, self.N)) / self.N
        self.B = np.ones((self.N, self.M)) / self.M
    
    def set_parameters(self, pi, A, B):
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.B = np.array(B)
    
    def forward_algorithm(self, obs_sequence):
        T = len(obs_sequence)
        alpha = np.zeros((T, self.N))
        
        alpha[0] = self.pi * self.B[:, obs_sequence[0]]
        
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, obs_sequence[t]]
        
        P = np.sum(alpha[T-1])
        return alpha, P
    
    def backward_algorithm(self, obs_sequence):
        T = len(obs_sequence)
        beta = np.zeros((T, self.N))
        
        beta[T-1] = np.ones(self.N)
        
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_sequence[t+1]] * beta[t+1])
        
        return beta
    
    def compute_gamma(self, alpha, beta, P):
        if P == 0:
            return np.zeros_like(alpha)
        return (alpha * beta) / P
    
    def compute_xi(self, alpha, beta, A, B, obs_sequence, P):
        T = len(obs_sequence)
        xi = np.zeros((T-1, self.N, self.N))
        
        for t in range(T-1):
            for i in range(self.N):
                for j in range(self.N):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, obs_sequence[t+1]] * beta[t+1, j]
        
        if P > 0:
            xi = xi / P
        
        return xi
    
    def baum_welch(self, obs_sequence, max_iterations=100, tolerance=1e-6):
        """Run Baum-Welch algorithm"""
        history = {
            'pi': [self.pi.copy()],
            'A': [self.A.copy()],
            'B': [self.B.copy()],
            'log_likelihood': []
        }
        
        for iteration in range(max_iterations):
            alpha, P = self.forward_algorithm(obs_sequence)
            beta = self.backward_algorithm(obs_sequence)
            
            if P == 0:
                break
            
            history['log_likelihood'].append(np.log(P) if P > 0 else -np.inf)
            
            gamma = self.compute_gamma(alpha, beta, P)
            xi = self.compute_xi(alpha, beta, self.A, self.B, obs_sequence, P)
            
            # Update initial probabilities
            new_pi = gamma[0]
            
            # Update transition probabilities
            new_A = np.zeros((self.N, self.N))
            for i in range(self.N):
                gamma_sum = np.sum(gamma[:-1, i])
                if gamma_sum > 0:
                    new_A[i, :] = np.sum(xi[:, i, :], axis=0) / gamma_sum
            
            # Update emission probabilities
            new_B = np.zeros((self.N, self.M))
            for j in range(self.N):
                for o_idx in range(self.M):
                    mask = np.array(obs_sequence) == o_idx
                    if np.sum(mask) > 0:
                        new_B[j, o_idx] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])
            
            # Normalize
            new_pi = new_pi / np.sum(new_pi)
            new_A = new_A / np.sum(new_A, axis=1, keepdims=True)
            new_B = new_B / np.sum(new_B, axis=1, keepdims=True)
            
            # Check convergence
            pi_change = np.max(np.abs(new_pi - self.pi))
            A_change = np.max(np.abs(new_A - self.A))
            B_change = np.max(np.abs(new_B - self.B))
            
            self.pi = new_pi
            self.A = new_A
            self.B = new_B
            
            history['pi'].append(self.pi.copy())
            history['A'].append(self.A.copy())
            history['B'].append(self.B.copy())
            
            if pi_change < tolerance and A_change < tolerance and B_change < tolerance:
                break
        
        return self.pi, self.A, self.B, iteration + 1, history


def plot_transition_matrix(A, n_states):
    """Plot transition matrix as heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    states = [f"S{i}" for i in range(n_states)]
    
    im = ax.imshow(A, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    
    for i in range(n_states):
        for j in range(n_states):
            text = ax.text(j, i, f'{A[i, j]:.4f}',
                          ha="center", va="center", color="black" if A[i, j] < 0.5 else "white")
    
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('Transition Matrix (A)')
    plt.colorbar(im, ax=ax)
    
    return fig


def plot_emission_matrix(B, n_states, n_observations):
    """Plot emission matrix as heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    states = [f"S{i}" for i in range(n_states)]
    observations = [f"O{i}" for i in range(n_observations)]
    
    im = ax.imshow(B, cmap='Greens', aspect='auto')
    
    ax.set_xticks(range(n_observations))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(observations)
    ax.set_yticklabels(states)
    
    for i in range(n_states):
        for j in range(n_observations):
            text = ax.text(j, i, f'{B[i, j]:.4f}',
                          ha="center", va="center", color="black" if B[i, j] < 0.5 else "white")
    
    ax.set_xlabel('Observation')
    ax.set_ylabel('State')
    ax.set_title('Emission Matrix (B)')
    plt.colorbar(im, ax=ax)
    
    return fig


def plot_hmm_graph(n_states, A, B, pi, n_observations):
    """Plot HMM as a graph"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    G = nx.DiGraph()
    
    states = [f"S{i}" for i in range(n_states)]
    for state in states:
        G.add_node(state)
    
    pos = {}
    for i, state in enumerate(states):
        angle = 2 * np.pi * i / n_states - np.pi/2
        pos[state] = (0.5 + 0.35 * np.cos(angle), 0.5 + 0.35 * np.sin(angle))
    
    node_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:n_states]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if A[i, j] > 0.01:
                nx.draw_networkx_edges(G, pos, edgelist=[(from_state, to_state)],
                                       width=A[i, j] * 3, alpha=0.6, ax=ax)
    
    edge_labels = {}
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            edge_labels[(from_state, to_state)] = f"A: {A[i, j]:.3f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    
    observations = [f"O{i}" for i in range(n_observations)]
    emission_text = "Emission Probabilities:\n"
    for i, state in enumerate(states):
        emission_text += f"{state}: " + ", ".join([f"{observations[j]}:{B[i,j]:.2f}" for j in range(n_observations)]) + "\n"
    
    ax.annotate(emission_text, xy=(0.5, -0.1), xycoords='axes fraction',
                fontsize=10, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('HMM Structure', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return fig


def plot_convergence(history, n_states):
    """Plot convergence of parameters"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Skip initial state in history (which is before training)
    n_iterations = len(history['log_likelihood'])
    iterations = range(n_iterations)
    
    # Log likelihood
    if n_iterations > 0:
        axes[0].plot(iterations, history['log_likelihood'], 'b-o', markersize=4)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Log Likelihood')
        axes[0].set_title('Log Likelihood')
        axes[0].grid(True, alpha=0.3)
    
    # Initial probabilities - use only the values after each iteration
    pi_history = history['pi'][1:]  # Skip initial state
    for i in range(n_states):
        pi_values = [h[i] for h in pi_history]
        if len(pi_values) > 0:
            axes[1].plot(range(len(pi_values)), pi_values, '-o', markersize=3, label=f'S{i}')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Initial Probabilities (π)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Emission probabilities
    B_history = history['B'][1:]
    for i in range(n_states):
        B_values = [h[i, 0] for h in B_history]
        if len(B_values) > 0:
            axes[2].plot(range(len(B_values)), B_values, '-o', markersize=3, label=f'S{i}')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Probability')
    axes[2].set_title('Emission Probabilities')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_state_responsibilities(gamma, n_states, obs_sequence):
    """Plot state responsibilities over time"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    T = len(obs_sequence)
    x = range(T)
    
    bottom = np.zeros(T)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:n_states]
    
    states = [f"S{i}" for i in range(n_states)]
    for i, state in enumerate(states):
        ax.bar(x, gamma[:, i], bottom=bottom, label=state, color=colors[i], alpha=0.8)
        bottom += gamma[:, i]
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Probability')
    ax.set_title('State Responsibilities Over Time')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t+1}\n(O{obs})' for t, obs in enumerate(obs_sequence)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def main():
    st.markdown('<p class="main-header">🎯 HMM Baum-Welch Calculator</p>', 
                unsafe_allow_html=True)
    
    # Sidebar - Input Parameters
    st.sidebar.header("📥 Input Parameters")
    
    # Number of Hidden States
    n_states = st.sidebar.number_input("Number of Hidden States (N)", min_value=1, value=2, step=1)
    
    # Number of Observations
    n_observations = st.sidebar.number_input("Number of Observations (M)", min_value=1, value=2, step=1)
    
    # Observation Sequence
    st.sidebar.header("📝 Observation Sequence")
    st.sidebar.markdown("Enter observations as numbers (0, 1, 2, ...)")
    obs_input = st.sidebar.text_input("Observation sequence (comma-separated)", "0,1,0,1,1")
    
    # Algorithm Parameters
    st.sidebar.header("⚙️ Algorithm Parameters")
    max_iterations = st.sidebar.number_input("Max Iterations", min_value=1, value=100, step=10)
    tolerance = st.sidebar.number_input("Tolerance", min_value=1e-10, value=1e-6, format="%.0e")
    
    # Initial Parameters (optional)
    st.sidebar.header("📦 Initial Parameters (Optional)")
    use_custom_init = st.sidebar.checkbox("Use custom initial parameters", value=False)
    
    if n_states < 1 or n_observations < 1:
        st.error("Please enter valid number of states and observations.")
        return
    
    if not obs_input:
        st.error("Please enter an observation sequence.")
        return
    
    # Parse observation sequence as numbers
    try:
        obs_seq = [int(o.strip()) for o in obs_input.split(',')]
        # Validate observations are in range
        if any(o < 0 or o >= n_observations for o in obs_seq):
            st.error(f"Observations must be between 0 and {n_observations-1}")
            return
    except ValueError:
        st.error("Please enter observations as numbers separated by commas.")
        return
    
    # Initialize HMM
    hmm = HiddenMarkovModel(n_states, n_observations)
    
    # Set custom initial parameters if requested
    if use_custom_init:
        states = [f"S{i}" for i in range(n_states)]
        observations = [f"O{i}" for i in range(n_observations)]
        
        st.sidebar.markdown("### Initial Probabilities (π)")
        pi = []
        for i in range(n_states):
            val = st.sidebar.slider(f"π(S{i})", 0.0, 1.0, 1.0/n_states, 0.01)
            pi.append(val)
        pi = np.array(pi) / sum(pi)
        
        st.sidebar.markdown("### Transition Matrix (A)")
        A = np.zeros((n_states, n_states))
        for i in range(n_states):
            row_sum = 0
            for j in range(n_states):
                if j < n_states - 1:
                    val = st.sidebar.slider(f"A(S{i}→S{j})", 0.0, 1.0, 0.5, 0.01)
                    A[i, j] = val
                    row_sum += val
            A[i, -1] = max(0, 1 - row_sum)
        
        st.sidebar.markdown("### Emission Matrix (B)")
        B = np.zeros((n_states, n_observations))
        for i in range(n_states):
            row_sum = 0
            for j in range(n_observations):
                if j < n_observations - 1:
                    val = st.sidebar.slider(f"B(S{i}|O{j})", 0.0, 1.0, 0.5, 0.01)
                    B[i, j] = val
                    row_sum += val
            B[i, -1] = max(0, 1 - row_sum)
        
        A = A / A.sum(axis=1, keepdims=True)
        B = B / B.sum(axis=1, keepdims=True)
        hmm.set_parameters(pi, A, B)
    
    # Train Button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner('Training...'):
            new_pi, new_A, new_B, iterations, history = hmm.baum_welch(
                obs_seq, max_iterations=max_iterations, tolerance=tolerance
            )
        
        st.success(f"Training completed in {iterations} iterations!")
        
        # Display Results
        st.markdown("## 📊 Results")
        
        states = [f"S{i}" for i in range(n_states)]
        
        # Initial Probabilities
        st.markdown("### Initial Probabilities (π)")
        for i, state in enumerate(states):
            st.write(f"  {state}: **{new_pi[i]:.6f}**")
        
        # Transition Matrix
        st.markdown("### Transition Matrix (A)")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("From \\ To", " | ".join([f"{s:>8}" for s in states]))
            for i, state in enumerate(states):
                st.write(f"{state:>4}", " | ".join([f"{new_A[i,j]:>8.4f}" for j in range(n_states)]))
        with col2:
            fig = plot_transition_matrix(new_A, n_states)
            st.pyplot(fig)
        
        # Emission Matrix
        observations = [f"O{i}" for i in range(n_observations)]
        st.markdown("### Emission Matrix (B)")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("State \\ Obs", " | ".join([f"{o:>8}" for o in observations]))
            for i, state in enumerate(states):
                st.write(f"{state:>4}", " | ".join([f"{new_B[i,j]:>8.4f}" for j in range(n_observations)]))
        with col2:
            fig = plot_emission_matrix(new_B, n_states, n_observations)
            st.pyplot(fig)
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        # HMM Structure
        st.markdown("#### HMM Structure")
        fig = plot_hmm_graph(n_states, new_A, new_B, new_pi, n_observations)
        st.pyplot(fig)
        
        # State Responsibilities
        alpha, P = hmm.forward_algorithm(obs_seq)
        beta = hmm.backward_algorithm(obs_seq)
        gamma = hmm.compute_gamma(alpha, beta, P)
        
        st.markdown("#### State Responsibilities")
        fig = plot_state_responsibilities(gamma, n_states, obs_seq)
        st.pyplot(fig)
        
        # Convergence
        st.markdown("#### Convergence Progress")
        fig = plot_convergence(history, n_states)
        st.pyplot(fig)
        
        # Log likelihood
        st.markdown(f"**Final Log Likelihood**: {history['log_likelihood'][-1]:.6f}")
        st.markdown(f"**Probability P(O|λ)**: {np.exp(history['log_likelihood'][-1]):.8f}")


if __name__ == "__main__":
    main()
