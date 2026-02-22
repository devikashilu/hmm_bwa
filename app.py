"""
Hidden Markov Model - Baum-Welch Algorithm Implementation
Visual Interactive Application
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="HMM Baum-Welch Algorithm",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .formula-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .result-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .step-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .info-text {
        font-size: 1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


class HiddenMarkovModel:
    """Hidden Markov Model with Baum-Welch Algorithm implementation"""
    
    def __init__(self, states, observations):
        """
        Initialize HMM
        
        Args:
            states: List of state names
            observations: List of observation symbols
        """
        self.states = states
        self.observations = observations
        self.N = len(states)
        self.M = len(observations)
        
        # Initialize parameters randomly
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize with uniform probabilities"""
        # Initial state probabilities
        self.pi = np.ones(self.N) / self.N
        
        # Transition probability matrix
        self.A = np.ones((self.N, self.N)) / self.N
        
        # Emission probability matrix
        self.B = np.ones((self.N, self.M)) / self.M
    
    def set_parameters(self, pi, A, B):
        """Set HMM parameters"""
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.B = np.array(B)
    
    def observation_to_index(self, obs):
        """Convert observation to index"""
        return [self.observations.index(o) for o in obs]
    
    def forward_algorithm(self, obs_sequence):
        """
        Forward algorithm implementation
        
        Args:
            obs_sequence: List of observations
            
        Returns:
            alpha: Forward probabilities matrix (T x N)
            P: Probability of the observation sequence
        """
        T = len(obs_sequence)
        obs_idx = self.observation_to_index(obs_sequence)
        
        # Initialize alpha
        alpha = np.zeros((T, self.N))
        
        # Initialization: alpha_1(i) = pi_i * b_i(O_1)
        alpha[0] = self.pi * self.B[:, obs_idx[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, obs_idx[t]]
        
        # Termination
        P = np.sum(alpha[T-1])
        
        return alpha, P
    
    def backward_algorithm(self, obs_sequence):
        """
        Backward algorithm implementation
        
        Args:
            obs_sequence: List of observations
            
        Returns:
            beta: Backward probabilities matrix (T x N)
        """
        T = len(obs_sequence)
        obs_idx = self.observation_to_index(obs_sequence)
        
        # Initialize beta
        beta = np.zeros((T, self.N))
        
        # Initialization: beta_T(i) = 1
        beta[T-1] = np.ones(self.N)
        
        # Recursion (going backwards)
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_idx[t+1]] * beta[t+1])
        
        return beta
    
    def compute_gamma(self, alpha, beta, P):
        """
        Compute state responsibilities (gamma)
        
        gamma_t(i) = P(q_t = i | O, lambda) = alpha_t(i) * beta_t(i) / P(O)
        """
        if P == 0:
            return np.zeros_like(alpha)
        return (alpha * beta) / P
    
    def compute_xi(self, alpha, beta, A, B, obs_sequence, P):
        """
        Compute transition responsibilities (xi)
        
        xi_t(i,j) = alpha_t(i) * a_ij * b_j(O_{t+1}) * beta_{t+1}(j) / P(O)
        """
        T = len(obs_sequence)
        obs_idx = self.observation_to_index(obs_sequence)
        xi = np.zeros((T-1, self.N, self.N))
        
        for t in range(T-1):
            for i in range(self.N):
                for j in range(self.N):
                    xi[t, i, j] = (alpha[t, i] * A[i, j] * 
                                   B[j, obs_idx[t+1]] * beta[t+1, j])
        
        if P > 0:
            xi = xi / P
        
        return xi
    
    def baum_welch(self, obs_sequence, max_iterations=100, tolerance=1e-6, verbose=True):
        """
        Baum-Welch algorithm implementation
        
        Args:
            obs_sequence: List of observations
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Whether to print progress
            
        Returns:
            pi, A, B: Updated parameters
            iterations: Number of iterations
            history: History of parameter changes
        """
        history = {
            'pi': [self.pi.copy()],
            'A': [self.A.copy()],
            'B': [self.B.copy()],
            'log_likelihood': []
        }
        
        for iteration in range(max_iterations):
            # E-step: Forward and Backward
            alpha, P = self.forward_algorithm(obs_sequence)
            beta = self.backward_algorithm(obs_sequence)
            
            if P == 0:
                if verbose:
                    st.error("Probability zero - algorithm cannot continue")
                break
            
            history['log_likelihood'].append(np.log(P) if P > 0 else -np.inf)
            
            # Compute auxiliary variables
            gamma = self.compute_gamma(alpha, beta, P)
            xi = self.compute_xi(alpha, beta, self.A, self.B, obs_sequence, P)
            
            # M-step: Update parameters
            
            # Update initial probabilities
            new_pi = gamma[0]
            
            # Update transition probabilities
            # a_ij = sum(xi_t(i,j)) / sum(gamma_t(i))
            new_A = np.zeros((self.N, self.N))
            for i in range(self.N):
                gamma_sum = np.sum(gamma[:-1, i])  # Sum over t=1 to T-1
                if gamma_sum > 0:
                    new_A[i, :] = np.sum(xi[:, i, :], axis=0) / gamma_sum
            
            # Update emission probabilities
            # b_j(o) = sum(gamma_t(j) for t where O_t = o) / sum(gamma_t(j))
            new_B = np.zeros((self.N, self.M))
            obs_idx = self.observation_to_index(obs_sequence)
            for j in range(self.N):
                for o_idx in range(self.M):
                    # Sum gamma where observation equals o_idx
                    mask = np.array(obs_idx) == o_idx
                    if np.sum(mask) > 0:
                        new_B[j, o_idx] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])
            
            # Normalize to ensure valid probabilities
            new_pi = new_pi / np.sum(new_pi)
            new_A = new_A / np.sum(new_A, axis=1, keepdims=True)
            new_B = new_B / np.sum(new_B, axis=1, keepdims=True)
            
            # Check convergence
            pi_change = np.max(np.abs(new_pi - self.pi))
            A_change = np.max(np.abs(new_A - self.A))
            B_change = np.max(np.abs(new_B - self.B))
            
            # Update parameters
            self.pi = new_pi
            self.A = new_A
            self.B = new_B
            
            history['pi'].append(self.pi.copy())
            history['A'].append(self.A.copy())
            history['B'].append(self.B.copy())
            
            if verbose and iteration % 5 == 0:
                st.info(f"Iteration {iteration + 1}: max changes - pi: {pi_change:.6f}, A: {A_change:.6f}, B: {B_change:.6f}")
            
            if pi_change < tolerance and A_change < tolerance and B_change < tolerance:
                if verbose:
                    st.success(f"Converged after {iteration + 1} iterations!")
                break
        
        return self.pi, self.A, self.B, iteration + 1, history


def visualize_hmm(states, A, B, pi):
    """Visualize HMM as a graph"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes for states
    for i, state in enumerate(states):
        G.add_node(state, pos=(0.3 if i %0 else 0.7, 0.5 if i < 2 ==  len(states)/2 else 0.5))
    
    # Get positions
    pos = {}
    n = len(states)
    for i, state in enumerate(states):
        angle = 2 * np.pi * i / n - np.pi/2
        pos[state] = (0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle))
    
    # Draw nodes
    node_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:n]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', ax=ax)
    
    # Draw edges (transitions)
    edge_labels = {}
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if A[i, j] > 0.05:  # Only show significant transitions
                edge_labels[(from_state, to_state)] = f"{A[i, j]:.2f}"
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                           arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax)
    
    # Add initial state indicator
    ax.annotate('Initial: ' + ', '.join([f"{s}:{p:.2f}" for s, p in zip(states, pi)]),
                xy=(0.5, 0.95), xycoords='axes fraction',
                fontsize=12, ha='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('Hidden Markov Model Structure', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    return fig


def visualize_iteration(states, history, iteration):
    """Visualize parameter changes over iterations"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    iterations = range(len(history['log_likelihood']))
    
    # Plot log likelihood
    axes[0].plot(iterations, history['log_likelihood'], 'b-o', markersize=4)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Log Likelihood')
    axes[0].set_title('Log Likelihood Progress')
    axes[0].grid(True, alpha=0.3)
    
    # Plot initial probabilities
    for i, state in enumerate(states):
        pi_values = [h[i] for h in history['pi']]
        axes[1].plot(iterations, pi_values, '-o', markersize=3, label=state)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Initial Probabilities')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot emission probabilities (for first observation)
    for i, state in enumerate(states):
        B_values = [h[i, 0] for h in history['B']]
        axes[2].plot(iterations, B_values, '-o', markersize=3, label=f'{state}: obs1')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Probability')
    axes[2].set_title('Emission Probabilities')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_gamma(states, gamma, observations):
    """Visualize state responsibilities over time"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    T = len(observations)
    x = range(T)
    
    # Stacked bar chart
    bottom = np.zeros(T)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:len(states)]
    
    for i, state in enumerate(states):
        ax.bar(x, gamma[:, i], bottom=bottom, label=state, color=colors[i], alpha=0.8)
        bottom += gamma[:, i]
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Probability')
    ax.set_title('State Responsibilities Over Time')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t+1}\n({obs})' for t, obs in enumerate(observations)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def main():
    """Main application"""
    
    # Title
    st.markdown('<p class="main-header">🎯 Baum-Welch Algorithm for Hidden Markov Models</p>', 
                unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("📚 Navigation")
    section = st.sidebar.radio("Go to", 
        ["Introduction", "HMM Theory", "Numerical Example (Short)", 
         "Numerical Example (Long)", "Custom Demo"])
    
    if section == "Introduction":
        show_introduction()
    elif section == "HMM Theory":
        show_hmm_theory()
    elif section == "Numerical Example (Short)":
        show_short_example()
    elif section == "Numerical Example (Long)":
        show_long_example()
    elif section == "Custom Demo":
        show_custom_demo()


def show_introduction():
    """Introduction section"""
    st.markdown("## 📖 Introduction to Baum-Welch Algorithm")
    
    st.markdown("""
    ### What is Hidden Markov Model (HMM)?
    
    Hidden Markov Models are **probabilistic models** for sequential data where the system
    evolves through a sequence of **hidden states** that cannot be directly observed. 
    Instead, each hidden state probabilistically generates an **observable output**.
    
    ### What is Baum-Welch Algorithm?
    
    The Baum-Welch algorithm is an **Expectation-Maximization (EM) algorithm** used to learn
    HMM parameters when only the observation sequences are available. It finds the 
    maximum likelihood estimate of the HMM parameters.
    
    ### Key Components of HMM
    
    | Component | Symbol | Description |
    |-----------|--------|-------------|
    | States | Q | Hidden states that the system can be in |
    | Observations | O | Observable symbols emitted by states |
    | Initial Probabilities | π | Probability of starting in each state |
    | Transition Probabilities | A | Probability of moving between states |
    | Emission Probabilities | B | Probability of emitting observations |
    
    ### The Learning Problem
    
    Given only observation sequences, Baum-Welch learns:
    - **π**: Initial state probabilities
    - **A**: State transition matrix
    - **B**: Emission probabilities
    
    Use the sidebar to explore different sections!
    """)
    
    # Visual diagram
    st.markdown("### 🔄 How Baum-Welch Works")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**1. E-Step**\n\nCompute forward and backward probabilities to estimate state responsibilities")
    with col2:
        st.info("**2. M-Step**\n\nUpdate parameters based on expected counts from E-step")
    with col3:
        st.info("**3. Repeat**\n\nIterate until parameters converge")
    
    st.markdown("---")
    st.markdown("### 🎯 Interactive Demo")
    st.markdown("Navigate to **Custom Demo** in the sidebar to create your own HMM and see Baum-Welch in action!")


def show_hmm_theory():
    """HMM Theory section"""
    st.markdown("## 📚 Hidden Markov Model Theory")
    
    # Forward Probability
    st.markdown("### 🔹 Forward Probability (Prefix Evidence)")
    
    st.markdown(r"""
    $$\alpha_t(i) = P(O_1, O_2, \ldots, O_t, q_t = i | \lambda)$$
    
    **Semantic meaning:** The probability that the first t observations have occurred 
    and the system is currently in state i.
    
    **Initialization:**
    $$\alpha_1(i) = \pi_i \cdot b_i(O_1)$$
    
    **Recursion:**
    $$\alpha_{t+1}(j) = \sum_{i=1}^{N} \alpha_t(i) \cdot a_{ij} \cdot b_j(O_{t+1})$$
    """)
    
    # Backward Probability
    st.markdown("### 🔹 Backward Probability (Suffix Evidence)")
    
    st.markdown(r"""
    $$\beta_t(i) = P(O_{t+1}, O_{t+2}, \ldots, O_T | q_t = i, \lambda)$$
    
    **Semantic meaning:** If the system were in state i at time t, how well could it 
    explain all future observations?
    
    **Initialization:**
    $$\beta_T(i) = 1$$
    
    **Recursion:**
    $$\beta_t(i) = \sum_{j=1}^{N} a_{ij} \cdot b_j(O_{t+1}) \cdot \beta_{t+1}(j)$$
    """)
    
    # Auxiliary Variables
    st.markdown("### 🔹 Auxiliary Variables (Soft Credit Assignment)")
    
    st.markdown(r"""
    **State Responsibility:**
    $$\gamma_t(i) = P(q_t = i | O, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(O)}$$
    
    Meaning: Fraction of belief that the system was in state i at time t.
    
    **Transition Responsibility:**
    $$\xi_t(i, j) = \frac{\alpha_t(i) a_{ij} b_j(O_{t+1}) \beta_{t+1}(j)}{P(O)}$$
    
    Meaning: Fraction of probability flow through transition i → j.
    """)
    
    # Parameter Updates
    st.markdown("### 🔹 Parameter Update Equations")
    
    st.markdown(r"""
    **Initial Probabilities:**
    $$\pi_i^{new} = \gamma_1(i)$$
    
    **Transition Probabilities:**
    $$a_{ij}^{new} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$
    
    **Emission Probabilities:**
    $$b_j(o)^{new} = \frac{\sum_{t: O_t = o} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}$$
    """)


def show_short_example():
    """Short observation sequence example from tutorial"""
    st.markdown("## 📊 Numerical Example: Short Sequence")
    st.markdown("### Observation Sequence: (Walk, Shop)")
    
    # Define the example
    states = ['Rainy', 'Sunny']
    observations = ['Walk', 'Shop']
    
    # Initial parameters from tutorial
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.9], [0.6, 0.4]])
    
    # Display initial parameters
    st.markdown("### Initial Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Initial Probabilities (π)**")
        st.write(f"Rainy: {pi[0]}")
        st.write(f"Sunny: {pi[1]}")
    with col2:
        st.markdown("**Transition Matrix (A)**")
        st.write("From Rainy:", A[0])
        st.write("From Sunny:", A[1])
    with col3:
        st.markdown("**Emission Matrix (B)**")
        st.write("Rainy:", B[0])
        st.write("Sunny:", B[1])
    
    # Create HMM and run
    hmm = HiddenMarkovModel(states, observations)
    hmm.set_parameters(pi.copy(), A.copy(), B.copy())
    
    obs_seq = ['Walk', 'Shop']
    
    # Compute forward probabilities
    alpha, P = hmm.forward_algorithm(obs_seq)
    
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### Step 1: Forward Computation")
    st.markdown(f"**α₁(Rainy)** = π_Rainy × b_Rainy(Walk) = 0.6 × 0.1 = **0.06**")
    st.markdown(f"**α₁(Sunny)** = π_Sunny × b_Sunny(Walk) = 0.4 × 0.6 = **0.24**")
    st.markdown(f"**α₂(Rainy)** = (0.06×0.7 + 0.24×0.4) × 0.9 = 0.1242")
    st.markdown(f"**α₂(Sunny)** = (0.06×0.3 + 0.24×0.6) × 0.4 = 0.0648")
    st.markdown(f"**P(O)** = α₂(Rainy) + α₂(Sunny) = **0.189**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Compute backward probabilities
    beta = hmm.backward_algorithm(obs_seq)
    
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("### Step 2: Backward Computation")
    st.markdown("**β₂(Rainy)** = β₂(Sunny) = 1 (initialization)")
    st.markdown(f"**β₁(Rainy)** = 0.7×0.9×1 + 0.3×0.4×1 = **0.75**")
    st.markdown(f"**β₁(Sunny)** = 0.4×0.9×1 + 0.6×0.4×1 = **0.60**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Compute state responsibilities
    gamma = hmm.compute_gamma(alpha, beta, P)
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("### State Responsibilities")
    st.markdown(f"**γ₁(Rainy)** = 0.06×0.75/0.189 = **{0.06*0.75/0.189:.3f}**")
    st.markdown(f"**γ₁(Sunny)** = 0.24×0.60/0.189 = **{0.24*0.60/0.189:.3f}**")
    st.markdown(f"**γ₂(Rainy)** = 0.1242×1/0.189 = **{0.1242/0.189:.3f}**")
    st.markdown(f"**γ₂(Sunny)** = 0.0648×1/0.189 = **{0.0648/0.189:.3f}**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualize
    st.markdown("### Visual Representation")
    
    fig = visualize_gamma(states, gamma, obs_seq)
    st.pyplot(fig)
    
    # Updated parameters
    st.markdown("### Updated Emission Probabilities")
    st.markdown("From the tutorial:")
    st.markdown("b_Rainy(Walk) = γ₁(Rainy) / (γ₁(Rainy) + γ₂(Rainy)) = 0.238 / 0.895 = **0.266**")
    st.markdown("b_Rainy(Shop) = 1 - 0.266 = **0.734**")
    st.markdown("b_Sunny(Walk) = 0.689, b_Sunny(Shop) = 0.311")
    
    st.info("Note: With limited data, estimates change sharply. The algorithm needs more observations for stable estimates.")


def show_long_example():
    """Long observation sequence example from tutorial"""
    st.markdown("## 📊 Numerical Example: Long Sequence")
    st.markdown("### Observation Sequence: (Walk, Shop, Shop, Walk, Shop)")
    
    # Define the example
    states = ['Rainy', 'Sunny']
    observations = ['Walk', 'Shop']
    
    # Initial parameters
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.9], [0.6, 0.4]])
    
    # Create HMM and run
    hmm = HiddenMarkovModel(states, observations)
    hmm.set_parameters(pi.copy(), A.copy(), B.copy())
    
    obs_seq = ['Walk', 'Shop', 'Shop', 'Walk', 'Shop']
    
    # Run Baum-Welch
    with st.spinner('Running Baum-Welch algorithm...'):
        new_pi, new_A, new_B, iterations, history = hmm.baum_welch(
            obs_seq, max_iterations=50, verbose=False
        )
    
    # Display results
    st.success(f"Converged in {iterations} iterations!")
    
    st.markdown("### Initial vs Final Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Initial Parameters**")
        st.write(f"π: {pi}")
        st.write("A:", A)
        st.write("B:", B)
    
    with col2:
        st.markdown("**After Learning**")
        st.write(f"π: {new_pi.round(3)}")
        st.write("A:", new_A.round(3))
        st.write("B:", new_B.round(3))
    
    # State responsibilities
    alpha, P = hmm.forward_algorithm(obs_seq)
    beta = hmm.backward_algorithm(obs_seq)
    gamma = hmm.compute_gamma(alpha, beta, P)
    
    st.markdown("### State Responsibilities")
    
    # Create table
    resp_data = {
        'Position': list(range(1, 6)),
        'Observation': obs_seq,
        'γ(Rainy)': gamma[:, 0].round(3),
        'γ(Sunny)': gamma[:, 1].round(3)
    }
    st.table(resp_data)
    
    # Visualize
    st.markdown("### Visual: State Responsibilities Over Time")
    fig = visualize_gamma(states, gamma, obs_seq)
    st.pyplot(fig)
    
    # Convergence plot
    st.markdown("### Visual: Convergence Progress")
    fig2 = visualize_iteration(states, history, iterations)
    st.pyplot(fig2)
    
    st.markdown("""
    ### Key Observations
    
    1. **More observations = better estimates**: With longer sequences, the algorithm has more data to learn from
    
    2. **Responsibilities sum to 1**: At each time step, γ_Rainy + γ_Sunny = 1
    
    3. **Patterns emerge**: The algorithm identifies that Rainy state is more likely when 'Shop' is observed
    
    4. **Convergence**: Parameters stabilize as iterations increase
    """)


def show_custom_demo():
    """Custom demonstration"""
    st.markdown("## 🔧 Custom HMM Demo")
    
    # Sidebar for configuration
    st.sidebar.markdown("### HMM Configuration")
    
    # Define states and observations
    states_input = st.sidebar.text_input("States (comma-separated)", "Rainy,Sunny")
    observations_input = st.sidebar.text_input("Observations (comma-separated)", "Walk,Shop")
    
    if states_input and observations_input:
        states = [s.strip() for s in states_input.split(',')]
        observations = [o.strip() for o in observations_input.split(',')]
        
        st.sidebar.markdown("### Initial Parameters")
        
        # Initial probabilities
        st.sidebar.markdown("**Initial Probabilities**")
        pi = []
        for i, state in enumerate(states):
            val = st.sidebar.slider(f"π({state})", 0.0, 1.0, 1.0/len(states), 0.01)
            pi.append(val)
        pi = np.array(pi) / sum(pi)  # Normalize
        
        # Transition matrix
        st.sidebar.markdown("**Transition Probabilities**")
        A = np.zeros((len(states), len(states)))
        for i, from_state in enumerate(states):
            row_sum = 0
            for j, to_state in enumerate(states):
                if j < len(states) - 1:
                    val = st.sidebar.slider(f"A({from_state}→{to_state})", 0.0, 1.0, 0.5, 0.01)
                    A[i, j] = val
                    row_sum += val
            A[i, -1] = 1 - row_sum  # Ensure row sums to 1
        
        # Emission matrix
        st.sidebar.markdown("**Emission Probabilities**")
        B = np.zeros((len(states), len(observations)))
        for i, state in enumerate(states):
            row_sum = 0
            for j, obs in enumerate(observations):
                if j < len(observations) - 1:
                    val = st.sidebar.slider(f"B({state}|{obs})", 0.0, 1.0, 0.5, 0.01)
                    B[i, j] = val
                    row_sum += val
            B[i, -1] = 1 - row_sum
        
        # Normalize matrices
        A = A / A.sum(axis=1, keepdims=True)
        B = B / B.sum(axis=1, keepdims=True)
        
        # Observation sequence input
        st.markdown("### Observation Sequence")
        obs_input = st.text_input("Enter observations (comma-separated)", "Walk,Shop,Walk,Shop")
        
        if obs_input:
            obs_seq = [o.strip() for o in obs_input.split(',')]
            
            # Validate observations
            valid_obs = all(o in observations for o in obs_seq)
            
            if not valid_obs:
                st.error(f"Invalid observations! Available: {observations}")
            else:
                # Create and run HMM
                hmm = HiddenMarkovModel(states, observations)
                hmm.set_parameters(pi.copy(), A.copy(), B.copy())
                
                # Display current model
                st.markdown("### Current HMM Model")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Initial Probabilities (π)**")
                    for i, state in enumerate(states):
                        st.write(f"  {state}: {pi[i]:.3f}")
                    
                    st.markdown("**Transition Matrix (A)**")
                    st.write("  ", "  ".join([f"{s:>8}" for s in states]))
                    for i, state in enumerate(states):
                        st.write(f"  {state}: " + "  ".join([f"{A[i,j]:>8.3f}" for j in range(len(states))]))
                
                with col2:
                    st.markdown("**Emission Matrix (B)**")
                    st.write("  ", "  ".join([f"{o:>8}" for o in observations]))
                    for i, state in enumerate(states):
                        st.write(f"  {state}: " + "  ".join([f"{B[i,j]:>8.3f}" for j in range(len(observations))]))
                
                # Run Baum-Welch
                if st.button("🚀 Run Baum-Welch Algorithm"):
                    with st.spinner('Learning parameters...'):
                        new_pi, new_A, new_B, iterations, history = hmm.baum_welch(
                            obs_seq, max_iterations=100, verbose=False
                        )
                    
                    st.success(f"Converged in {iterations} iterations!")
                    
                    # Show results
                    st.markdown("### Learned Parameters")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Initial Probabilities (π)**")
                        for i, state in enumerate(states):
                            st.write(f"  {state}: {new_pi[i]:.4f}")
                        
                        st.markdown("**Transition Matrix (A)**")
                        st.write("  ", "  ".join([f"{s:>8}" for s in states]))
                        for i, state in enumerate(states):
                            st.write(f"  {state}: " + "  ".join([f"{new_A[i,j]:>8.4f}" for j in range(len(states))]))
                    
                    with col2:
                        st.markdown("**Emission Matrix (B)**")
                        st.write("  ", "  ".join([f"{o:>8}" for o in observations]))
                        for i, state in enumerate(states):
                            st.write(f"  {state}: " + "  ".join([f"{new_B[i,j]:>8.4f}" for j in range(len(observations))]))
                    
                    # Visualizations
                    st.markdown("### Visualizations")
                    
                    # Model structure
                    st.markdown("#### HMM Structure")
                    fig = visualize_hmm(states, new_A, new_B, new_pi)
                    st.pyplot(fig)
                    
                    # State responsibilities
                    alpha, P = hmm.forward_algorithm(obs_seq)
                    beta = hmm.backward_algorithm(obs_seq)
                    gamma = hmm.compute_gamma(alpha, beta, P)
                    
                    st.markdown("#### State Responsibilities")
                    fig2 = visualize_gamma(states, gamma, obs_seq)
                    st.pyplot(fig2)
                    
                    # Convergence
                    st.markdown("#### Convergence Progress")
                    fig3 = visualize_iteration(states, history, iterations)
                    st.pyplot(fig3)
                    
                    # Interpretation
                    st.markdown("### Interpretation")
                    st.markdown(f"""
                    - **Most likely initial state**: {states[np.argmax(new_pi)]}
                    - **Most common transition**: {states[np.argmax(new_A.sum(axis=1))]} → {states[np.argmax(new_A.max(axis=1))]}
                    - **Observation likelihood**: P(O|λ) = {P:.6f}
                    """)


if __name__ == "__main__":
    main()
