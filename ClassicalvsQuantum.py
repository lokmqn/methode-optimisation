import time
import random
import sklearn.utils
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals

# ==========================================
# PART 4: Classical vs Quantum SVM (The Showdown)
# ==========================================
print("\n[4] Classical SVM vs. Quantum SVM (QSVM)...")

# 4.1 Data Preparation (4 Features -> 4 Qubits)
print("   - Preparing Data (4 Features -> 4 Qubits)...")
X_train_q, y_train_q = [], []

# Reduce training data size for faster quantum simulation
# Generating synthetic training data
for _ in range(50): 
    w_sim = [random.randint(50, 130) for _ in range(15)]
    v_sim = [random.randint(100, 250) for _ in range(15)]
    c_sim = random.randint(500, 800)
    # Using the previously defined solve_milp function
    _, selected_idx = solve_milp(w_sim, v_sim, c_sim)
    for i in range(15):
        # 4 Features = 4 Qubits
        features = [w_sim[i], v_sim[i], v_sim[i]/w_sim[i], w_sim[i]/c_sim]
        label = 1 if i in selected_idx else 0
        X_train_q.append(features)
        y_train_q.append(label)

# Downsample to 100 samples for speed (Simulation is slow)
X_train_q, y_train_q = sklearn.utils.resample(X_train_q, y_train_q, n_samples=100, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_q)

# Prepare Test Data (Real P01 Dataset)
X_test = []
for i in range(n):
    X_test.append([weights[i], values[i], values[i]/weights[i], weights[i]/capacity])
X_test_scaled = scaler.transform(X_test)


# --- [A] Classical SVM ---
print("\n   [A] Training Classical SVM...")
start_time = time.time()
svm_classic = SVC(kernel='rbf', C=10.0)
svm_classic.fit(X_train_scaled, y_train_q)
classic_time = time.time() - start_time
print(f"       -> Training Time: {classic_time:.4f} seconds")

# Classical Prediction
classic_preds = svm_classic.predict(X_test_scaled)
classic_score = 0
w_c = 0
for i in range(n):
    if classic_preds[i] == 1 and w_c + weights[i] <= capacity:
        w_c += weights[i]
        classic_score += values[i]
print(f"       -> Classical SVM Score: {classic_score}")


# --- [B] Quantum SVM (QSVM) ---
print("\n   [B] Training Quantum SVM (QSVC)...")
algorithm_globals.random_seed = 12345

# 1. Quantum Feature Map (ZZFeatureMap)
# Transforms classical data into quantum states (4 Qubits)
num_qubits = 4 
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')

# 2. Quantum Kernel
q_kernel = FidelityQuantumKernel(feature_map=feature_map)

# 3. Initialize and Train QSVC
qsvc = QSVC(quantum_kernel=q_kernel)

start_time_q = time.time()
qsvc.fit(X_train_scaled, y_train_q)
quantum_time = time.time() - start_time_q
print(f"       -> Training Time: {quantum_time:.4f} seconds")

# Quantum Prediction
print("       -> Predicting with Quantum Circuit...")
quantum_preds = qsvc.predict(X_test_scaled)
quantum_score = 0
w_q = 0
for i in range(n):
    if quantum_preds[i] == 1 and w_q + weights[i] <= capacity:
        w_q += weights[i]
        quantum_score += values[i]
print(f"       -> Quantum SVM Score: {quantum_score}")


# ==========================================
# Final Comparison Update
# ==========================================
print("\n" + "="*70)
print(f"{'METHOD':<25} | {'SCORE':<10} | {'TIME (Train)':<15} | {'ACCURACY GAP'}")
print("-" * 70)
print(f"{'Classical SVM':<25} | {classic_score:<10} | {classic_time:.4f} s        | {((milp_score-classic_score)/milp_score)*100:.1f}%")
print(f"{'Quantum SVM (QSVM)':<25} | {quantum_score:<10} | {quantum_time:.4f} s        | {((milp_score-quantum_score)/milp_score)*100:.1f}%")
print("="*70)
