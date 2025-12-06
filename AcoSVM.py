import pulp
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ==========================================
# DATASET: OR-Library P01 (Real Benchmark)
# ==========================================
weights = [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]
values = [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240]
capacity = 750
n = len(values)

print("--- Project: Optimization Comparison (MILP vs ACO vs Lagrange vs SVM) ---")
print(f"Dataset: P01 Benchmark | Items: {n} | Capacity: {capacity}")
print("-" * 65)

# ==========================================
# PART 1: Exact Method (MILP)
# ==========================================
print("\n[1] Running Exact Method (MILP)...")

def solve_milp(w, v, cap):
    num_items = len(v)
    prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(num_items), 0, 1, pulp.LpBinary)
    
    # Objective
    prob += pulp.lpSum([v[i] * x[i] for i in range(num_items)])
    # Constraint
    prob += pulp.lpSum([w[i] * x[i] for i in range(num_items)]) <= cap
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    selected = [i for i in range(num_items) if x[i].value() == 1]
    return pulp.value(prob.objective), selected

milp_score, milp_items = solve_milp(weights, values, capacity)
print(f" >> Optimal Solution (MILP): {milp_score}")


# ==========================================
# PART 2: Metaheuristic (ACO)
# ==========================================
print("\n[2] Running Ant Colony Optimization (ACO)...")

n_ants = 20
n_iterations = 50
pheromone = [1.0] * n
alpha, beta, rho, Q = 1.0, 2.0, 0.5, 100
eta = [values[i]/weights[i] for i in range(n)]
best_aco_val = 0

for _ in range(n_iterations):
    iter_sols = []
    for _ in range(n_ants):
        bag = []
        curr_w = 0
        curr_v = 0
        candidates = list(range(n))
        
        while candidates:
            feasible = [i for i in candidates if curr_w + weights[i] <= capacity]
            if not feasible:
                break
            
            probs = [(pheromone[i]**alpha)*(eta[i]**beta) for i in feasible]
            
            # Roulette Wheel Selection
            if sum(probs) == 0:
                selected = random.choice(feasible)
            else:
                selected = random.choices(feasible, weights=probs, k=1)[0]
            
            bag.append(selected)
            curr_w += weights[selected]
            curr_v += values[selected]
            candidates.remove(selected)
        
        if curr_v > best_aco_val:
            best_aco_val = curr_v
        iter_sols.append((bag, curr_v))
        
    # Pheromone Update
    for i in range(n):
        pheromone[i] *= (1 - rho)
    for bag, val in iter_sols:
        delta = val / Q
        for item in bag:
            pheromone[item] += delta

print(f" >> ACO Result: {best_aco_val}")


# ==========================================
# PART 3: Mathematical Bound (Lagrangian)
# ==========================================
print("\n[3] Calculating Lagrangian Upper Bound...")

def solve_lagrangian(w, v, cap):
    lamb = 0.0
    best_bound = float('inf')
    num_items = len(v)
    
    for k in range(200):
        # Solve Relaxed Problem
        curr_val = 0
        sum_w_relaxed = 0
        for i in range(num_items):
            reduced_cost = v[i] - lamb * w[i]
            if reduced_cost > 0:
                curr_val += reduced_cost
                sum_w_relaxed += w[i]
        
        L_value = curr_val + lamb * cap
        if L_value < best_bound:
            best_bound = L_value
            
        # Update Lambda (Subgradient)
        gradient = cap - sum_w_relaxed
        step_size = 1.0 / (k + 1)
        lamb = max(0, lamb - step_size * gradient)
        
    return best_bound

lagrange_bound = solve_lagrangian(weights, values, capacity)
print(f" >> Lagrangian Upper Bound: {lagrange_bound:.2f}")


# ==========================================
# PART 4: Machine Learning (SVM)
# ==========================================
print("\n[4] Training SVM (Sim-to-Real Transfer)...")

# 4.1 Generate Simulated Training Data
print("   - Generating synthetic training data...")
X_train = []
y_train = []

for _ in range(200): # 200 random scenarios
    # Generate random problem similar to P01 scale
    w_sim = [random.randint(50, 130) for _ in range(15)]
    v_sim = [random.randint(100, 250) for _ in range(15)]
    c_sim = random.randint(500, 800)
    
    # Get labels using MILP
    _, selected_idx = solve_milp(w_sim, v_sim, c_sim)
    
    for i in range(15):
        # Features: [Weight, Value, Ratio, NormWeight]
        features = [w_sim[i], v_sim[i], v_sim[i]/w_sim[i], w_sim[i]/c_sim]
        label = 1 if i in selected_idx else 0
        X_train.append(features)
        y_train.append(label)

# 4.2 Train SVM
print("   - Training SVC model...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

svm_model = SVC(kernel='rbf', C=10.0)
svm_model.fit(X_train_scaled, y_train)

# 4.3 Test on Real Data (P01)
print("   - Testing on real dataset...")
X_test = []
for i in range(n):
    X_test.append([weights[i], values[i], values[i]/weights[i], weights[i]/capacity])

X_test_scaled = scaler.transform(X_test)
predictions = svm_model.predict(X_test_scaled)

# Construct SVM Solution
svm_val = 0
svm_w = 0
# Add items predicted as '1' if they fit
for i in range(n):
    if predictions[i] == 1:
        if svm_w + weights[i] <= capacity:
            svm_w += weights[i]
            svm_val += values[i]

print(f" >> SVM Predicted Score: {svm_val}")


# ==========================================
# PART 5: Final Comparison
# ==========================================
print("\n" + "="*60)
print(f"{'METHOD':<20} | {'SCORE':<10} | {'GAP TO OPTIMAL'}")
print("-" * 60)
print(f"{'Exact (MILP)':<20} | {milp_score:<10} | 0.0 %")
print(f"{'ACO (Heuristic)':<20} | {best_aco_val:<10} | {((milp_score-best_aco_val)/milp_score)*100:.2f} %")
print(f"{'SVM (ML)':<20} | {svm_val:<10} | {((milp_score-svm_val)/milp_score)*100:.2f} %")
print(f"{'Lagrange Bound':<20} | {lagrange_bound:.2f}      | +{((lagrange_bound-milp_score)/milp_score)*100:.2f} % (Theoretical)")
print("="*60)
