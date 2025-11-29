import pulp
import random

poid = [2,3,6,7,5,9,4]
value = [10,7,25,24,15,30,9]
max_weight = 15
n=len(value)

print(f'Number of items: {n} and max weight: {max_weight}')

prob=pulp.LpProblem("Knapsack",pulp.LpMaximize)
x=[pulp.LpVariable(f'x{i}',0,1,pulp.LpInteger) for i in range(n)]
prob+=pulp.lpSum([value[i]*x[i] for i in range(n)]) , "total_value"

#contraints
prob+=pulp.lpSum([poid[i]*x[i] for i in range(n)]) <= max_weight

prob.solve()
print("Status:",pulp.LpStatus[prob.status])
print("Best value for Z ",pulp.value(prob.objective))
print("Selected items:")
items_choisies= []
poid_total = 0
for i in range(n):
    if x[i].varValue==1:
        print(f'Item {i+1} with weight {poid[i]} and value {value[i]}')
        items_choisies.append(i)
        poid_total += poid[i]
        
print(f"Total weight: {poid_total}")


print("\n start aco algorithm")

n_ants = 10
n_iterations = 100
alpha = 1
beta = 2
rho = 0.1
Q = 100

pheromone = [1]*n

eta = []
for i in range(n):
    densite = value[i]/(poid[i] + 0.000001)
    eta.append(densite)

print("phéromone is ready: ", pheromone)  
print(f"phéromone intial {pheromone}")
print(f"Heuristic (Eta) :{eta}")


#prrinciple llop

best_solution_globale = None
best_value_globale = 0

print("\n start searching :")

for iteration in range(n_iterations):
    solutions_in_generation = []

    for ant in range(n_ants):
        current_bag = []
        current_weight = 0
        current_value = 0

        candidate_list = list(range(n))

        while candidate_list:
            feasible_candidates = [i for i in candidate_list if current_weight + poid[i] <= max_weight]
            if not feasible_candidates:
                break #stop 


     #probabilty calculation

            probabilities = []
            for i in feasible_candidates:
                prob = pheromone[i]**alpha * eta[i]**beta
                probabilities.append(prob)


            selected_item = random.choices(feasible_candidates, weights = probabilities , k=1)[0] 
            
            current_bag.append(selected_item)
            current_weight += poid[selected_item]
            current_value += value[selected_item]
            
            candidate_list.remove(selected_item) 


        solutions_in_generation.append((current_bag, current_weight, current_value))


        if current_value > best_value_globale:
            best_value_globale = current_value
            best_solution_globale = list(current_bag)



    for i in range(n):
        pheromone[i] = (1-rho)*pheromone[i] 

        for sol_bag , _, sol_value in solutions_in_generation:
            delta_tau = sol_value / Q
            for item_index in sol_bag:
                pheromone[item_index] += delta_tau



    if (iteration+1) % 10 == 0:

        print(f"\nIteration {iteration+1}")
        print(f"Best solution: {best_solution_globale}")
        print(f"Best value: {best_value_globale}")
        print(f"Pheromone: {pheromone}")
                       