import pulp

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

