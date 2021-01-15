import numpy as np
import sys
import itertools
from operator import or_
from  docplex.mp.model import Model


class ilpTIP(object):
    def __init__(self, tsp_model, cities, stop_t, node_zero, node_final, *args, **kwargs):
        self.n, self.cm, self.wp = tsp_model["num_nodes"], tsp_model["cost_matrix"], tsp_model["duration"]
        self.cities = cities
        self.stop_t = stop_t
        self.node_zero = node_zero
        self.node_final = node_final
        self.__dict__.update((key, value) for key, value in kwargs.items())
        if cities["together"] == []:
            self.run_ftp()
        else:
            self.run_gftp()

    def run_ftp(self):
        n=self.n
        cm=self.cm
        N = [i for i in range(1,n-1)]
        V = [i for i in range(n)]
        s = {i:self.wp[i-1] for i in N} # minimum stop time at each city
        A = [(i,j) for i in V for j in V if i!=j] # arco [i,i] não existe
        A_aux = [(i,j) for i in V for j in V]
    
        I = {(i,j):len(cm) for i,j in A if i!=j}
        I0 = {(i,i):0 for i in V}
        I.update(I0)

        c = {(i,j,k):cm[k][i][j] for i,j in A for k in range(I[(i,j)])}
        c0 = {(i,i,0):0 for i in V}
        c.update(c0)
        d = {(i,j,k):k for i,j in A for k in range(I[(i,j)])}
        d0 = {(i,i,0):0 for i in V}
        d.update(d0)
        a = {(i,j,k):k for i,j in A for k in range(I[(i,j)])}
        a0 = {(i,i,0):0 for i in V}
        a.update(a0)
        
        mdl = Model('FTP')

        x = mdl.binary_var_dict(c,name='x')
        u = mdl.integer_var_list(V,name='u')

        ## Objective Function
        mdl.minimize(mdl.sum(mdl.sum(c[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for i,j in A_aux))

        ## Constraints
        mdl.add_constraint(mdl.sum(mdl.sum(d[0,j,k]*x[0,j,k] for k in range(I[(0,j)])) for j in range(1,n)) >= 0)
        mdl.add_constraints(mdl.sum(mdl.sum(d[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for j in range(1,n)) 
                            - mdl.sum(mdl.sum(a[j,i,k]*x[j,i,k] for k in range(I[(j,i)])) for j in range(n-1)) 
                            == s[i] for i in N)
        mdl.add_constraint(mdl.sum(mdl.sum(a[i,n-1,k]*x[i,n-1,k] for k in range(I[(i,n-1)])) for i in range(n-1)) 
                        <= len(cm))
        mdl.add_constraints(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in range(1,n)) 
                            == 1 for i in range(n-1))
        mdl.add_constraints(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for i in range(n-1)) 
                            == 1 for j in range(1,n))
        mdl.add_constraint(mdl.sum(mdl.sum(x[i,0,k] for k in range(I[(i,0)])) for i in range(1,n)) 
                        == 0)
        mdl.add_constraint(mdl.sum(mdl.sum(x[n-1,j,k] for k in range(I[(n-1,j)])) for j in range(n-1)) 
                        == 0)

        ## MTZ formulation constraints - u_i represents the position of i in the final arc.
        mdl.add_constraint(u[0]==0) # Fixing the first city to the first position of the arc
        mdl.add_constraint(u[n-1]==n-1) # Fixing the last city to the last position of the arc
        for i in range(1,n):
            for j in range(1,n):
                if i!=j:
                    mdl.add_constraint(u[i]-u[j]+(n-1)*mdl.sum(x[i,j,k] for k in range(I[(i,j)]))
                                        <= n-2)
        
        #solution = mdl.solve()
        solution = mdl.solve(log_output=True)
        mdl.export_as_lp('/tmp/FTP.lp')
        #mdl.print_solution()
        #mdl.solution.solve_status
        #mdl.print_information()
        var_x=list(solution.as_dict().items())[:n-1]
        var_u=list(solution.as_dict().items())[n-1:]
        var_u_arc=list(solution.as_dict().values())[n-1:]

        indicator_x = []
        indicator_u = []
        for i in range(n-1):
            indicator_x += [[int(s) for s in var_x[i][0].split('_',-1) if s.isdigit()]]
            indicator_u += [[int(s) for s in var_u[i][0].split('_',-1) if s.isdigit()]]
        indicator_u_arc=[int(s) for s in var_u_arc]

        self.best_start_time = indicator_x[0][2]
        closed_tour = [0]*(n)
        count=1
        for i in indicator_u_arc:
            closed_tour[i] += count
            count += 1
        self.closed_tour = closed_tour
        self.best_cost = solution.objective_value

        print(self.cities)
        
    def run_gftp(self):
        cities = self.cities
        stop_t = self.stop_t
        stop_time = [[int(column) for column in row] for row in stop_t["together"]]
        stop_time = [[1]] + stop_time + [[1]]
        cm=self.cm
        n = len(cities["together"]) # number of intermediate clusters
        m = len(cities["separate"]) # number of intermediate possible cities
        N = [i for i in range(1,n+1)]
        Ni = [0] + N + [n+1]

        aux=1
        for i in range(n):
            for j in range(len(cities["together"][i])):
                cities["together"][i][j]=aux
                aux +=1
        V = [[0]] + cities["together"] + [[m+1]]

        aux=1
        for i in range(m):
            cities["separate"][i]=aux
            aux +=1
        Vs = [0] + cities["separate"] + [m+1]

        s = {i:self.wp[i-1] for i in cities["separate"]} # minimum stop time at each city

        A = [(i,j) for i in Vs for j in Vs if i!=j] # arco [i,i] não existe
        Ai = {(i,j):1 for i in range(len(V)) for j in range(len(V)) if i!=j}
        A_aux = [(i,j) for i in Vs for j in Vs]
        
        I = {(i,j):len(cm) for i,j in A if i!=j}
        I0 = {(i,i):0 for i in Vs}
        I.update(I0)

        c = {(i,j,k):cm[k][i][j] for i,j in A for k in range(I[(i,j)])}
        c0 = {(i,i,0):0 for i in Vs}
        c.update(c0)

        d = {(i,j,k):k for i,j in A for k in range(I[(i,j)])}
        d0 = {(i,i,0):0 for i in Vs}
        d.update(d0)

        a = {(i,j,k):k for i,j in A for k in range(I[(i,j)])}
        a0 = {(i,i,0):0 for i in Vs}
        a.update(a0)
        
        mdl = Model('gFTP')

        x = mdl.binary_var_dict(c,name='x')
        w = mdl.binary_var_dict(Ai,name='w')
        u = mdl.integer_var_list(Ni,name='u')

        ## Objective Function
        mdl.minimize(mdl.sum(mdl.sum(c[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for i,j in A_aux))

        ## Constraints
        # Time
        mdl.add_constraints(mdl.sum(mdl.sum(d[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for j in Vs if j not in V[0]) 
                                >= 0 for i in V[0])
        mdl.add_constraints(mdl.sum(mdl.sum(a[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for i in Vs if i not in V[n+1]) 
                                <= len(cm) for j in V[n+1])
        #Degree
        mdl.add_constraints(mdl.sum(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in Vs if j not in V[p]) for i in V[p]) 
                                == 1 for p in range(n+1))
        mdl.add_constraints(mdl.sum(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in V[p]) for i in Vs if i not in V[p]) 
                                == 1 for p in range(1,n+2))
        mdl.add_constraints(mdl.sum(mdl.sum(x[j,i,k] for k in range(I[(j,i)])) for j in Vs if j not in V[n+1] if j!=i)
                                - mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in Vs if j not in V[0] if j!=i)
                                == 0 for i in Vs if i not in V[0] if i not in V[n+1])
        mdl.add_constraints(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for i in Vs if i not in V[0]) 
                                == 0 for j in V[0])
        mdl.add_constraints(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in Vs if j not in V[n+1]) 
                                == 0 for i in V[n+1])

        for p in range(n+1):
            for q in range(n+1):
                if p!=q:
                    mdl.add_constraint(w[p,q] == mdl.sum(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in V[q]) for i in V[p]))

        #for p in range(1,n+1):
        #    for q in range(n+2):
        #        if p!=q:
        #            mdl.add_constraint(mdl.sum(mdl.sum(d[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for i in V[p] for j in V[q] if j not in V[0])
        #                            - mdl.sum(mdl.sum(a[j,i,k]*x[j,i,k] for k in range(I[(j,i)])) for i in V[p] for j in V[q] if j not in V[n+1]) 
        #                            <= s[i])
        #            mdl.add_constraint(-w[p,q] + mdl.sum(mdl.sum(d[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for i in V[p] for j in V[q] if j not in V[0])
        #                                    - mdl.sum(mdl.sum(a[j,i,k]*x[j,i,k] for k in range(I[(j,i)])) for i in V[p] for j in V[q] if j not in V[n+1])
        #                                    >= s[i]-1)
        for p in range(1,n+1):
            for q in range(1,n+1):
                if p!=q:
                    mdl.add_constraints(-x[i,j,k] + mdl.sum(x[j,r,a[i,j,k]+s[j]] for r in Vs if r not in V[p] and r not in V[q] and r not in V[0])
                                        >= 0 for i in V[p] for j in V[q] for k in range(I[(i,j)]) if a[i,j,k]+s[j]<I[(i,j)])

        #for p in range(1,n+1):
        #    for q in range(0,n+2):
        #        if p!=q:
        #            mdl.add_constraint(mdl.sum(mdl.sum(mdl.sum(d[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for j in V[q] if q!=0) for i in V[p])
        #                                    - mdl.sum(mdl.sum(mdl.sum(a[j,i,k]*x[j,i,k] for k in range(I[(j,i)])) for j in V[q] if q!=n+1) for i in V[p])
        #                                    == w[p,q]*stop_time[p][0])

        for p in range(1,n+2):
            for q in range(1,n+2):
                if p!=q:
                    mdl.add_constraint(u[p] - u[q] + (n+1)*w[p,q] <= n)

        #for p in range(1,n+2):
        #    for q in range(1,n+2):
        #        if p!=q:
        #            mdl.add_constraint(u[p] - u[q] + (n+1)*w[p,q] + (n-1)*w[q,p] <= n)
        #mdl.add_constraints(u[p] - mdl.sum(w[p,q] for q in range(1,n+2) if q!=p) >= 1 for p in range(1,n+2))
        #mdl.add_constraints(u[p] + n*w[0,p] <= n+1 for p in range(1,n+2))

        solution = mdl.solve(log_output=True)
        mdl.export_as_lp('/tmp/gFTP.lp')
        #solution = mdl.solve()
        #solution = mdl.solve(log_output=True)
        mdl.print_solution()
        mdl.solution.solve_status
        mdl.print_information()


        var_x=list(solution.as_dict().items())[:m-1]
        var_w=list(solution.as_dict().items())[m-1:]
        var_u_arc=list(solution.as_dict().values())[m-1:]

        indicator_x = []
        indicator_u = []
        for i in range(m-1):
            indicator_x += [[int(s) for s in var_x[i][0].split('_',-1) if s.isdigit()]]
        for i in range(n-1):  
            indicator_w += [[int(s) for s in var_w[i][0].split('_',-1) if s.isdigit()]]
        indicator_u_arc=[int(s) for s in var_u_arc]

        self.best_start_time = indicator_x[0][2]
        closed_tour = [0]*(n)
        count=1
        for i in indicator_u_arc:
            closed_tour[i] += count
            count += 1
        self.closed_tour = closed_tour
        self.best_cost = solution.objective_value

        print(self.cities)

    def result(self):
        obj_return = {
        "num_nodes": self.n,
        "start_time": self.best_start_time,
        "cost": self.best_cost,
        "path": self.closed_tour,
        }
        return obj_return