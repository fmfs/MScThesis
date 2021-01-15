import numpy as np
import sys
import itertools
import time
import codecs, json
import csv
import os.path
from operator import or_
from datetime import datetime
from  docplex.mp.model import Model

#fp = '/media/fmfs/Data1/Tese/befly/datafiles/mylog.txt'
#fp = '/home/fmfs/befly/datafiles/mylog.txt'
#sys.stdout = open(fp, 'w')
"""class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
"""
class ilpTIP(object):
    def __init__(self, tsp_model, cities, stop_t, time_min, time_max, cmsetup, node_zero, node_final, *args, **kwargs):
        self.n, self.cm, self.wp = tsp_model["num_nodes"], tsp_model["cost_matrix"], tsp_model["duration"]
        self.cities = cities
        self.stop_t = stop_t
        self.node_zero = node_zero
        self.node_final = node_final
        self.cmsetup = cmsetup
        self.time_max = time_max
        self.time_min = time_min
        ds1 = datetime.strptime(time_min, "%d/%m/%Y")
        self.ds1 = ds1
        ds2 = datetime.strptime(time_max, "%d/%m/%Y")
        self.__dict__.update((key, value) for key, value in kwargs.items())
        T_start_max = abs((ds2 - ds1).days)
        self.T_start_max = T_start_max

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
        mdl.print_solution()
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
        
    def run_gftp(self):
        cities = self.cities
        stop_t = self.stop_t
        maxsum_stop_t = 0
        print(cities)
        print(stop_t)
        stop_time = [None]*len(stop_t["together"])
        for i in range(len(stop_t["together"])):    
            stop_time[i] = list(map(int, stop_t["together"][i]))
        
        for count in range(len(stop_time)):  
            maxsum_stop_t += max(stop_time[count])
        print(maxsum_stop_t)
        if self.cmsetup == 'c': 
            lista = [None]*len(self.cities["together"])
            cities_t_str = self.cities["together"]
            stop_t_str = self.stop_t["together"]
            for l in range(len(cities_t_str)):
                lista[l]=list(zip(cities_t_str[l], stop_t_str[l]))
            namefile=str(self.T_start_max)+'d'
            for h in range(len(lista)):
                namefile+='_'
                for j in lista[h]:
                    namefile+='_'
                    namefile+='_'.join(j)

        cm=self.cm
        T_start_max = self.T_start_max
        print(T_start_max)
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
        
        I_value = T_start_max
        for i in range(len(stop_time)):
            I_value += np.max(stop_time[i])
        
        I = {(i,j):I_value+1 for i,j in A if i!=j}
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
        #mdl.add_constraints(mdl.sum(mdl.sum(a[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for i in Vs if i not in V[n+1]) 
        #                        <= len(cm) for j in V[n+1])
        mdl.add_constraints(mdl.sum(mdl.sum(d[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for j in Vs if j not in V[0])
                                <= T_start_max for i in V[0])
        #Degree
        mdl.add_constraints(mdl.sum(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in Vs if j not in V[p]) for i in V[p]) 
                                == 1 for p in range(n+1))
        mdl.add_constraints(mdl.sum(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in V[p]) for i in Vs if i not in V[p]) 
                                == 1 for p in range(1,n+2))
        mdl.add_constraints(mdl.sum(mdl.sum(x[j,i,k] for k in range(I[(j,i)])) for j in Vs if j not in V[n+1] if j!=i)
                                - mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in Vs if j not in V[0] if j!=i)
                                == 0 for i in Vs if i not in V[0] if i not in V[n+1])
        
        #mdl.add_constraints(mdl.sum(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in V[p] if j!=i) for i in V[p])
        #                        == 0 for p in range(1,n+1))
        
        mdl.add_constraints(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for i in Vs if i not in V[0]) 
                                == 0 for j in V[0])
        mdl.add_constraints(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in Vs if j not in V[n+1]) 
                                == 0 for i in V[n+1])

        for p in range(n+1):
            for q in range(1,n+2):
                if p!=q:
                    mdl.add_constraint(w[p,q] == mdl.sum(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in V[q]) for i in V[p]))

        for p in range(n+1):
            for q in range(1,n+1):
                if p!=q:
                    #mdl.add_constraints(mdl.sum(x[j,r,a[i,j,k]+s[j]] for r in Vs if r not in V[p] and r not in V[q] and r not in V[0])
                    #                    <= 1 for i in V[p] for j in V[q] for k in range(I[(i,j)]) if a[i,j,k]+s[j]<I[(i,j)])
                    mdl.add_constraints(-x[i,j,k] + mdl.sum(x[j,r,a[i,j,k]+s[j]] for r in Vs if r not in V[p] and r not in V[q] and r not in V[0])
                                        >= 0 for i in V[p] for j in V[q] for k in range(I[(i,j)]) if a[i,j,k]+s[j]<I[(i,j)])
        
        ## MTZ LP Relaxation constraints - u_i represents the position of i in the final arc.
        mdl.add_constraint(u[0]==0) # Fixing the first city to the first position of the arc
        mdl.add_constraint(u[n+1]==n+1) # Fixing the last city to the last position of the arc
        for p in range(n+1):
            for q in range(1,n+2):
                if p!=q:
                    mdl.add_constraint(u[p] - u[q] + (n+1)*w[p,q] <= n)

        #for p in range(1,n+1):
        #    for q in range(1,n+2):
        #        if p!=q:
        #            mdl.add_constraint(u[p] - u[q] + (n+1)*w[p,q] + (n-1)*w[q,p] <= n)
        #mdl.add_constraints(u[p] - mdl.sum(w[p,q] for q in range(1,n+2) if q!=p) >= 1 for p in range(1,n+2))
        
        #mdl.add_constraints(u[p] + n*w[0,p] <= n+1 for p in range(1,n+2))
        mdl.parameters.clocktype = 1
        mdl.parameters.timelimit = 1800
        start_time_wall = time.time()
        start_time_cpu = time.clock()
        solution = mdl.solve(log_output=True)
        elapsed_time_cpu = time.clock() - start_time_cpu
        elapsed_time_wall = time.time() - start_time_wall
        mdl.export_as_lp('/tmp/gFTP1.lp')
        #solution = mdl.solve()
        #solution = mdl.solve(log_output=True)
        mdl.print_solution()
        mdl.solution.solve_status
        mdl.print_information()

        if self.cmsetup == 'c':
        #    data = {}
        #    data['number_of_clusters'] = n
        #    data['number_of_cities'] = m
        #    data['window_size'] = T_start_max + maxsum_stop_t
        #    data['solve'] = list(solution.as_dict().items())
        #    data['obj'] = solution.objective_value
        #    data['elapsed_time'] = elapsed_time

            #file_path = '/media/fmfs/Data1/Tese/befly/datafiles/myjson_solution.json'
            #file_path = "/home/fmfs/befly/datafiles/" + str(self.cmsetup) + "/" + str(n)+ "clusters/solution/" + namefile ## your path variable
            #file_path = "/media/fmfs/Data1/Tese/befly/datafiles/" + str(self.cmsetup) + "/" + str(n)+ "clusters/solution/" + namefile 
            #file_path = "/media/fmfs/Data1/Tese/befly/datafiles/c/3clusters/solution/1d__BCN_1_CIA_1_SXF_1__CDG_1_BUD_1_IST_1__LHR_1_AMS_1_DME_1_solution" ## your path variable
            #file_path = "/media/fmfs/Data1/Tese/befly/datafiles/c/solution/1"
            #with open(file_path, 'a') as f:
            #    json.dump(data, f, cls=NumpyEncoder)
                #json_dump=json.dumps(data, cls=NumpyEncoder)
                #json.dump(json_dump, f)
            window_size = T_start_max + maxsum_stop_t

            if os.path.exists("/media/fmfs/Data1"):
                file_path = '/media/fmfs/Data1/Tese/befly/datafiles/mycsv_solution.csv'
            else:
                file_path = '/home/fmfs/befly/datafiles/mycsv_solution.csv'
            with open(file_path, 'a', newline='') as f:
                fieldnames = ['number_of_clusters', 'number_of_cities', 'window_size', 'solve', 'obj', 'elapsed_time_cpu', 'elapsed_time_wall']
                thewriter = csv.DictWriter(f, fieldnames=fieldnames)
            
                thewriter.writerow({'number_of_clusters' : n, 'number_of_cities' : m, 'window_size' : window_size, 'solve' : list(solution.as_dict().items()), 'obj' : solution.objective_value, 'elapsed_time_cpu' : elapsed_time_cpu, 'elapsed_time_wall' : elapsed_time_wall})
  
            #with open(file_path, 'wb') as f:  # Python 3: open(..., 'wb')
            #    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        var_x=list(solution.as_dict().items())[:n+1]
        var_w=list(solution.as_dict().items())[n+1:2*n+2]

        indicator_x=[]
        indicator_w=[]
        for i in range(n+1):
            indicator_x+=[[int(s) for s in var_x[i][0].split('_',-1) if s.isdigit()]]
            indicator_w+=[[int(s) for s in var_w[i][0].split('_',-1) if s.isdigit()]]

        indicator_w_index=[0]*len(indicator_w)
        indicator_w_index[0] = indicator_w[0][0]
        for i in range(1,len(indicator_w)):
            indicator_w_index[i] = indicator_w[indicator_w_index[i-1]][1]
        indicator_w_index.remove(0) 

        closed_tour = [0]*(n+2)
        closed_tour[n+1] = m+1
        for i in range(1,len(indicator_w_index)+1):
            s = indicator_w_index[i-1]
            closed_tour[i] = indicator_x[s][0]
        
        self.best_start_time = indicator_x[0][2]
        self.closed_tour = closed_tour
        self.best_cost = solution.objective_value
        mdl.end()

    def result(self):
        obj_return = {
        "num_nodes": self.n,
        "start_time": self.best_start_time,
        "cost": self.best_cost,
        "path": self.closed_tour,
        }
        return obj_return
