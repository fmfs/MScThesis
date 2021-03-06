import numpy as np
import sys
import itertools
import pickle
import json
import codecs
import ast
import os.path
from collections import OrderedDict
import csv
from operator import or_
from datetime import datetime
from  docplex.mp.model import Model
import math
import time
import gc
import random as rnd
from six.moves import urllib

rnd.seed(0)
count = 0
def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1
"""class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
"""
class ilpTIP(object):
    
    def __init__(self):
        
        self.run_big_data()
        
    def run_big_data(self):

        if os.path.exists("/media/fmfs/Data1"):
            file_path = '/media/fmfs/Data1/Tese/befly/datafiles/alldatacosts.json'
        else:
            file_path = '/home/fmfs/befly/datafiles/alldatacosts.json'
        obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        cm_p = np.array(b_new)

        if os.path.exists("/media/fmfs/Data1"):
            file_path = '/media/fmfs/Data1/Tese/befly/datafiles/alldatatimes.json'
        else:
            file_path = '/home/fmfs/befly/datafiles/alldatatimes.json'
        obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        cm_d = np.array(b_new)

        for count in range(8000):
    
            maxDate = ['01/10/2019', '02/10/2019', '03/10/2019', '04/10/2019', '05/10/2019',
                       '06/10/2019', '07/10/2019', '08/10/2019', '09/10/2019', '10/10/2019',
                       '11/10/2019', '12/10/2019', '13/10/2019', '14/10/2019', '15/10/2019']
            cities = ['LHR', 'CDG', 'AMS', 'FRA', 'IST', 'MAD', 'BCN', 'MUC', 'DME', 'CIA',
                      'DUB', 'ZRH', 'CPH', 'OSL', 'ARN', 'BRU', 'ATH', 'WAW', 'BUD', 'PRG',
                      'HEL', 'OTP', 'MIL', 'IEV', 'KEF', 'RIX', 'MLA', 'BEG', 'VNO', 'MSQ',
                      'TIA', 'EVN', 'VIE', 'GYD', 'SJJ', 'SOF', 'ZAG', 'LCA', 'TLL', 'TBS',
                      'ALA', 'LUX', 'SKP', 'KIV', 'PRN', 'TIV', 'EDI', 'CWL', 'BFS', 'LJU']
            Departure_city = 'LIS'
            Return_city = 'LIS'
            key = range(1,51)
            c = zip(key,cities)
            dict_cities = {i:city for i,city in c}
            duration = [rnd.randint(2,5) for city in dict_cities]
            num_clusters = rnd.randint(2,9)
            #num_clusters = 2
            num_DestCluster = {i:rnd.randint(2,5) for i in range(1,num_clusters+1)}
            #num_DestCluster = {i:1 for i in range(1,num_clusters+1)}
            StartSpan = rnd.choice(maxDate)
            clusters = [i for i in range(1,num_clusters+1)]
            DestCluster = [None] * (num_clusters+1)
            DestDuration = [None] * (num_clusters+1)
            c = list(zip(dict_cities, duration))
            rnd.shuffle(c)
            cities, duration = zip(*c)
            cities = list(cities)
            duration = list(duration)
            for i in clusters:
                aux_cities = [None] * (num_DestCluster[i]+1)
                aux_duration = [None] * (num_DestCluster[i]+1)
                for j in range(1,num_DestCluster[i]+1):
                    aux_cities[j] = cities.pop()
                    aux_duration[j] = duration.pop()
                DestCluster[i] = aux_cities
                DestCluster[i] = [x for x in DestCluster[i] if x is not None]
                DestDuration[i] = aux_duration
                DestDuration[i] = [x for x in DestDuration[i] if x is not None]
            DestCluster = [x for x in DestCluster if x is not None]
            DestDuration = [x for x in DestDuration if x is not None]
            cities_t_str = list(map(str, DestCluster))
            stop_t_str = list(map(str, DestDuration))
            for i in range(len(cities_t_str)):
                cities_t_str[i] = cities_t_str[i].replace('[', '(')
                cities_t_str[i] = cities_t_str[i].replace(']', ')')
                stop_t_str[i] = stop_t_str[i].replace('[', '(')
                stop_t_str[i] = stop_t_str[i].replace(']', ')')

            ds1 = datetime.strptime('01/10/2019', "%d/%m/%Y")
            ds2 = datetime.strptime(StartSpan, "%d/%m/%Y")
            T_start_max = abs((ds2 - ds1).days)

            max_stop_t = [None]*len(DestDuration)
            min_stop_t = [None]*len(DestDuration)
            for i in range(len(DestDuration)):
                max_stop_t[i] = max(DestDuration[i])
                max_stop_t[i] = int(max_stop_t[i])
                min_stop_t[i] = min(DestDuration[i])
                min_stop_t[i] = int(min_stop_t[i])

            #first_day = min(min_stop_t)
            last_day = T_start_max
            for i in range(len(DestDuration)):
                last_day += np.max(DestDuration[i])
            last_day += 1
            flat_DestCluster = [item for sublist in DestCluster for item in sublist]
            flat_DestDuration = [item for sublist in DestDuration for item in sublist]

            N = recursive_len(DestCluster) + 2
            self.n = N - 2
            cost_matrix, time_matrix = np.full([last_day, N, N], np.inf), np.full([last_day, N, N], np.inf)

            #final_DestCluster = [0] + flat_DestCluster + [51] 

            for day in range(last_day):
                for departure,i in zip(flat_DestCluster,range(1,len(flat_DestCluster)+1)):
                    cost_matrix[day][i][N-1] = cm_p[day][departure][51]
                    time_matrix[day][i][N-1] = cm_d[day][departure][51]
                    for arrival,j in zip(flat_DestCluster,range(1,len(flat_DestCluster)+1)):
                        cost_matrix[day][0][j] = cm_p[day][0][arrival]
                        time_matrix[day][0][j] = cm_d[day][0][arrival]
                        if arrival != departure:
                            cost_matrix[day][i][j] = cm_p[day][departure][arrival]
                            time_matrix[day][i][j] = cm_d[day][departure][arrival]

            self.cities = {}
            self.stop_t = {}
            self.cities["separate"] = flat_DestCluster
            self.cities["together"] = DestCluster
            self.stop_t["separate"] = flat_DestDuration
            self.stop_t["together"] = DestDuration
            self.w = flat_DestDuration
            self.time_min = maxDate[0]
            self.time_max = StartSpan
            self.node_zero = Departure_city
            self.node_final = Return_city
            self.cost_matrix = cost_matrix
            self.time_matrix = time_matrix
            ds1 = datetime.strptime(self.time_min, "%d/%m/%Y")
            ds2 = datetime.strptime(self.time_max, "%d/%m/%Y")
            self.T_start_max = abs((ds2 - ds1).days)

            if os.path.exists("/media/fmfs/Data1"):
                file_path = '/media/fmfs/Data1/Tese/befly/datafiles/mycsv_gFTP.csv'
            else:
                file_path = '/home/fmfs/befly/datafiles/mycsv_gFTP.csv'
            with open(file_path, 'a', newline='') as f:
                fieldnames = ['cities', 'stop_t', 'StartSpan', 'cost_matrix', 'time_matrix']
                thewriter = csv.DictWriter(f, fieldnames=fieldnames)
                thewriter.writerow({'cities' : self.cities, 'stop_t' : self.stop_t, 'StartSpan' : self.time_max, 'cost_matrix': self.cost_matrix, 'time_matrix' : self.time_matrix})


            self.run_gftp()

    def run_gftp(self):
        cities = self.cities
        stop_t = self.stop_t
        maxsum_stop_t = 0

        #print(cities)
        stop_time = [None]*len(stop_t["together"])
        for i in range(len(stop_t["together"])):    
            stop_time[i] = list(map(int, stop_t["together"][i]))
        
        for count in range(len(stop_time)):  
            maxsum_stop_t += max(stop_time[count])

        cm = self.cost_matrix
        #print(cm)
        #cm = self.time_matrix
        tm = self.time_matrix

        T_start_max = self.T_start_max
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

        s = {i:self.stop_t["separate"][i-1] for i in cities["separate"]} # minimum stop time at each city

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
        t = {(i,j,k):tm[k][i][j] for i,j in A for k in range(I[(i,j)])}
        t0 = {(i,i,0):0 for i in Vs}
        t.update(t0)
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
                                        >= 0 for i in V[p] for j in V[q] for k in range(I[(i,j)]) if a[i,j,k]+s[j]<I[(i,j)]) #HERE I[(i,j)] is "wrongly" used. should me
        
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
        solution = mdl.solve()
        #solution = mdl.solve(log_output=True)
        #mdl.print_solution()
        #mdl.solution.solve_status
        #mdl.print_information()
        print("next")
        #print(self.cost_matrix)
        
        if solution != None:
            solution_dict = list(solution.as_dict().items())
            var_x=list(solution.as_dict().items())[:n+1]

            indicator_x=[]
            for i in range(n+1):
                indicator_x+=[[int(s) for s in var_x[i][0].split('_',-1) if s.isdigit()]]
            objective_cost = 0
            objective_time = 0
            for index in range(len(indicator_x)):
                i = indicator_x[index][0]
                j = indicator_x[index][1]
                k = indicator_x[index][2]
                objective_cost += c[i,j,k]
                objective_time += t[i,j,k]
        else:
            solution_dict = [None]
            objective_cost = None
            objective_time = None
        window_size = T_start_max + maxsum_stop_t

        if os.path.exists("/media/fmfs/Data1"):
            file_path = '/media/fmfs/Data1/Tese/befly/datafiles/mycsv_gFTP_solution.csv'
        else:
            file_path = '/home/fmfs/befly/datafiles/mycsv_gFTP_solution.csv'
        with open(file_path, 'a', newline='') as f:
            fieldnames = ['number_of_clusters', 'number_of_cities', 'window_size', 'solve', 'cost', 'time', 'elapsed_time_cpu', 'elapsed_time_wall']
            thewriter = csv.DictWriter(f, fieldnames=fieldnames)
            thewriter.writerow({'number_of_clusters' : n, 'number_of_cities' : m, 'window_size' : window_size, 'solve' : solution_dict, 'cost' : objective_cost, 'time' : objective_time, 'elapsed_time_cpu' : elapsed_time_cpu, 'elapsed_time_wall' : elapsed_time_wall})
        mdl.end()

if __name__ == "__main__":
    ilpTIP()
