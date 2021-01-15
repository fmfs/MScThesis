import numpy as np
import sys
import itertools
import json
import codecs
import ast
import cplex
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

        for count in range(9000):
    
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
            num_cities = rnd.randint(2,10)
            StartSpan = rnd.choice(maxDate)
            cities = [i for i in range(1,num_cities+1)]
            DestCities = [None] * (num_cities+1)
            DestDuration = [None] * (num_cities+1)
            c = list(zip(dict_cities, duration))
            rnd.shuffle(c)
            cities, duration = zip(*c)
            cities = list(cities)
            duration = list(duration)
            for i in range(len(DestCities)-1):
                DestCities[i] = cities.pop()
                DestDuration[i] = duration.pop()
            DestCities = [x for x in DestCities if x is not None]
            #print(DestCities)
            DestDuration = [x for x in DestDuration if x is not None]
            cities_t_str = list(map(str, DestCities))
            stop_t_str = list(map(str, DestDuration))
            for i in range(len(cities_t_str)):
                cities_t_str[i] = cities_t_str[i].replace('[', '(')
                cities_t_str[i] = cities_t_str[i].replace(']', ')')
                stop_t_str[i] = stop_t_str[i].replace('[', '(')
                stop_t_str[i] = stop_t_str[i].replace(']', ')')

            ds1 = datetime.strptime('01/10/2019', "%d/%m/%Y")
            ds2 = datetime.strptime(StartSpan, "%d/%m/%Y")
            T_start_max = abs((ds2 - ds1).days)

            #first_day = min(min_stop_t)
            last_day = T_start_max
            last_day += sum(DestDuration)
            last_day += 1

            N = recursive_len(DestCities) + 2
            self.n = N - 2
            cost_matrix, time_matrix = np.full([last_day, N, N], np.inf), np.full([last_day, N, N], np.inf)

            #final_DestCluster = [0] + flat_DestCluster + [51] 

            for day in range(last_day):
                for departure,i in zip(DestCities,range(1,len(DestCities)+1)):
                    cost_matrix[day][i][N-1] = cm_p[day][departure][51]
                    time_matrix[day][i][N-1] = cm_d[day][departure][51]
                    for arrival,j in zip(DestCities,range(1,len(DestCities)+1)):
                        cost_matrix[day][0][j] = cm_p[day][0][arrival]
                        time_matrix[day][0][j] = cm_d[day][0][arrival]
                        if arrival != departure:
                            cost_matrix[day][i][j] = cm_p[day][departure][arrival]
                            time_matrix[day][i][j] = cm_d[day][departure][arrival]

            self.cities = {}
            self.stop_t = {}
            self.cities["separate"] = DestCities
            self.cities["together"] = DestCities
            self.stop_t["separate"] = DestDuration
            self.stop_t["together"] = DestDuration
            self.w = DestDuration
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
                file_path = '/media/fmfs/Data1/Tese/befly/datafiles/mycsv_FTP.csv'
            else:
                file_path = '/home/fmfs/befly/datafiles/mycsv_FTP.csv'
            with open(file_path, 'a', newline='') as f:
                fieldnames = ['cities', 'stop_t', 'StartSpan', 'cost_matrix', 'time_matrix']
                thewriter = csv.DictWriter(f, fieldnames=fieldnames)
                thewriter.writerow({'cities' : self.cities, 'stop_t' : self.stop_t, 'StartSpan' : self.time_max, 'cost_matrix': self.cost_matrix, 'time_matrix' : self.time_matrix})


            self.run_ftp()

    def run_ftp(self):
        
        cities = self.cities
        stop_t = self.stop_t
        maxsum_stop_t = 0

        stop_time = stop_t["separate"]        
        
        cm = self.cost_matrix
        #print(cm)
        tm = self.time_matrix

        T_start_max = self.T_start_max
        n = len(cities["separate"]) # number of intermediate clusters
        aux=1
        for i in range(n):
            cities["separate"][i]=aux
            aux +=1
        N = [i for i in range(1,n+1)]
        V = [0] + cities["separate"] + [n+1]
        s = {i:self.stop_t["separate"][i-1] for i in cities["separate"]} # minimum stop time at each city
        A = [(i,j) for i in V for j in V if i!=j] # arco [i,i] não existe
        A_aux = [(i,j) for i in V for j in V]
    
        I_value = T_start_max
        for i in range(len(stop_time)):
            I_value += np.max(stop_time[i])
        I = {(i,j):I_value+1 for i,j in A if i!=j}
        I0 = {(i,i):0 for i in V}
        I.update(I0)

        c = {(i,j,k):cm[k][i][j] for i,j in A for k in range(I[(i,j)])}
        c0 = {(i,i,0):0 for i in V}
        c.update(c0)
        t = {(i,j,k):tm[k][i][j] for i,j in A for k in range(I[(i,j)])}
        t0 = {(i,i,0):0 for i in V}
        t.update(t0)
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
        mdl.add_constraint(mdl.sum(mdl.sum(d[0,j,k]*x[0,j,k] for k in range(I[(0,j)])) for j in range(1,n+2)) >= 0)
        mdl.add_constraints(mdl.sum(mdl.sum(d[i,j,k]*x[i,j,k] for k in range(I[(i,j)])) for j in range(1,n+2)) 
                            - mdl.sum(mdl.sum(a[j,i,k]*x[j,i,k] for k in range(I[(j,i)])) for j in range(n+1)) 
                            == s[i] for i in N)
        mdl.add_constraint(mdl.sum(mdl.sum(a[i,n+1,k]*x[i,n+1,k] for k in range(I[(i,n+1)])) for i in range(n+1)) 
                        <= len(cm))
        mdl.add_constraints(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for j in range(1,n+2)) 
                            == 1 for i in range(n+1))
        mdl.add_constraints(mdl.sum(mdl.sum(x[i,j,k] for k in range(I[(i,j)])) for i in range(n+1)) 
                            == 1 for j in range(1,n+2))
        mdl.add_constraint(mdl.sum(mdl.sum(x[i,0,k] for k in range(I[(i,0)])) for i in range(1,n+2)) 
                        == 0)
        mdl.add_constraint(mdl.sum(mdl.sum(x[n+1,j,k] for k in range(I[(n+1,j)])) for j in range(n+1)) 
                        == 0)

        ## MTZ formulation constraints - u_i represents the position of i in the final arc.
        mdl.add_constraint(u[0]==0) # Fixing the first city to the first position of the arc
        mdl.add_constraint(u[n+1]==n+1) # Fixing the last city to the last position of the arc
        for i in range(1,n+1):
            for j in range(1,n+2):
                if i!=j:
                    mdl.add_constraint(u[i]-u[j]+(n+1)*mdl.sum(x[i,j,k] for k in range(I[(i,j)]))
                                        <= n)
        
        mdl.parameters.clocktype = 1
        mdl.parameters.timelimit = 1800
        #solution = mdl.solve()
        start_time_wall = time.time()
        start_time_cpu = time.clock()
        solution = mdl.solve(log_output=True)
        elapsed_time_cpu = time.clock() - start_time_cpu
        elapsed_time_wall = time.time() - start_time_wall
        mdl.export_as_lp('/tmp/FTP.lp')
        #mdl.print_solution()
        #mdl.solution.solve_status
        #mdl.print_information()

        if solution != None:
            solution_dict = list(solution.as_dict().items())
            var_x=list(solution.as_dict().items())[:n+1]

            indicator_x=[]
            for i in range(n+1):
                indicator_x+=[[int(s) for s in var_x[i][0].split('_',-1) if s.isdigit()]]
            #print(indicator_x)
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
            file_path = '/media/fmfs/Data1/Tese/befly/datafiles/mycsv_FTP_solution.csv'
        else:
            file_path = '/home/fmfs/befly/datafiles/mycsv_FTP_solution.csv'
        with open(file_path, 'a', newline='') as f:
            fieldnames = ['number_of_cities', 'window_size', 'solve', 'cost', 'time', 'elapsed_time_cpu', 'elapsed_time_wall']
            thewriter = csv.DictWriter(f, fieldnames=fieldnames)
            thewriter.writerow({'number_of_cities' : n, 'window_size' : window_size, 'solve' : solution_dict, 'cost' : objective_cost, 'time': objective_time, 'elapsed_time_cpu' : elapsed_time_cpu, 'elapsed_time_wall' : elapsed_time_wall})
        mdl.end()

if __name__ == "__main__":
    ilpTIP()