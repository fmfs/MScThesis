#from TIWID.SimAnn import local 
from OS import local
#from TIWID.AntColOpt import misc
from OS import misc
#import matplotlib.pyplot as plt
import random 
import numpy
import math 
import time 
import json
import uuid
import sys

def calc_rel_err(iterable, true_x):
  return list(map(lambda x: (x-true_x)*100/true_x, iterable))


class AntGraph(object):
  """ This class instantiates a Graph describing the (TD)TSP.
      class variables:
        - nn            # num nodes 
        - cm            # cost matrix
        - pm            # pherormone matrix
        - st            # list of start times
      class methods:
        - cost          # returns cost of entry
        - etha          # returns inverse of cost
        - tau           # returns pherormone
        - update_pm     # updates pherormone  
  """
  def __init__(self, nn, cm, wp, nn_cost = False):
    self.nn = nn                    # num nodes
    self.cm = cm                    # cost matrix
    if isinstance(cm, list):
      self.cm = numpy.array(cm)
    self.dims = self.cm.shape
    self.wp = wp                    # waiting period list
    self.nn_cost = nn_cost

  def init(self, ACS, rho, c_elite, sn, en): 
    self.init_pm(ACS, rho, c_elite, sn, en)                  # create pherormone matrix
    self.init_st()                  # create list of allowed start times 

  def cost(self, t, i, j):
    return self.cm[t,i,j]

  def etha(self, t, i, j):
    c = self.cost(t, i, j)
    return 1 if c == 0 else 1.0/c

  def update_pm(self, t, i, j, val):
    self.pm[t, i, j] = val 

  def init_pm(self, ACS, rho, c_elite, sn, en):
    td = True if self.dims[0] > 1 else False
    if not self.nn_cost:
      cost, path = local.nearest_neighbour(self.nn, self.cm, td, self.wp, sn, en)
    else:
      cost = self.nn_cost
    if ACS:
      self.tau_zero = 1/(self.nn*cost)
    else:
      self.tau_zero = (c_elite + self.nn)/(rho*cost)
    self.pm = numpy.full(self.dims, self.tau_zero)

  def init_st(self):
    dur = numpy.sum(self.wp)
    max_time = self.dims[0]
    self.st = [i for i in range(0,max_time-dur)]

  def show(self):
    return {
      "num_nodes": self.nn,
      "durations": self.wp,
      "cost_matrix": self.cm,
      "pherormone_matrix": self.pm,
      "start_times": self.start_times
    }


class AntColony(object):

  ACS = True 
  start_node = None
  end_node = None
  use_daemon = False 
  max_time = False 
  max_iter = 10000
  num_ants = 10
  num_deposit = 5
  c_elite = 3
  beta = 5
  rho = 0.1
  zeta = 0.1
  alpha = 1 - rho
  q0 = 0.9
  nn_cost = False
  opt_cost = False 

  def __init__(self, tsp_model, name = None, *args, **kwargs):
    nn, cm, wp = tsp_model["num_nodes"], tsp_model["cost_matrix"], tsp_model["duration"]
    self.__dict__.update((key, value) for key, value in kwargs.items())
    self.graph = AntGraph(nn, cm, wp, self.nn_cost)
    if name:
      self.name = name 
    else:
      self.name = "tsp_n_{}".format(self.graph.nn) 
    if not self.ACS:
      self.num_ants = self.graph.nn
      self.rho = 0.5
      self.q0 = 0.5
      self.c_elite =  5           # self.graph.nn 
      self.num_deposit = 5        # self.graph.nn
    self.graph.init(self.ACS, self.rho, self.c_elite, self.start_node, self.end_node) 
    self.make_records()
    self.run_meta()
    self.save_result()
    #self.plot_result()
    #self.plot_two()

  def make_ants(self):
    ants = []
    while len(ants) < self.num_ants:
      sn = self.start_node if self.start_node != None else random.randint(0, self.graph.nn-1)   # start node
      fn = self.end_node if self.end_node != None else sn                                 # final node
      st = 0 if len(self.graph.st) == 0 else random.choice(self.graph.st)                                               # start time
      ntv = [i for i in range(self.graph.nn) if i != sn and i != fn]
      idx = len(ants)
      ants.append(Ant(idx, sn, fn, st, ntv, self))
    return ants 

  def make_records(self):
    self.best_cost = 123456789
    self.best_iter = -1
    self.best_start_time = -1
    self.best_path_mat = []
    self.best_path_vec = []
    self.best_cost_iter = []
    self.best_cost_over_time = []
    self.solutions_iter = []
    self.time_iter = []

  def run_meta(self):
    iter_curr, time_start = 0, time.time()
    while True:
      ants = self.make_ants()
      for ant in ants:
        ant.run()
        self.update(ant, iter_curr)
        if self.ACS:
          self.local_update(ant.path_mat)
      iter_curr, time_exec = iter_curr + 1, time.time() - time_start
      if self.end(iter_curr, time_exec):
        return
      if self.ACS:
        self.acs_update()
      else:
        self.elitist_update()
      self.update_time_and_rec(time_exec)
      #self.update_user(iter_curr)
      self.solutions_iter = []
      self.iterations_completed = iter_curr

  def update_time_and_rec(self, t_exec):
    self.time_iter.append(t_exec)
    self.best_cost_iter.append(min(self.solutions_iter, key = lambda x: x["cost"])["cost"])
    self.best_cost_over_time.append(self.best_cost)

  def update_user(self, iter_curr):
    if iter_curr % 100 == 0:
      if self.ACS: 
        s = "Running ACS with {} nodes, current iteration: {} out of {}".format(self.graph.nn, iter_curr, self.max_iter)
      else:
        s = "Running Elitist ACO with {} nodes, current iteration: {} out of {}".format(self.graph.nn, iter_curr, self.max_iter)
      print(s, end = "\r")
      #sys.stdout.flush()
        
  def end(self, iter_curr, time_exec):
    self.time_exec = time_exec
    if self.max_time and time_exec > self.max_time:
      return True 
    elif self.max_iter and iter_curr > self.max_iter:
      return True 
    return False 

  def update(self, ant, iter_curr):
    self.solutions_iter.append({"path_mat": ant.path_mat, "cost": ant.cost})
    if ant.cost < self.best_cost:
      self.best_cost = ant.cost 
      self.best_iter = iter_curr
      self.best_start_time = ant.st
      self.best_path_mat = ant.path_mat
      self.best_path_vec = ant.path_vec
    
  def iter_over_sol_and_update(self, path_mat, c_tau, c_p, payload):
    indexes = numpy.where(path_mat == 1)
    if len(indexes) != 3: return
    a, b, c = indexes[0].tolist(), indexes[1].tolist(), indexes[2].tolist()
    for (t, i, j) in zip(a, b, c):
      val = c_tau*self.graph.pm[t,i,j] + c_p*payload
      self.graph.update_pm(t, i, j, val)

  def acs_update(self):
    path_mat, c_tau, c_p, payload = self.best_path_mat, self.alpha, self.rho, 1/self.best_cost 
    self.iter_over_sol_and_update(path_mat, c_tau, c_p, payload)
    #print(self.graph.pm.max(),self.graph.pm.min(), self.graph.tau_zero)

  def local_update(self, path_mat):
    c_tau, c_p, payload = (1-self.zeta), self.zeta, self.graph.tau_zero 
    self.iter_over_sol_and_update(path_mat, c_tau, c_p, payload)

  def create_update_set(self):
    S_upd = self.solutions_iter
    S_upd = sorted(S_upd, key = lambda x: x["cost"])
    S_upd = S_upd[0:self.num_deposit]
    for item in S_upd:
      item["g"] = 1/item["cost"]
    elitist = {"cost": self.best_cost, "path_mat": self.best_path_mat, "g": self.c_elite/self.best_cost}
    S_upd.append(elitist) 
    return S_upd

  def elitist_update(self):
    S_upd = self.create_update_set()
    # pheromone evaporation
    self.graph.pm *= self.alpha
    # deposit on entries belonging to S_upd 
    for sol in S_upd:
      path_mat, c_tau, c_p, payload = sol["path_mat"], 1, 1, sol["g"]
      self.iter_over_sol_and_update(path_mat, c_tau, c_p, payload)

  def result(self):
    obj_return = {
      "ACS": self.ACS,
      "num_nodes": self.graph.nn,
      "start_time": self.best_start_time,
      "cost": self.best_cost,
      "path": self.best_path_vec,
      #"cost_at_iter": self.best_cost_iter,
      #"time_at_iter": self.time_iter,
      #"best_cost_over_time": self.best_cost_over_time,
      "daemon": self.use_daemon,
      "execution_time": self.time_exec,
      "num_iterations": self.iterations_completed,
    }
    if self.opt_cost:
      obj_return["rel_error_final"] = (self.best_cost-self.opt_cost)*100/self.opt_cost
      #obj_return["opt_tsp_cost"] = self.opt_cost
      #rel_error_iter, rel_error_best = calc_rel_err(self.best_cost_iter, self.opt_cost), calc_rel_err(self.best_cost_over_time, self.opt_cost)
      #obj_return["rel_error_iter"] = rel_error_iter
      #obj_return["rel_error_best"] = rel_error_best
    print(self.ACS)
    print(self.graph.nn)
    print(self.best_start_time)
    print(self.best_cost)
    print(self.best_path_vec)
    return obj_return
    
  def save_result(self, f_path = None):
    if not f_path:
      f_path =  'C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python36_64\\TIWID\\Tests\\TDTSP\\ACO\\{}-{}.json'.format(self.graph.nn, str(uuid.uuid4()))
    result = self.result()
    with open(f_path, 'a+') as f:
      json.dump(result, f)

  def plot_result(self, f_path = None):
    if not f_path:
      f_path =  'C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python36_64\\TIWID\\AntColOpt\\tests\\results\\{}-{}.png'.format(self.name, str(uuid.uuid4()))
    result = self.result()
    cost_iter, opt_tsp_cost = result["best_cost_over_time"], result["opt_tsp_cost"]
    static_cost = [opt_tsp_cost for _ in range(len(cost_iter))]
    x = list(range(len(cost_iter)))
    plt.plot(x, cost_iter, label = "ACO cost")
    plt.plot(x, static_cost, label = "optimal cost")
    plt.title(self.name)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.legend()
    plt.savefig(f_path)

  def plot_two(self):
    f_path =  'C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python36_64\\TIWID\\AntColOpt\\tests\\tsp\\{}-{}.png'.format(self.name, str(uuid.uuid4()))
    result = self.result()
    #cost_iter = result["best_cost_over_time"]
    #cost_iter = result["cost_at_iter"]
    #cost_iter = result["rel_error_best"] 
    cost_iter = result["rel_error_iter"]
    time_iter, opt_tsp_cost =  result["time_at_iter"], result["opt_tsp_cost"]
    
    fig, ax1 = plt.subplots()
    
    x = list(range(len(cost_iter)))
    ln_1 = ax1.plot(x, cost_iter, 'b.', label = "relative error (%)")
    #ln_2 = ax1.plot(x, list(opt_tsp_cost for _ in range(len(x))), label = "optimal cost")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("relative error (%)", color = "b")

    ax2 = ax1.twinx()
    ln_3 = ax2.plot(x, time_iter, color = "r", label = "time")
    ax2.set_ylabel("time (s)", color = "r")

    # added these three lines
    #lns = ln_1+ln_2+ln_3
    lns = ln_1+ln_3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc = "upper left", bbox_to_anchor=(1.1,1))
    plt.tight_layout(pad = 1.5)
    
    title = "{} - system: {} - nodes: {}".format(self.name, "ACS" if self.ACS else "Elitist", self.graph.nn)
    plt.title(title)
    plt.savefig(f_path)
    plt.close()


class Ant(object):
  def __init__(self, id, sn, fn, st, ntv, colony):
    self.id = id                    # ant id
    self.sn = sn                    # start node 
    self.fn = fn                    # final node 
    self.st = st                    # start time 
    self.ntv = ntv                  # nodes to visit
    self.colony = colony            # ant colony
    self.graph = self.colony.graph  # ant graph
    self.init_books()

  def init_books(self):
    self.cost = 0
    self.path_mat = numpy.full(self.graph.dims, 0)
    self.path_vec = [self.sn]

  def run(self):
    #print("Ant {} is running".format(self.id))
    #print("start_node: {}, end_node: {}, ntv: {}".format(self.sn, self.fn, self.ntv))
    node_curr = self.sn 
    time_curr = self.st 
    while self.ntv:
      node_new = self.transition_rule(node_curr, time_curr)
      self.path_mat[time_curr, node_curr, node_new] = 1
      self.path_vec.append(node_new)
      self.cost += self.graph.cm[time_curr, node_curr, node_new]
      node_curr = node_new
      time_curr += self.graph.wp[node_curr - 1]
    # close tour
    self.path_vec.append(self.fn)
    self.path_mat[time_curr, node_curr, self.fn] = 1
    self.cost += self.graph.cm[time_curr, node_curr, self.fn] 
    if type(self.cost) != int:
      self.cost = numpy.asscalar(self.cost)
    # run daemon actions 
    self.run_daemon()
    #print("Ant {} finished; path created {}".format(self.id, self.path_vec))
  
  def run_daemon(self):
    if self.colony.use_daemon:
      is_time_dependent = len(self.colony.graph.cm) > 1
      res_daemon = misc.twoOpt(self.path_vec, self.graph.cm, is_time_dependent)
      cc = self.cost
      self.path_vec, self.path_mat, self.cost = res_daemon["path_vec"], res_daemon["path_matrix"], res_daemon["cost"]
      #print("before daemon: {}, after: {}".format(cc, self.cost))

  def transition_rule(self, node_curr, time_curr):
    if self.colony.ACS:
      node_new = self.acs_transition(node_curr, time_curr)
    else:
      node_new = self.as_transition(node_curr, time_curr)
    self.ntv.remove(node_new)
    return node_new

  # Ant Colony System Transition Rule
  def acs_transition(self, node_curr, time_curr):
    q = random.random()
    if q < self.colony.q0:                     
      node_new = self.exploitation(node_curr, time_curr)
    else:
      node_new = self.exploration(node_curr, time_curr)
    return node_new

  # Ant System Transition rule
  def as_transition(self, node_curr, time_curr):
    node_new = self.exploration(node_curr, time_curr)
    return node_new

  def exploitation(self, node_curr, time_curr):
    values = [self.numerator(node_curr, node, time_curr) for node in self.ntv]
    max_index = values.index(max(values))
    return self.ntv[max_index]

  def exploration(self, node_curr, time_curr):
    den = self.denominator(node_curr, time_curr)
    if den == 0:
      den = 1
    values = [self.numerator(node_curr, node, time_curr)/den for node in self.ntv]
    max_index = values.index(max(values))
    return self.ntv[max_index]

  def numerator(self, i, j, t):
    return self.graph.pm[t, i, j] * math.pow(self.graph.etha(t, i, j), self.colony.beta)

  def denominator(self, i, t):
    den = 0
    for node in self.ntv:
      den += self.numerator(i, node, t)
    return den 


