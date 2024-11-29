import numpy as np
import time
import networkx as nx
import random
from sporadics import m1
import sat_colorability_solver
from graph import Graph

#fast triangle free checker
def check_triangle_free(g):
  adj = np.array(g.adjMatrix)
  squ = np.matmul(adj, adj)
  tri = np.matmul(squ, adj)
  trace = np.trace(tri)
  if trace == 0:
    return True
  else:
    return False

#slow unoptimized triangle identifier
def print_triangles(g):
  for i in range(g.size):
    for j in range(g.size):
      for k in range(g.size):
        if (g.adjMatrix[i][j] == 1 and g.adjMatrix[j][k] == 1 and g.adjMatrix[i][k] == 1):
          print(i, j, k)

#Geeks for geeks
def check_two_colorable(adjMatrix):
  for x in range(len(adjMatrix)):
    colorArr = [-1] * len(adjMatrix)
    colorArr[x] = 1
    queue = []
    queue.append(x)
    while queue:
      u = queue.pop()
      if adjMatrix[u][u] == 1:
        return False
      for v in range(len(adjMatrix)):
        if adjMatrix[u][v] == 1 and colorArr[v] == -1:
          colorArr[v] = 1 - colorArr[u]
          queue.append(v)
        elif adjMatrix[u][v] == 1 and colorArr[v] == colorArr[u]:
          return False
  return True


#get list of neighbors of a vertex
def N(vertex, adjMatrix):
  c = 0
  l = []
  for i in adjMatrix[vertex]:
    if i == 1:
      l.append(c)
    c += 1
  return l

#some algorithm for clique testing
def bronkerbosch(r, p, x, g):
  if len(p) == 0 and len(x) == 0:
    g.append_clique(r)
  for vertex in p[:]:
    r_new = r[:]
    r_new.append(vertex)
    p_new = [val for val in p if val in N(vertex, g.adjMatrix)]
    x_new = [val for val in x if val in N(vertex, g.adjMatrix)]
    bronkerbosch(r_new, p_new, x_new, g)
    p.remove(vertex)
    x.append(vertex)
  g.cliques.sort(key=len)
  return g

#helper for next func
def check_mpg_complement_inner(g, fast=False, silent=False):
  if not silent:
    print("Valid coloring list:")
  colorings = sat_colorability_solver.allSolutions(g.size, g.edge_list())
  if colorings == []:
    return None # this means no coloring found

  edge_add = []
  for coloring in colorings:
    if (not silent): print(coloring)
    while (edge := check_can_add(g, coloring, fast=True, silent=silent)) != (-1, -1):
      edge_add += [edge]
      if fast: 
        return edge_add
    
  return edge_add

#check if graph is mpgc
def check_mpg_complement(g, fast=False, silent=False):
  t1 = time.time()
  if not is_complement_connected(g):
    if not silent:
      print("MPG is not connected.")
    return False
  if check_triangle_free(g):
    minimal = check_mpg_complement_inner(g, fast=fast, silent=silent)
    

    t2 = time.time()
    if not silent:
      print("Time taken:", str(t2 - t1), "seconds")
    if (minimal == None):
      if not silent:
        print("Graph is not colorable")
      return False
    if (minimal == []):
      if not silent:
        print("Graph is minimal")
      return True
    else:
      if not silent:
        print("Not minimal")
      return False
  if not silent:
    print("Not triangle free")
  t2 = time.time()
  if not silent:
    print("Time taken:", str(t2 - t1), "seconds")
  return False

#find if edge can be added given graph and coloring. returns edge if possible, otherwise (-1, -1)
def check_can_add(g, c, fast=False, silent=False, permutation=None):
  if permutation == None:
    permutation = range(g.size)
  can_add = (-1, -1)
  for k in range(len(permutation)):
    u = permutation[k]
    for l in range(k+1,len(permutation)):
      v = permutation[l]
      if (c[u] != c[v]) and (g.adjMatrix[u][v] == 0):
        g.adjMatrix[u][v] = 1
        if check_triangle_free(g):
          if not silent:
            print("Added edge between nodes", str(u), "and", str(v))
          can_add = (u, v)
          if (fast): return can_add
        g.adjMatrix[u][v] = 0
  return can_add


#input "g" is the complement of a graph
#tells if g represents the complement of a superminimal prime graph
#Need to consider graphs of size <=2 ???
#Also requires graph to be connected
def check_superminimal_complement_helper(g, domain=None, original_coloring=None, fast=False, silent=False):
  removable = []
  if (domain == None):
    domain = range(g.size)
  for v in domain:
    if not silent:
      print("---------------- Removing vertex", str(v))
    h = g.remove_vertex(v)
    
    if (original_coloring != None):
      if not check_mpg_complement(
          h,
          fixed_coloring={i:(original_coloring[:v] + original_coloring[v + 1:])[i] for i in range(g.size - 1)},
          fast=fast, silent=silent):
        continue
    if check_mpg_complement(h, fast=fast, silent=silent):
      if not silent:
        print("Vertex", v, "was removed!")
      removable.append(v)
      if (fast):
        return False
  if not silent:
    print("----------------")
  if (len(removable) == 0):
    if not silent:
      print("Graph is superminimal!")
    return True
  else:
    if not silent:
      print("Graph is not minimal!")
      print(f"Can remove {removable}.")
    return False

#check if graph is superminimal complement lol
def check_superminimal_complement(g, fast=False, silent=False):
  if check_mpg_complement(g, fast=fast, silent=silent):
    return check_superminimal_complement_helper(g, fast=fast, silent=silent)
  return False

#check if coloring is valid on graph g, with edge list e. returns first failing edge or (-1, -1) if successful
def check_coloring(g, c, e):
  for (u, v) in e:  #ordered from smallest to largest
    if (c[u] == c[v]): return (u, v)
  return (-1, -1)

#compare to graphs to see if they are isomorphic
def compgraph(g, h):
  g1 = nx.Graph()
  g1.add_edges_from(g.edge_list())
  h1 = nx.Graph()
  h1.add_edges_from(h.edge_list())
  return nx.is_isomorphic(g1, h1)

#check if complement is connected 
def is_complement_connected(g):
  g1 = nx.Graph()
  g1.add_nodes_from(range(g.size))
  for i in range(g.size):
    for j in range(i + 1, g.size):
      if (g.adjMatrix[i][j] == 0):
        g1.add_edge(i, j)
  return nx.is_connected(g1)

#generate cycle
def generateCycle(n):
  g = Graph(n)
  g.add_edge(0, n - 1)
  for i in range(n-1):
    g.add_edge(i, i + 1)
  return g
#generate N graph, predecessor to W graphs, definitely fixed coloring mpgc but not necessarily superminimal
def generateNgraph(n):
  if (n % 3 != 0):
    raise Exception()
  n //= 3
  edge_list = []
  rows = [range(0, n), range(n, 2 * n), range(2 * n, 3 * n)]
  for row in range(2):
    for i in range(n):
      if (i % 2 == 0):
        for j in range(n):
          if (i - j <= 1 and j - i <= 1):
            edge_list.append((rows[row][i], rows[row + 1][j]))
      else:
        edge_list.append((rows[row][i], rows[row + 1][i]))
  for i in range(n):
    if (i % 2 == 0):
      for j in range(n):
        if (i - j < -1 or j - i < -1):
          edge_list.append((rows[0][i], rows[2][j]))
    else: 
      for j in range(n):
        if (i != j):
          edge_list.append((rows[0][i], rows[2][j]))
  g = Graph(3 * n)
  for edge in edge_list:
    g.add_edge(edge[0], edge[1])
  return g

'''
  2   6   10    14
  1 3 5 7  9 11 
0   4   8    12
'''
#generate W graph as defined by me, proven to be superminimal, unique coloring
def generateWgraph(n): #n is multiple of 2
  if (n % 2 != 0): raise Exception()
  edge_list = []
  low = [i for i in range(n + 1) if (i % 4 == 0)]
  mid = [i for i in range(n - 1) if (i % 2 == 1)]
  high  = [i for i in range(n + 1) if (i % 4 == 2)]

  for i in mid:
    edge_list.append((i, i - 1))
    edge_list.append((i, i + 1))
    edge_list.append((i, i + 3))
  for i in high:
    for j in low:
      if (i - j != 2 and j - i != 2):
        edge_list.append((i, j))
  g = Graph(n + 1)
  for (v1, v2) in edge_list:
    g.add_edge(v1, v2)
  return g.remove(n - 1)

#generate g graph as defined by other paper, proven to be superminimal
def generateGgraph(n):
  if(n%6 != 0 and n%6 != 5): raise Exception()
  k = (n+2)//6
  edge_list = []
  for i in range(n):
    for l in range(k, 2*k):
      edge_list.append([i, (i + l) % n])
  g = Graph(n)
  for (v1, v2) in edge_list:
    g.add_edge(v1, v2)
  return g

#generate P graph, i forgot what this one is tbh but it might have some superminimals idk
def generatePgraph(n):
  edge_list = []
  if(n%3 == 2): 
    for i in range(n):
      for j in range(i + 1, n):
        if ((j - i) % 3 == 1):
          edge_list.append([i, j])
  else:
    raise Exception()
  g = Graph(n)
  for (v1, v2) in edge_list:
    g.add_edge(v1, v2)
  return g


#generate O graph, not superminimal family but has some superminimal components
def generateOgraph(n):
  if n % 4 != 2:
    raise Exception()
  n //= 2
  i = Graph(2*n)
  [i.add_edge(j, (j+1)%n) for j in range(n)]
  [i.add_edge(j+n, (j+1)%n+n) for j in range(n)]
  for k in range(0, n - 2, 2):
    [i.add_edge(j+n, (j+k)%n) for j in range(n)]
  return i
  
#generate a ship graph with n vertices, conjectured to always be superminimal
def generateShipgraph(n):
  if n % 3 == 1:
    raise Exception()

  k = (n - 2)//3
  if n % 3 == 0:
    g = generateShipgraph(n - 1)
    g.add_node()
    g.remove_edge(0, n - 2)
    g.add_edge(0, n - 1)
    g.add_edge(n - 2, n - 1)
    for i in range(k + 1, 2*k + 1):
      g.add_edge(i, n - 1)
    return g
  
  g = Graph(n)
  for i in range(2, 2*k):
    for j in range(k, 2*k):
      if (i + j < n - 2):
        g.add_edge(i, i + j)
  for i in range(n - k - 1, n):
    g.add_edge(0, i)
  for i in range(k + 1):
    g.add_edge(n - 1, i)
  g.add_edge(1, k+1)
  g.add_edge(1, k+2)
  g.add_edge(1, n-2)
  for i in range(k + 3, 2*k + 1):
    g.add_edge(n - 2, i)
  return g

#print matrix in good format for visulaization
def printm(matrix):
  for i in range(len(matrix)):
    [print("0" if matrix[i][j] == 0 else "1", end=" ") for j in range(len(matrix[i]))]
    print()

#generate mpgc, not necessarily accurate
def generateRandomMinimal(n):
  g = Graph(n)
  coloring = [random.randrange(0,3) for i in range(n)]
  addRandomMinimal(g, coloring)
  return [g, coloring]

#given a unique coloring, add arbitrary edges to graph until it becomes a mpgc (likely not superminimal)
def addRandomMinimal(g, c):
  nodes = list(range(g.size))
  random.shuffle(nodes)
  edge = check_can_add(g, c, fast=True, permutation=nodes, silent=True)
  while edge != (-1, -1):
    g.add_edge(*edge)
    random.shuffle(nodes)
    edge = check_can_add(g, c, fast=True, permutation=nodes, silent=True)
  return g

#util for permuting the order of vertices in a graph
def permute(g, perm):
  new_adj = [[0]*g.size for i in range(g.size)]
  for i in range(g.size):
    for j in range(g.size):
      new_adj[i][j] = g.adjMatrix[perm[i]][perm[j]]
  g.adjMatrix = new_adj

#check the conjecture that any 4-path in a mpgc is also part of a 5-cycle
def find5cycle(g, c):
  for (u, v) in g.edge_list():
      for w in range(g.size):
        if (g.adjMatrix[v][w] != 1 or w == u):
          continue
        for x in range(g.size):
          if (g.adjMatrix[w][x] != 1 or c[u] != c[x] or x in [u, v]):
            continue
          cycle = False
          for y in range(g.size):
            if (y in [u, v, w, x]):
              continue
            if (g.adjMatrix[x][y] == 1 and g.adjMatrix[y][u] == 1):
              cycle = True
          if (not cycle):
            print(u, v, w, x, "not part of a cycle!!!!")
            return True
  return False

#checks the conjecture that every edge in a mpgc is part of a 5-cycle
def find5cycle_strong(g):
  for (u, v) in g.edge_list():
      cycle = False
      for w in range(g.size):
        if (g.adjMatrix[v][w] != 1 or w == u):
          continue
        for x in range(g.size):
          if (g.adjMatrix[w][x] != 1 or x in [u, v]):
            continue
          for y in range(g.size):
            if (y in [u, v, w, x]):
              continue
            if (g.adjMatrix[x][y] == 1 and g.adjMatrix[y][u] == 1):
              if (u == 0 and v == 17):
                print(u, v, w, x, y)
              cycle = True
      if (not cycle):
        print(u, v, "not part of a cycle!!!!")
        return True
  return False  

#use nx to find any induced n-cycles of g
def find_induced_n_cycle(g, n):
  g1 = nx.Graph()
  g1.add_edges_from(g.edge_list())
  h1 = nx.cycle_graph(n)
  gm = nx.isomorphism.GraphMatcher(g1, h1)
  print(gm.subgraph_is_isomorphic())
  return gm.subgraph_isomorphisms_iter()

#generate the brinkmann graph, a square and triangle free 4 chromatic graph
def brinkmann_graph():
  g = Graph(21)
  for i in range(7):
    g.add_edge(i, (i+3)%7)
    g.add_edge(i, i + 7)
    g.add_edge(i, (i + 2)%7 + 7)
    g.add_edge(i + 7, i + 14)
    g.add_edge((i + 1)%7 + 7, i + 14)
    g.add_edge(i + 14, (i + 2)%7 + 14)
  return g

#find k-neighborhood of a graph g
def kneighbors(g, k):
  return [N(i, g.adjMatrix) for i in range(g.size)] if k == 1 else [{j for n in x for j in kneighbors(g, 1)[n]} for x in kneighbors(g, k - 1)]

#generate actually guaranteed minimal prime graphs
def guaranteedMPG(n):
  [g, c] = generateRandomMinimal(n)
  if (c.count(0) > len(c)/2 or c.count(1) > len(c)/2 or c.count(2) > len(c)/2):
    return guaranteedMPG(n)
  colorings = sat_colorability_solver.allSolutions(g.size, g.edge_list())
  if (len(colorings) > 100):
    #print(len(colorings), " too long.")
    return guaranteedMPG(n)
  #print(len(colorings))
  for coloring in colorings:
    if (check_coloring(g, coloring, g.edge_list())):
      while (edge := check_can_add(g, coloring, fast=True, silent=True)) != (-1, -1):
        g.add_edge(*edge)
        #print(edge)

  if is_complement_connected(g):
    return g

  return guaranteedMPG(n)

#check proportion of graphs with properties
def proportionOfMPG(n, iters, pred, epoch=20):
  t = time.time()
  count = 0
  for i in iters:
    g = guaranteedMPG(n)
    if (pred(g)):
      count += 1

    if i % 20 == 0:
      print(f"{count} / {i} at {count/i} percent in {time.time()-t}")

def construct_from_adj(adj):
  g = Graph(len(adj))
  g.adjMatrix = adj
  return g

def construct_from_edge_list(edges):
  g = Graph(max({v for edge in edges for v in edge}) + 1)
  for e in edges:
    g.add_edge(*e)
  return g

def generateSUPERMINIMAL(v):
  g = guaranteedMPG(v)
  end = False
  while (not end):
    end = True
    
    r = list(range(g.size))
    random.shuffle(r)
    
    for v in r:
      g2 = Graph(g.size)
      g2.adjMatrix = g.adjMatrix
      if check_mpg_complement(g2.remove(v), fast=True, silent=True):
        g.remove(v)
        end = False
        break
        
  return g


#dont recompute this between function executions
sporadics = []
for m in m1:
  sporadics += [construct_from_edge_list(m)]

#check if a superminimal graph is isomorphic to a graph that has already been found
def check_duplicate(g):
  #families
  superminimal_families = [
    generateShipgraph,
    generateWgraph,
    generatePgraph
  ]
  
  for i in range(len(sporadics)):
    gr = sporadics[i]
    if (gr.size == g.size):
      if (compgraph(gr, g)):
        #print(f"isomorphic to sporadic {i}")
        return True

  for i in range(len(superminimal_families)):
    f = superminimal_families[i]
    try:
      gr = f(g.size)
      if (compgraph(gr, g)):
        #print(f"isomorphic to family {i}")
        return True
        
    except: 
      continue

  return False

#create graphs with given size, and check if they are new superminimal graphs  
def looped_superminimal_finder(size, iterations):
  #random.setstate(pickle.loads(b))
  #random.seed(6)
  graphs = []
  for i in range(iterations):
    if (i % 25 == 0):
      print(i)
    #print(i, end=": ")
    g = generateSUPERMINIMAL(size)
    if (not check_duplicate(g)):
      graphs += [g]
      print("NEW SUPERMINIMAL?")
      print(g.edge_list())

  final = []
  for g in graphs:
    iso = False
    for l in final:
      if (compgraph(g, l)):
        iso = True
    if not iso:
      final += [g]
  print(f"Purged {len(graphs) - len(final)} duplicates. {len(final)} remaining.")
  print("FINAL GRAPHS ARE: \n\n\n")
  for g in final:
    print(g.edge_list(), end=',\n')
      
  #print(pickle.dumps(random.getstate()))

def main():
  amnt = {}
  count= 0
  for m in m1:
    m = construct_from_edge_list(m).adjMatrix
    #printm(m)
    #print()
    amnt[len(m)] = amnt.get(len(m), 0)+1
    count += 1
  print(dict(sorted(amnt.items())))
  print(count)
  for i in range(7, 20):
    print(i)
    looped_superminimal_finder(i, 500)
  
  #below is a potential strategy for forming a new family...
  '''
  n = 6 * 3
  #top = {4}
  bottom = {0, 3, 6}
  g = Graph(n)
  for i in range(n):
    #for j in top:
    #  g.add_edge(i, (i + 2 * j) % n)
    if (i % 2 == 0):
      for j in bottom:
        g.add_edge(i, (i + 2 * j + 1) % n)
    g.add_edge(i, (i+2) % n)
  g.add_edge(0, 8)
  g.add_edge(2, 10)
  g.add_edge(4, 12)
  g.add_edge(6, 14)
  g.add_edge(1, 9)
  g.add_edge(3, 11)
  g.add_edge(5, 13)
  g.add_edge(7, 15)
  print_triangles(g)
  check_superminimal_complement(g)
  print(g.edge_list())
  '''

  #looped_superminimal_finder()
  
  

if __name__ == '__main__':
  main()

