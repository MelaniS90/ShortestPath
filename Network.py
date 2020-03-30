from .Node import Node
from .Edge import Edge
from math import sqrt
from matplotlib import pyplot as plt


class Network(object):
    def __init__(self):
        pass

    nodes = []
    edges = []
    nodes_for_algorithm = []
    graph={}

    #czytaj graf z pliku
    def Graph(self):
        try:
            with open('data.txt', 'r') as input_file:
                for line in input_file:
                    if line.lstrip().startswith('#'):
                        continue
                    elif line.lstrip().startswith(('WEZLY', 'LACZA')):
                        present = line.split('=')[0].strip(' \n')
                    elif line.lstrip().startswith('ALGORYTM'):
                        present = 'ALGORYTM'
                        self.algorithm = line.split('=')[1].strip(' \n')
                    else:
                        if present == 'WEZLY':
                            values = [int(p.strip(' \n')) for p in line.split(' ')]
                            self.nodes.append(Node(*values))
                        elif present == 'LACZA':
                            values = [int(p.strip(' \n')) for p in line.split(' ')]
                            print(values)
                            self.edges.append(Edge(*values))



                        else:
                            values = [int(p.strip(' \n')) for p in line.split(' ')]
                            self.nodes_for_algorithm.append((values[0], values[1]))
        except FileNotFoundError:
            print('Nie znaleziono pliku "data.txt"')

    def nodes_weight(self):
        for edge in self.edges:
            edge.weight=int(sqrt(((self.nodes[edge.start_node-1].x - self.nodes[edge.end_node-1].x )**2)
                                 +((self.nodes[edge.end_node-1].y - self.nodes[edge.start_node-1].y)**2)))

    def create_neigh_nodes(self, v, g):
        # tworzymy macierz zerowa o wymiarach [vxv]
        neigh_nodes = []
        for i in range(0, v):
            neigh_nodes.append([])
            for j in range(0, v):
                neigh_nodes[i].append(0)

        for i in range(0, len(g)):
            # g[i][0] - pierwszy wierzcholek w obiekcie klasy Edge
            # g[i][1] - drugi wierzcholek w obiekcie klasy Edge
            # g[i][2] - waga krawedzi pomiedzy tymi wierzcholkami
            neigh_nodes[g[i][0]][g[i][1]] = g[i][2]
            neigh_nodes[g[i][1]][g[i][0]] = g[i][2]

        return neigh_nodes

    def find_connection(self, start, goal):
        # funkcja sprawdzajaca, czy jest bezposrednie polaczenie
        # miedzy dwoma wierzcholkami
        for edge in self.edges:
            if (int(edge.start_node) == int(start)
                    and int(edge.end_node) == int(goal)):
                return edge.id_edge
            # delete it when you need directed connections

            elif (int(edge.end_node) == int(start)
                  and int(edge.start_node) == int(goal)):
                return edge.id_edge
        return 0

    # zwroc MST- Prim
    def minimumSpanningTree(self):
        graph = []  # na poczatku graf jest pusty
        edges_mst = []  # wynik koncowy - mst w formie listy obiektow klasy Edge
        total_weight = 0  # dlugosc drzewa rozpinajacego

        for edge in self.edges:
            graph.append([edge.start_node - 1, edge.end_node - 1, edge.weight])  # dodanie do grafu wierzcholkow

        neigh_nodes = self.create_neigh_nodes(len(self.nodes), graph)
        # stworzenie macierzy wierzcholkow bezposrednio ze soba polaczonych

        node = 0
        mst = []  # minimalne drzewo rozpinajace w formie tablicy dwuwymiarowej
        potential_edges = []  # wszystkie krawedzie
        visited = []  # odwiedzone krawedzie
        min_edge = [None, None, float('inf')]  # na poczatku minimalna krawedz ma dlugosc nieskonczonosc

        while len(mst) != len(self.nodes) - 1:  # krawedzi jest o 1 mniej niz wierzcholkow (mst to krawedzie)

            # mark this node as visited
            visited.append(node)

            # add each edge to list of potential edges
            for edge_id in range(0, len(self.nodes)):  # sprawdzamy kazda krawedz
                if neigh_nodes[node][edge_id] != 0:  # dlugosc pomiedzy dwoma prawidlowymi wierzcholkami jest niezerowa
                    potential_edges.append([node, edge_id, neigh_nodes[node][edge_id]])

            # find edge with the smallest weight to a node
            # that has not yet been visited
            for e in range(0, len(potential_edges)):  # przechodzimy wszystkie potencjalne krawedzie
                # jesli dlugosc potencjalnej krawedzi jest mniejsza od dlugosci minimalnej krawedzi
                # i edge_id potencjalnej krawedzi nie jest juz odwiedzone,
                # zastap obecna minimalna krawedz, krotsza od niej potencjalna
                if potential_edges[e][2] < min_edge[2] and potential_edges[e][1] not in visited:
                    min_edge = potential_edges[e]

            # remove min weight edge from list of potential_edges
            potential_edges.remove(min_edge)

            # push min edge to MST
            mst.append(min_edge)

            # start at new node and reset min edge
            # przechodzimy do wierzcholka polaczonego przed chwila wyliczona minimalna krawedzia
            node = min_edge[1]
            # resetujemy minimalna krawedz
            min_edge = [None, None, float('inf')]

        # przepisanie tablicy dwuwymiarowej na obiekty klasy Edge
        for i in mst:
            j = self.find_connection(i[0] + 1, i[1] + 1) - 1
            edges_mst.append(self.edges[j])
            total_weight += self.edges[j].weight
        print(f'Total weight of the edges in minimum spanning tree: {total_weight}')
        print()
        print(f'MST = {mst}\n'f'Edge_MST = {edges_mst}')
        #self.draw(edges_mst, self.nodes)

        return edges_mst


    def generate_graph_dict(self):
        graph = {}
        for node in self.nodes:
            node_neighbours = {}
            for edge in self.edges:
                if node.id_node == edge.start_node:
                    node_neighbours[edge.end_node] = edge.weight
                elif node.id_node == edge.end_node:
                    node_neighbours[edge.start_node] = edge.weight
            graph[node.id_node] = node_neighbours
        return graph

    # zwroc najkrotsza sciezke - Dijkstra
    def Dijkstra(self , id_start , id_end):
        graph = self.generate_graph_dict()

        # dictionary to record the cost to reach the node.
        # We will constantly update this dictionary as we move along the graph.
        shortest_distance = {}

        # dictionary to keep track of path that led to that node.
        track_predecessor = {}

        unseenNodes = graph  # to iterate through all nodes
        infinity = 9999  # infinity can be considered a very large number
        track_path = []  # dictionary to record as we trace back our journey

        # =============================================================================
        # Initially we want to assign 0 as the cost to reach to source node and infinity as cost to all other nodes
        # =============================================================================

        for node in unseenNodes:
            shortest_distance[node] = infinity
        shortest_distance[id_start] = 0

        # =============================================================================
        # The loop will keep running until we have entirely exhausted the graph, until we have seen all the nodes
        # =============================================================================
        # =============================================================================
        # To iterate through the graph, we need to determine the min_distance_node every time.
        # =============================================================================

        while unseenNodes:
            min_distance_node = None

            for node in unseenNodes:
                if min_distance_node is None:
                    min_distance_node = node

                elif shortest_distance[node] < shortest_distance[min_distance_node]:
                    min_distance_node = node
            print(min_distance_node)

            # =============================================================================
            # From the minimum node, what are our possible paths
            # =============================================================================

            path_options = graph[min_distance_node].items()
            print(path_options)

            # =============================================================================
            # We have to calculate the cost each time for each path we take and only update it if it is lower than the existing cost
            # =============================================================================

            for (child_node, weight) in path_options:

                if weight + shortest_distance[min_distance_node] < shortest_distance[child_node]:
                    shortest_distance[child_node] = weight + shortest_distance[min_distance_node]

                    track_predecessor[child_node] = min_distance_node

            # =============================================================================
            # We want to pop out the nodes that we have just visited so that we dont iterate over them again.
            # =============================================================================
            unseenNodes.pop(min_distance_node)

        # =============================================================================
        # Once we have reached the destination node, we want trace back our path and calculate the total accumulated cost.
        # =============================================================================

        currentNode = id_end

        while currentNode != id_start:

            try:
                track_path.insert(0, currentNode)
                currentNode = track_predecessor[currentNode]
            except KeyError:
                print('Path not reachable')
                break
        track_path.insert(0, id_start)

        # =============================================================================
        #  If the cost is infinity, the node had not been reached.
        # =============================================================================
        if shortest_distance[id_end] != infinity:
            print('Shortest distance is ' + str(shortest_distance[id_end]))
            print('And the path is ' + str(track_path))

        return track_path

    def draw_all(self):
        plt.xlim((0, 100))
        plt.ylim((0, 100))

        for edge in self.edges:
            #print('[{0}]'.format(edge.id_node))
            plt.plot([self.nodes[edge.start_node - 1].x,
                      self.nodes[edge.end_node - 1].x],
                     [self.nodes[edge.start_node - 1].y,
                      self.nodes[edge.end_node - 1].y],
                     'b-')

            plt.text((self.nodes[edge.start_node - 1].x + self.nodes[edge.end_node - 1].x)/2,
                     (self.nodes[edge.start_node - 1].y + self.nodes[edge.end_node - 1].y)/2,edge.weight)


        for node in self.nodes:
            plt.plot(node.x, node.y, 'ro')
            print('[{0}, {1}]'.format(node.x, node.y))

            plt.text(node.x,node.y,node.id_node,
                     bbox=dict(boxstyle="circle, pad=0.3",fc="violet"),fontsize=11)

        plt.show()

    def draw_path(self,path):
        plt.xlim((0,100))
        plt.ylim((0,100))

        x_axis=[self.nodes[path_element-1].x for path_element in path]
        y_axis=[self.nodes[path_element-1].y for path_element in path]
        plt.plot(x_axis , y_axis , 'r-')

        for node in self.nodes:
            plt.plot(node.x, node.y, 'bo')
            plt.text(node.x, node.y, node.id_node,
                 bbox=dict(boxstyle="circle, pad=0.3", fc="violet"), fontsize=11)

        plt.show()

    def draw_tree(self,edges,nodes):
        plt.xlim((0, 100))
        plt.ylim((0, 65))

        for edge in self.edges:
            # print('[{0}]'.format(edge.id_node))
            plt.plot([self.nodes[edge.start_node - 1].x,
                      self.nodes[edge.end_node - 1].x],
                     [self.nodes[edge.start_node - 1].y,
                      self.nodes[edge.end_node - 1].y],
                     'b-')

        for edge in edges:
            plt.plot([self.nodes[edge.start_node - 1].x,
                          self.nodes[edge.end_node - 1].x],
                         [self.nodes[edge.start_node - 1].y,
                          self.nodes[edge.end_node - 1].y],
                         'red')

        for node in self.nodes:
             plt.plot(node.x, node.y, 'ro')
             print('[{0}, {1}]'.format(node.x, node.y))

             plt.text(node.x, node.y, node.id_node,
             bbox=dict(boxstyle="circle, pad=0.3", fc="violet"), fontsize=11)

        plt.show()


    def hack_path(self):
        self.Graph()
        self.nodes_weight()
        self.edges=self.minimumSpanningTree()


        nodes_tab=[]
        path=[]

        path_d=self.Dijkstra(self.nodes_for_algorithm[0][0], self.nodes_for_algorithm[0][1])

        self.draw_path(path_d)

    def hack_node(self):
        self.Graph()
        self.nodes_weight()
        self.edges = self.minimumSpanningTree()
        elements=[]
        tab=[]

        for start_node in self.nodes:
            for end_node in self.nodes:
                if start_node != end_node:
                    self.Graph()
                    self.nodes_weight()
                    self.edges = self.minimumSpanningTree()
                    path_d = self.Dijkstra(start_node, end_node)
                    elements.extend(path_d)

        for node in self.nodes:
            tab.append(elements.count(node.id_node))
        print(elements)
        print(tab)





