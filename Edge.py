
class Edge(object):
    def __init__(self, id_edge,start_node,end_node,weight=9999):
        self.id_edge=int(id_edge)
        self.start_node=int(start_node)
        self.end_node=int(end_node)
        self.weight=weight
