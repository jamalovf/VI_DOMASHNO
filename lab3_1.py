from sys import maxsize as infinity
import bisect

class Queue:
def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)
def Stack():
    return []


class FIFOQueue(Queue):
    def __init__(self):
        self.A = []
        self.start = 0

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A) - self.start

    def extend(self, items):
        self.A.extend(items)

    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A) / 2:
            self.A = self.A[self.start:]
            self.start = 0
        return e

    def __contains__(self, item):
        return item in self.A[self.start:]


class PriorityQueue(Queue):
   def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)
class Problem:

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
       state, if there is a unique goal.  Your subclass's constructor can add
       other arguments."""
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """Given a state, return a dictionary of {action : state} pairs reachable
       from this state. If there are many successors, consider an iterator
       that yields the successors one at a time, rather than building them
       all at once. Iterators will work fine within the framework. Yielding is not supported in Python 2.7"""
        raise NotImplementedError

    def actions(self, state):
        """Given a state, return a list of all actions possible from that state"""
        raise NotImplementedError

    def result(self, state, action):
        """Given a state and action, return the resulting state"""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
       state to self.goal, as specified in the constructor. Implement this
       method if checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
       state1 via action, assuming cost c to get up to state1. If the problem
       is such that the path doesn't matter, this function will only look at
       state2.  If the path does matter, it will consider c and maybe state1
       and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
       and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________
# Definiranje na klasa za strukturata na jazel od prebaruvanje
# Klasata Node ne se nasleduva

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
   that this is a successor of) and to the actual state for this node. Note
   that if a state is arrived at by two paths, then there are two nodes with
   the same state.  Also includes the action that got us to this state, and
   the total path_cost (also known as g) to reach the node.  Other functions
   may add an f and h value; see best_first_graph_search and astar_search for
   an explanation of how the f and h values are handled. You will not need to
   subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "Return a child node from this node"
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def solve(self):
        "Return the sequence of states to go from the root to this node."
        return [node.state for node in self.path()[0:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        return list(reversed(result))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)




def memoize(fn, slot=None):
    """ Запамети ја пресметаната вредност за која била листа од
    аргументи. Ако е специфициран slot, зачувај го резултатот во
    тој slot на првиот аргумент. Ако slot е None, зачувај ги
    резултатите во речник.

    :param fn: зададена функција
    :param slot: име на атрибут во кој се чуваат резултатите од функцијата
    :return: функција со модификација за зачувување на резултатите
    """
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}
    return memoized_fn


def best_first_graph_search(problem, f):
 
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def greedy_best_first_graph_search(problem, h=None):
    """ Greedy best-first пребарување се остварува ако се специфицира дека f(n) = h(n).

    :param problem: даден проблем
    :param h: дадена функција за евристика
    :return: Node or None
    """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, h)


def astar_search(problem, h=None):
    """ A* пребарување е best-first graph пребарување каде f(n) = g(n) + h(n).

    :param problem: даден проблем
    :param h: дадена функција за евристика
    :return: Node or None
    """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


def recursive_best_first_search(problem, h=None):
    """Recursive best first search - ја ограничува рекурзијата
	преку следење на f-вредноста на најдобриот алтернативен пат
	од било кој јазел предок (еден чекор гледање нанапред).

    :param problem: даден проблем
    :param h: дадена функција за евристика
    :return: Node or None
    """
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0  # (втората вредност е неважна)
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Подреди ги според најниската f вредност
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = infinity
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, infinity)
    return result


def updateP1(P1):
    x = P1[0]
    y = P1[1]
    n = P1[2]
    if (y == 4 and n == 1):
        n = n * (-1)
    if (y == 0 and n == -1):
        n = n * (-1)
    ynew = y + n
    return (x, ynew, n)


def updateP2(P2):
    x = P2[0]
    y = P2[1]
    n = P2[2]
    if (x == 5 and y == 4 and n == 1):
        n = n * (-1)
    if (y == 0 and x == 9 and n == -1):
        n = n * (-1)
    xnew = x - n
    ynew = y + n
    return (xnew, ynew, n)


def updateP3(P3):
    x = P3[0]
    y = P3[1]
    n = P3[2]
    if (x == 5 and n == -1):
        n = n * (-1)
    if (x == 9 and n == 1):
        n = n * (-1)
    xnew = x + n
    return (xnew, y, n)


def isValid(x, y, P1, P2, P3):
    t = 1
    if (x == P1[0] and y == P1[1]) or (x == P1[0] and y == P1[1] + 1):
        t = 0
    if (x == P2[0] and y == P2[1]) or (x == P2[0] and y == P2[1] + 1) or (x == P2[0] + 1 and y == P2[1]) or (
            x == P2[0] + 1 and y == P2[1] + 1):
        t = 0
    if (x == P3[0] and y == P3[1]) or (x == P3[0] + 1 and y == P3[1]):
        t = 0
    return t


class Istrazuvac(Problem):
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        successors = dict()
        X = state[0]
        Y = state[1]
        P1 = state[2]
        P2 = state[3]
        P3 = state[4]
        P1new = updateP1(P1)
        P2new = updateP2(P2)
        P3new = updateP3(P3)

        # Desno
        if X < 5 and Y < 5:
            Ynew = Y + 1
            Xnew = X
            if (isValid(Xnew, Ynew, P1new, P2new, P3new)):
                successors['Desno'] = (Xnew, Ynew, P1new, P2new, P3new)
        elif X >= 5 and Y < 10:
            Xnew = X
            Ynew = Y + 1
            if (isValid(Xnew, Ynew, P1new, P2new, P3new)):
                successors['Desno'] = (Xnew, Ynew, P1new, P2new, P3new)
        # Levo
        if Y - 1 >= 0:
            Ynew = Y - 1
            Xnew = X
            if (isValid(Xnew, Ynew, P1new, P2new, P3new)):
                successors['Levo'] = (Xnew, Ynew, P1new, P2new, P3new)
        # Gore
        if Y >= 6 and X > 5:
            Xnew = X - 1
            Ynew = Y
            if (isValid(Xnew, Ynew, P1new, P2new, P3new)):
                successors['Gore'] = (Xnew, Ynew, P1new, P2new, P3new)
        elif Y < 6 and X > 0:
            Xnew = X - 1
            Ynew = Y
            if (isValid(Xnew, Ynew, P1new, P2new, P3new)):
                successors['Gore'] = (Xnew, Ynew, P1new, P2new, P3new)
        # Dolu
        if X < 10:
            Xnew = X + 1
            Ynew = Y
            if (isValid(Xnew, Ynew, P1new, P2new, P3new)):
                successors['Dolu'] = (Xnew, Ynew, P1new, P2new, P3new)
        return successors

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]

    def goal_test(self, state):
        g = self.goal
        return (state[0] == g[0] and state[1] == g[1])

    def h(self, node):
        rez = abs(node.state[0] - self.goal[0]) + abs(node.state[1] - self.goal[1])
        return rez


CoveceRedica = eval(input())
CoveceKolona = eval(input())
KukaRedica = eval(input())
KukaKolona = eval(input())


IstrazuvacInstance = Istrazuvac((CoveceRedica, CoveceKolona, (2, 2, -1), (7, 2, 1), (7, 8, 1)), (KukaRedica, KukaKolona))

answer = astar_search(IstrazuvacInstance).solution()
print(answer)