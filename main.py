import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

# Example data
V = np.array([[0,0], [1,0], [1,1], [0,1]])
T = np.array([[0,1,2], [0,2,3]])

def grid_triangulation(grid, subdivisions=1):
    '''
    :param grid: Numpy NxM grid of 0s and 1s, where 1s represent the presence of a square
    :param subdivisions: Number of subdivisions for each square
    :return: V, T where V is a list of vertices and T is a list of triangles (as indices into V)
    '''
    V = []
    T = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i,j] == 1:
                for x in range(subdivisions):
                    for y in range(subdivisions):
                        vertices = [(i*subdivisions + x, j*subdivisions + y), (i*subdivisions + x + 1, j*subdivisions + y), (i*subdivisions + x + 1, j*subdivisions + y + 1), (i*subdivisions + x, j*subdivisions + y + 1)]
                        for v in vertices:
                            if v not in V:
                                V.append(v)
                        idx = [V.index(v) for v in vertices]
                        # T.append([idx[0], idx[1], idx[2]])
                        # T.append([idx[0], idx[2], idx[3]])
                        T.append([idx[0], idx[1], idx[3]])
                        T.append([idx[1], idx[2], idx[3]])
    return np.array(V, dtype=np.float64), np.array(T)

def plot_triangulation(V, T):
    triang = mtri.Triangulation(V[:,0], V[:,1], T)

    plt.triplot(triang)
    plt.gca().set_aspect('equal')
    plt.show()

# V, T = grid_triangulation(np.array(
#     [[0,0,0,0,0,0,0],
#      [0,1,1,1,1,0,0],
#      [0,1,1,1,1,1,0],
#      [0,0,0,1,1,1,0],
#      [0,0,0,0,0,1,0],
#      [0,0,0,0,0,0,0]]
# ).T, 3)


V, T = grid_triangulation(np.array(
    [
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,0],
    ]
)[::-1].T, 10)

def triangle_area(points):
    '''
    :param points: List of 3 vertices of the triangle
    '''
    v1, v2, v3 = points
    return 0.5 * abs(v1[0]*(v2[1]-v3[1]) + v2[0]*(v3[1]-v1[1]) + v3[0]*(v1[1]-v2[1]))

def triangle_vector(points, index):
    '''
    :param points: List of 3 vertices of the triangle
    :param index: Index of the vertex to compute the vector for (0, 1, or 2)
    :return: Vector perpendicular to the edge opposite to the vertex at index
    '''

    if index == 0:
        t, o1, o2 = points
    elif index == 1:
        o1, t, o2 = points
    elif index == 2:
        o1, o2, t = points

    edge = o2 - o1
    n_edge = edge / np.linalg.norm(edge)
    o1_t = t - o1
    return (n_edge.T @ o1_t)*n_edge - o1_t

def construct_elasticity_matrix(E, nu):
    '''
    :param E: Young's modulus
    :param nu: Poisson's ratio
    :return: Elasticity matrix C
    '''
    # C = np.zeros((2,2,2,2))
    # C[0,0,0,0] = C[1,1,1,1] = E / (1 - nu**2)
    # C[0,0,1,1] = C[1,1,0,0] = E * nu / (1 - nu**2)
    # C[0,1,0,1] = C[1,0,1,0] = E / (2 * (1 + nu))
    # return C
    C = np.zeros((2,2,2,2), dtype=float)
    A = E/(1-nu**2)
    mu = E/(2*(1+nu))

    C[0,0,0,0] = A
    C[1,1,1,1] = A
    C[0,0,1,1] = A*nu
    C[1,1,0,0] = A*nu

    # all shear minor-symmetry entries:
    C[0,1,0,1] = mu
    C[0,1,1,0] = mu
    C[1,0,0,1] = mu
    C[1,0,1,0] = mu
    return C

def construct_stiffness_matrix(V, T, C, dirichlet):
    '''
    :param V: List of vertices
    :param T: List of triangles (as indices into V)
    :param C: Elasticity matrix
    :param dirichlet: List of tuples (vertex_index, coordinate_index) for Dirichlet boundary conditions
    :return: Stiffness matrix K
    '''
    K = np.zeros((len(V)*2, len(V)*2))
    for tri in T:
        area = triangle_area(V[tri])
        for shape_i in range(3):
            for test_i in range(3):
                t = triangle_vector(V[tri], test_i)
                s = triangle_vector(V[tri], shape_i)
                # for coordinate in range(2):
                for x in range(2):
                    for y in range(2):
                        contraction = np.einsum('ijlk, j, k -> il', C, t, s)
                        entry = contraction[x, y] * area / (t.T@t) / (s.T@s)

                        K[tri[test_i]*2 + x, tri[shape_i]*2 + y] += entry

    # Apply Dirichlet boundary conditions
    for vertex_index, coordinate_index in dirichlet:
        K[vertex_index*2 + coordinate_index, :] = 0
        K[:, vertex_index*2 + coordinate_index] = 0
        K[vertex_index*2 + coordinate_index, vertex_index*2 + coordinate_index] = 1

    return K

def construct_load_vector(V, T, B, dirichlet, neumann, body_force=None):
    '''
    :param V: List of vertices
    :param T: List of triangles (as indices into V) s
    :param B: List of boundaries (as indices into V)
    :param dirichlet: List of tuples (vertex_index, coordinate_index) for Dirichlet boundary conditions
    :param neumann: List of tuples (boundary_index, traction) for Neumann boundary conditions
    :param body_force: Function that takes a vertex and returns the body force vector at that vertex
    :return: Load vector F
    '''
    F = np.zeros(len(V)*2)
    
    # Implement Neumann boundary conditions
    for boundary_index, force_vector in neumann:
        v0 = V[B[boundary_index][0]]
        v1 = V[B[boundary_index][1]]
        length = np.linalg.norm(v1 - v0)
        F[B[boundary_index][0]*2] += force_vector[0] * length / 2
        F[B[boundary_index][0]*2+1] += force_vector[1] * length / 2
        F[B[boundary_index][1]*2] += force_vector[0] * length / 2
        F[B[boundary_index][1]*2+1] += force_vector[1] * length / 2

    # Implement body force contribution
    if body_force is not None:
        for tri in T:
            area = triangle_area(V[tri])
            for vertex_index in tri:
                force = body_force(V[vertex_index])
                F[vertex_index*2] += force[0] * area / 3
                F[vertex_index*2+1] += force[1] * area / 3

            
    # Fix Dirichlet loadings
    for vertex_index, coordinate_index in dirichlet:
        F[vertex_index*2 + coordinate_index] = 0
    return F


boundary_vertices = []
for i in range(V.shape[0]):
    if V[i, 0] == 0:
        j = 0
        for j in range(len(boundary_vertices)):
            if boundary_vertices[j][1][1] < V[i, 1]:
                break
        boundary_vertices.insert(j, (i, V[i]))

dirichlet = []
for i in range(len(boundary_vertices)):
    dirichlet.append((boundary_vertices[i][0], 0))
    dirichlet.append((boundary_vertices[i][0], 1))

C = construct_elasticity_matrix(2e5, 0.3)
# dirichlet = [(1, 0), (1, 1), (3, 1)]

K = construct_stiffness_matrix(V, T, C, dirichlet)
invK = np.linalg.inv(K)

# B = np.array([[63, 59], [59, 55], [55, 51], [51,47], [47,43], [43,39], [39,35], [35,31]])
# force = np.array([0, -10000])
# neumann = [(0, 1*force), (1, 0.9*force), (2, 0.8*force), (3, 0.7*force), (4, 0.6*force)]
# B = np.array([[560, 559], [559, 558], [558, 557], [557, 556], [556, 555], [555, 554], [554, 553], [553, 552], [552, 551], [551, 550]])
# force = np.array([0, 1000])
# neumann = [(0, force), (1, force), (2, force), (3, force), (4, force), (5, force), (6, force), (7, force), (8, force), (9, force)]


F = construct_load_vector(V, T, [], dirichlet, [], body_force=lambda v: np.array([0,-2]))
u = invK @ F

V += u.reshape(-1, 2)
plot_triangulation(V, T)
