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
                        T.append([idx[0], idx[1], idx[2]])
                        T.append([idx[0], idx[2], idx[3]])
    return np.array(V), np.array(T)

def plot_triangulation(V, T):
    triang = mtri.Triangulation(V[:,0], V[:,1], T)

    plt.triplot(triang)
    plt.gca().set_aspect('equal')
    plt.show()

V, T = grid_triangulation(np.array(
    [[0,0,0,0,0,0,0],
     [0,1,1,1,1,0,0],
     [0,1,1,1,1,1,0],
     [0,0,0,1,1,1,0],
     [0,0,0,0,0,1,0],
     [0,0,0,0,0,0,0]]
).T, 3)


# V, T = grid_triangulation(np.array(
#     [
#      [0,1,1,1],
#      [0,1,1,1],
#      [0,1,1,1],
#     ]
# ).T)   

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
    v1, v2, v3 = points
    if index == 0:
        return np.array([v2[1] - v3[1], v3[0] - v2[0]])
    elif index == 1:
        return np.array([v3[1] - v1[1], v1[0] - v3[0]])
    elif index == 2:
        return np.array([v1[1] - v2[1], v2[0] - v1[0]])

def construct_elasticity_matrix(E, nu):
    '''
    :param E: Young's modulus
    :param nu: Poisson's ratio
    :return: Elasticity matrix C
    '''
    C = np.zeros((2,2,2,2))
    C[0,0,0,0] = C[1,1,1,1] = E / (1 - nu**2)
    C[0,0,1,1] = C[1,1,0,0] = E * nu / (1 - nu**2)
    C[0,1,0,1] = C[1,0,1,0] = E / (2 * (1 + nu))
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
                for coordinate in range(2):
                    contraction = np.einsum('ijkl, j, k -> i', C, t, s)
                    entry = contraction[coordinate] * area / (t.T@t) / (s.T@s)

                    K[tri[test_i]*2 + coordinate, tri[shape_i]*2 + coordinate] += entry

    # Apply Dirichlet boundary conditions
    for vertex_index, coordinate_index in dirichlet:
        K[vertex_index*2 + coordinate_index, :] = 0
        K[:, vertex_index*2 + coordinate_index] = 0
        K[vertex_index*2 + coordinate_index, vertex_index*2 + coordinate_index] = 1

    return K

def construct_load_vector(V, T, load):
    pass


C = construct_elasticity_matrix(200e9, 0.3)
dirichlet = [(0, 0), (0, 1), (2, 0)]
K = construct_stiffness_matrix(V, T, C, dirichlet)
invK = np.linalg.inv(K)

