import jax
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import jax.numpy as jnp

# Example data
V = jnp.array([[0,0], [1,0], [1,1], [0,1]])
T = jnp.array([[0,1,2], [0,2,3]])

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
                        T.append([idx[0], idx[1], idx[3]])
                        T.append([idx[1], idx[2], idx[3]])
    V = jnp.array(V, dtype=jnp.float64)/subdivisions
    return V, jnp.array(T, dtype=jnp.int32)

def plot_triangulation(V, T):
    import numpy
    V = numpy.asarray(V)
    T = numpy.asarray(T)
    triang = mtri.Triangulation(V[:,0], V[:,1], T)

    plt.triplot(triang)
    plt.gca().set_aspect('equal')
    plt.show()

# V, T = grid_triangulation(jnp.array(
#     [[0,0,0,0,0,0,0],
#      [0,1,1,1,1,0,0],
#      [0,1,1,1,1,1,0],
#      [0,0,0,1,1,1,0],
#      [0,0,0,0,0,1,0],
#      [0,0,0,0,0,0,0]]
# ).T, 3)


V, T = grid_triangulation(jnp.array(
    [
     [0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,1,1,1],
     [1,1,1,1,1,1,1,1,1,1,1,1],
     [1,1,1,1,1,1,1,1,1,1,1,1],
     [1,1,1,0,0,0,0,0,0,0,1,1],
     [1,1,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0],
    ]
)[::-1].T, 5)


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
    n_edge = edge / jnp.linalg.norm(edge)
    o1_t = t - o1
    return (n_edge.T @ o1_t)*n_edge - o1_t

def construct_elasticity_matrix(E, nu):
    '''
    :param E: Young's modulus
    :param nu: Poisson's ratio
    :return: Elasticity matrix C
    '''
    C = jnp.zeros((2,2,2,2), dtype=float)
    A = E/(1-nu**2)
    mu = E/(2*(1+nu))

    C = C.at[0,0,0,0].set(A)
    C = C.at[1,1,1,1].set(A)
    C = C.at[0,0,1,1].set(A*nu)
    C = C.at[1,1,0,0].set(A*nu)

    # all shear minor-symmetry entries:
    C = C.at[0,1,0,1].set(mu)
    C = C.at[0,1,1,0].set(mu)
    C = C.at[1,0,0,1].set(mu)
    C = C.at[1,0,1,0].set(mu)
    return C

def construct_orthotropic_elasticity(E1, E2, nu12, G12):
    """
    2D orthotropic elasticity tensor C[i,j,k,l] for plane stress.
    Material axes are aligned with x,y.

    Ijnputs:
      E1, E2   : Young's moduli along x(=1) and y(=2)
      nu12     : Poisson ratio (strain in 2 due to stress in 1)
      G12      : shear modulus in the 1-2 plane

    Returns:
      C : (2,2,2,2) tensor with minor symmetries, such that sigma = C : eps
          where eps is the standard symmetric strain tensor.
    """
    C = jnp.zeros((2, 2, 2, 2), dtype=float)

    nu21 = nu12 * E2 / E1
    denom = 1.0 - nu12 * nu21
    if denom <= 0:
        raise ValueError("Invalid parameters: 1 - nu12*nu21 must be > 0 for stability (plane stress).")

    # Normal stiffnesses (plane stress orthotropic)
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom  # equals nu21 * E1 / denom

    # Fill tensor entries (i,j,k,l in {0,1} correspond to {1,2})
    C = C.at[0, 0, 0, 0].set(Q11)
    C = C.at[1, 1, 1, 1].set(Q22)
    C = C.at[0, 0, 1, 1].set(Q12)
    C = C.at[1, 1, 0, 0].set(Q12)

    # Shear: sigma12 = 2*G12*eps12 because eps12=eps21 and C1212=C1221=...
    C = C.at[0, 1, 0, 1].set(G12)
    C = C.at[0, 1, 1, 0].set(G12)
    C = C.at[1, 0, 0, 1].set(G12)
    C = C.at[1, 0, 1, 0].set(G12)

    return C

def construct_stiffness_matrix(V, T, C, dirichlet):
    '''
    :param V: List of vertices
    :param T: List of triangles (as indices into V)
    :param C: Elasticity matrix
    :param dirichlet: List of tuples (vertex_index, coordinate_index) for Dirichlet boundary conditions
    :return: Stiffness matrix K
    '''
    
    def accumulate_K(K, tri):
        area = triangle_area(V[tri])
        for shape_i in range(3):
            for test_i in range(3):
                t = triangle_vector(V[tri], test_i)
                s = triangle_vector(V[tri], shape_i)
                contraction = jnp.einsum('ijlk, j, k -> il', C, t, s)
                entry = contraction * area / (t.T@t) / (s.T@s)
                x, y = tri[test_i]*2, tri[shape_i]*2
                update = jax.lax.dynamic_slice(K, (x,y), (2,2)) + entry[:2, :2]
                K = jax.lax.dynamic_update_slice(K, update, (x,y))
        return K, None
    K, _ = jax.lax.scan(accumulate_K, jnp.zeros((len(V)*2, len(V)*2)), T)

    # Apply Dirichlet boundary conditions
    for vertex_index, coordinate_index in dirichlet:
        K = K.at[vertex_index*2 + coordinate_index, :].set(0)
        K = K.at[:, vertex_index*2 + coordinate_index].set(0)
        K = K.at[vertex_index*2 + coordinate_index, vertex_index*2 + coordinate_index].set(1)
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
    F = jnp.zeros(len(V)*2)
    
    # Implement Neumann boundary conditions
    for boundary_index, force_vector in neumann:
        v0 = V[B[boundary_index][0]]
        v1 = V[B[boundary_index][1]]
        length = jnp.linalg.norm(v1 - v0)
        F[B[boundary_index][0]*2] += force_vector[0] * length / 2
        F[B[boundary_index][0]*2+1] += force_vector[1] * length / 2
        F[B[boundary_index][1]*2] += force_vector[0] * length / 2
        F[B[boundary_index][1]*2+1] += force_vector[1] * length / 2

    # Implement body force contribution
    def accumulate_F(F, tri):
         area = triangle_area(V[tri])
         for vertex_index in tri:
             force = body_force(V[vertex_index])
             F = F.at[vertex_index*2].add(force[0] * area / 3)
             F = F.at[vertex_index*2+1].add(force[1] * area / 3)
         return F, None
    
    if body_force is not None:
        F, _ = jax.lax.scan(accumulate_F, F, T)
            
    # Fix Dirichlet loadings
    for vertex_index, coordinate_index in dirichlet:
        F = F.at[vertex_index*2 + coordinate_index].set(0)
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

C = construct_orthotropic_elasticity(4.0e5, 1.5e6, 0.30, 0.8e6)

# dirichlet = [(1, 0), (1, 1), (3, 1)]

K = construct_stiffness_matrix(V, T, C, dirichlet)

F = construct_load_vector(V, T, [], dirichlet, [], body_force=lambda v: jnp.array([0,-100]))

u = jnp.linalg.solve(K, F)

V += u.reshape(-1, 2)
plot_triangulation(V, T)
