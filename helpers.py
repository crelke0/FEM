import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import jax.numpy as jnp
import jax

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

def plot_triangulation_with_angle(V, T, rotation):
    triang = mtri.Triangulation(V[:, 0], V[:, 1], T)

    plt.figure()
    plt.tripcolor(
        triang,
        facecolors=rotation,
        cmap="twilight",
        vmin=-jnp.pi/2,
        vmax=jnp.pi/2,
        edgecolors="k"
    )

    plt.gca().set_aspect("equal")
    plt.colorbar(label="rotation (rad)")
    plt.show()

def plot_triangulation(V, T):
    triang = mtri.Triangulation(V[:, 0], V[:, 1], T)

    plt.figure()
    plt.triplot(triang, color="black")
    plt.gca().set_aspect("equal")
    plt.show()


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


def softmax_max(x, beta):
    m = jnp.max(x)
    return m + jnp.log(jnp.sum(jnp.exp(beta * (x - m)))) / beta

def generate_mesh_dual(V, T):
    """
    :param V: List of vertices
    :param T: List of triangles (as indices into V)
    :return: centroids, adjacency_list where centroids is a list of triangle centroids and adjacency_list is a list of lists of adjacent triangle indices
    """
    centroids = jnp.mean(V[T], axis=1)
    adjacency_list = []
    for tri in T:
        mask = jnp.isin(T, tri).sum(axis=1) == 2
        adjacency_list.append(mask.nonzero()[0])
    return centroids, adjacency_list

def triangle_quality(points):
    """
    :param points: Triangle points.
    :return: Triangle quality (between 0 and 1, where 1 is an equilateral triangle and 0 is a degenerate triangle)
    """
    v0 = points[0]
    v1 = points[1]
    v2 = points[2]

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    l0 = jnp.linalg.norm(e0)
    l1 = jnp.linalg.norm(e1)
    l2 = jnp.linalg.norm(e2)

    # area via cross product
    area2 = jnp.abs(e0[0]*e2[1] - e0[1]*e2[0])
    area = 0.5 * area2

    quality = 4 * jnp.sqrt(3) * area / (l0**2 + l1**2 + l2**2 + 1e-12)

    return quality

def jittered_grid(res, scale, jitter=0.25, key=jax.random.PRNGKey(0)):
    """
    Generate a jittered grid of points.
    :param res: Tuple (nx, ny) specifying the number of points in x and y directions
    :param scale: Tuple (sx, sy) specifying the size of the grid in x and y directions
    :param jitter: Amount of jitter to apply to each point (as a fraction of the grid spacing)
    :param key: JAX random key for reproducibility
    :return: Jittered grid of points as a (nx*ny, 2) array
    """
    nx, ny = res
    sx, sy = scale

    x = jnp.linspace(0, sx, nx)
    y = jnp.linspace(0, sy, ny)

    gx, gy = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([gx, gy], axis=-1).reshape(-1, 2)

    dx = sx / (nx - 1)
    dy = sy / (ny - 1)

    key, subkey = jax.random.split(key)
    noise = jax.random.uniform(subkey, grid.shape, minval=-1.0, maxval=1.0)

    return grid + jitter * noise * jnp.array([dx, dy])

def fourier_noise(frequency_count, key=jax.random.PRNGKey(0)):
    """
    Generate a Fourier noise function with the given number of frequencies.
    :param frequency_count: Number of Fourier frequencies to use
    :param key: JAX random key for reproducibility
    :return: A function that takes an angle and returns a noise value
    """
    def noise(angle):
        amplitudes = jax.random.uniform(key, (frequency_count,), minval=0.0, maxval=1.0)
        n=1
        frequencies = jnp.arange(n, frequency_count + n)
        return jnp.sum(amplitudes * jnp.sin(frequencies * angle))
    return noise

def plot_primal_dual(V, T, C, A):
    # temporary chat gpt function
    tri = mtri.Triangulation(V[:, 0], V[:, 1], T)

    plt.figure()

    # --- primal mesh ---
    plt.triplot(tri, color="black", alpha=0.3, linewidth=1)

    # --- primal vertices ---
    plt.scatter(V[:, 0], V[:, 1], s=5, color="black", alpha=0.4)

    # --- dual nodes (centroids) ---
    plt.scatter(C[:, 0], C[:, 1], s=12, color="red")

    # --- dual edges (from adjacency list) ---
    for i, neighbors in enumerate(A):
        for j in neighbors:
            # avoid double drawing if undirected
            if j > i:
                p0 = C[i]
                p1 = C[j]
                plt.plot([p0[0], p1[0]],
                         [p0[1], p1[1]],
                         color="red", linewidth=1, alpha=0.6)

    plt.gca().set_aspect("equal")
    plt.show()

def generate_mesh(key=jax.random.PRNGKey(3)):
    """
    Generate a random mesh. 
    :param key: JAX random key for reproducibility
    :return: V, T where V is a list of vertices and T is a list of triangles (as indices into V)
    """

    W = 9
    H = 9
    key, subkey = jax.random.split(key)
    grid = jittered_grid((15, 15), (W, H), key=subkey)
    tris = jnp.array(mtri.Triangulation(grid[:, 0], grid[:, 1]).triangles, dtype=jnp.int32)

    # mask out certain grid points to create a more interesting shape
    angles = jnp.arctan2(grid[:, 1] - W/2, grid[:, 0] - H/2)
    key, subkey = jax.random.split(key)
    grid_mask = jax.vmap(fourier_noise(5, key=subkey))(angles)*W/3+W/2 > jnp.linalg.norm(grid-jnp.array([W/2, H/2]), axis=-1)
    
    tris_mask = jax.vmap(lambda T: jnp.all(grid_mask[T]))(tris) # filter out the triangles that contain masked out points
    tris_mask = tris_mask & jax.vmap(lambda T: triangle_quality(grid[T])>0.5)(tris) # filter out the trianlges that are close to degenerate

    tris = tris[tris_mask]

    # Flood fill to remove disjoint pieces and reindex vertices

    _, adjacency_list = generate_mesh_dual(grid, tris)

    vertex_indices = jnp.full(len(grid), -1, dtype=jnp.int32)
    vertex_indices_inverse = jnp.full(len(grid), -1, dtype=jnp.int32)
    final_tris_mask = jnp.zeros(len(tris), dtype=bool)
    
    next_vertex_index = 0
    open_tris = [jnp.argmax(tris_mask)]
    while len(open_tris) > 0:
        i = open_tris.pop()
        if final_tris_mask[i]:
            continue

        for v in tris[i]:
            if vertex_indices[v] == -1:
                vertex_indices = vertex_indices.at[v].set(next_vertex_index)
                vertex_indices_inverse = vertex_indices_inverse.at[next_vertex_index].set(v)
                next_vertex_index += 1
        tris = tris.at[i].set(vertex_indices[tris[i]])
        
        for adj in adjacency_list[i]:
            if not final_tris_mask[adj]:
                open_tris.append(adj)
        
        final_tris_mask = final_tris_mask.at[i].set(True)

    V = grid[vertex_indices_inverse[vertex_indices_inverse != -1]]
    T = tris[final_tris_mask]

    return V, T

V, T = generate_mesh()
centroids, adjacency_list = generate_mesh_dual(V, T)
plot_primal_dual(V, T, centroids, adjacency_list)