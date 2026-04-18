import jax
import jax.numpy as jnp

import helpers
import fem

# LENGTH: mm
# PRESSURE: MPa = N/mm^2

V, T = helpers.grid_triangulation(jnp.array(
    [
     [0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,1,1,1,1],
     [1,1,1,1,1,1,1,1,1,1,1,1],
     [0,0,1,1,0,0,0,0,1,1,0,0],
     [0,0,1,1,0,0,0,0,1,1,0,0],
     [0,0,1,1,0,0,0,0,1,1,0,0],
    ]
)[::-1].T, 3)
V = V*10

boundary_vertices = []
for i in range(V.shape[0]):
    if V[i, 1] == 0:
        j = 0
        for j in range(len(boundary_vertices)):
            if boundary_vertices[j][1][1] < V[i, 0]:
                break
        boundary_vertices.insert(j, (i, V[i]))

dirichlet = []
for i in range(len(boundary_vertices)):
    dirichlet.append((boundary_vertices[i][0], 0))
    dirichlet.append((boundary_vertices[i][0], 1))
print("Dirichlet vertices: ", [V[i] for i, _ in dirichlet])


C = fem.construct_orthotropic_elasticity(
    E1 = 3500,
    E2 = 2000,
    nu12 = 0.33,
    G12 = 800
)

rots = jnp.array([0 for _ in range(len(T))], dtype=jnp.float64)

F = fem.construct_load_vector(V, T, [], dirichlet, [], body_force=lambda v: jnp.array([0,-10*jnp.exp(-(v[0]-6*10)**2)]))


@jax.jit
def forward(rots):
    K = fem.construct_stiffness_matrix(V, T, C, dirichlet, rots=rots)

    u = jnp.linalg.solve(K, F)
    u = u.reshape(-1, 2)
    stresses = fem.compute_stresses(u, V, T, C, rots=rots)
    vm_stresses = jax.vmap(fem.von_mises_stresses)(stresses)
    loss = helpers.softmax_max(vm_stresses, beta=10)
    return loss

for _ in range(50):
    gradient = jax.grad(forward)(rots)
    rots = rots - 0.001 * gradient

K = fem.construct_stiffness_matrix(V, T, C, dirichlet, rots=rots)

u = jnp.linalg.solve(K, F)
u = u.reshape(-1, 2)

V += u
helpers.plot_triangulation(V, T, rots)