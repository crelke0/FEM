import jax
import jax.numpy as jnp

import helpers

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

def construct_stiffness_matrix(V, T, C, dirichlet, rots=None):
    '''
    :param V: List of vertices
    :param T: List of triangles (as indices into V)
    :param C: Elasticity tensor
    :param rots: (optional) List of rotation angles for each triangle (in radians)
    :param dirichlet: List of tuples (vertex_index, coordinate_index) for Dirichlet boundary conditions
    :return: Stiffness matrix K
    '''
    def accumulate_K(i, K):
        rot = rots[i] if rots is not None else 0
        tri = T[i]
        area = helpers.triangle_area(V[tri])
        for shape_i in range(3):
            for test_i in range(3):
                t = helpers.triangle_vector(V[tri], test_i)
                s = helpers.triangle_vector(V[tri], shape_i)
                rot_mat = jnp.array([[jnp.cos(-rot), -jnp.sin(-rot)], [jnp.sin(-rot), jnp.cos(-rot)]])
                t = rot_mat @ t
                s = rot_mat @ s
                contraction = jnp.einsum('ijkl, j, l -> ik', C, t, s)

                entry = contraction * area / (t.T@t) / (s.T@s)
                x, y = tri[test_i]*2, tri[shape_i]*2
                update = jax.lax.dynamic_slice(K, (x,y), (2,2)) + entry[:2, :2]
                K = jax.lax.dynamic_update_slice(K, update, (x,y))
        return K
    
    K = jax.lax.fori_loop(0, len(T), accumulate_K, jnp.zeros((len(V)*2, len(V)*2)))

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
         area = helpers.triangle_area(V[tri])
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

def compute_stresses(u, V, T, C, rots=None):
    '''
    :param u: Displacement vectors shape (len(V), 2)
    :param V: List of vertices
    :param T: List of triangles (as indices into V)
    :param C: Elasticity matrix
    :param rots: (optional) List of rotation angles for each triangle (in radians)
    :return: List of stress tensors for each triangle
    '''
    rots = rots if rots is not None else jnp.zeros(len(T))
    def compute_triange_stress(tri, rot):
        grad = jnp.zeros((2,2))
        for vertex in range(3):
            s = helpers.triangle_vector(V[tri], vertex)
            rot_mat = jnp.array([[jnp.cos(-rot), -jnp.sin(-rot)], [jnp.sin(-rot), jnp.cos(-rot)]])
            s = rot_mat @ s
            grad = grad + jnp.einsum('i, j -> ij', s, u[tri[vertex]])
        strain = 0.5 * (grad + grad.T)
        stress = jnp.einsum('ijkl, kl->ij', C, strain)
        return stress
    triangle_stresses = jax.vmap(compute_triange_stress)(T, rots)
    
    return triangle_stresses

def von_mises_stresses(stress):
    '''
    :param stress: Stress tensor shape (2,2)
    :return: Von Mises stress scalar
    '''
    sxx = stress[0, 0]
    sxy = stress[0, 1]
    syy = stress[1, 1]
    return jnp.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)
