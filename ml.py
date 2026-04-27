import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import fem
import helpers
import numpy as np

class SineLayer(nn.Module):
    features: int
    omega_0: float
    is_first: bool = False

    @nn.compact
    def __call__(self, x):
        def kernel_init(key, shape, dtype=jnp.float32):
            lim = jnp.sqrt(6 / shape[0]) / self.omega_0
            return jax.random.uniform(key, shape, dtype, minval=-lim, maxval=lim)

        x = nn.Dense(self.features, kernel_init=kernel_init)(x)
        return jnp.sin(self.omega_0 * x)


class Siren(nn.Module):
    hidden_dim: int = 128
    hidden_layers: int = 4
    omega_0: float = 30.0
    x0_range: tuple = (0, 1)
    x1_range: tuple = (0, 1)

    @nn.compact
    def __call__(self, x):
        # x is NOT batched
        # renormalize x0 and x1 to be in [0, 1] based on x_range and y_range:
        x0 = (x[0] - self.x0_range[0]) / (self.x0_range[1] - self.x0_range[0])
        x1 = (x[1] - self.x1_range[0]) / (self.x1_range[1] - self.x1_range[0])
        x =  jnp.array([x0, x1])
        
        x = SineLayer(self.hidden_dim, self.omega_0, is_first=True)(x)

        for _ in range(self.hidden_layers - 1):
            x = SineLayer(self.hidden_dim, self.omega_0)(x)

        # final layer: NO sine
        x = nn.Dense(1)(x)
        return x

def train(V, T, rots):
    x0_min = V[:, 0].min()
    x0_max = V[:, 0].max()
    x1_min = V[:, 1].min()
    x1_max = V[:, 1].max()

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = Siren(x0_range=(x0_min, x0_max), x1_range=(x1_min, x1_max))
    params = model.init(subkey, jnp.zeros((1, 2)))
    optimizer = optax.adam(learning_rate=2e-5)
    opt_state = optimizer.init(params)

    def loss_fn(params, x, v, d=0.05, lambda_spacing=1.0):
        def scalar_fn(params, x):
            return model.apply(params, x).squeeze()

        grad_fn = jax.grad(scalar_fn, argnums=1)
        grad_u = jax.vmap(grad_fn, in_axes=(None, 0))(params, x)
        
        # alignment
        align = (grad_u * v).sum(axis=1)**2

        # spacing
        norm = jnp.sqrt((grad_u**2).sum(axis=1) + 1e-8)
        spacing = (norm - 1.0/d)**2

        return align.mean() + lambda_spacing * spacing.mean()

    @jax.jit
    def train_step(params, opt_state, x, v):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, v)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in range(100_000):
        key, subkey = jax.random.split(key)
        sampled_points, sampled_rots = helpers.sample_mesh(V, T, rots, num_samples=3000, key=subkey)

        sampled_vectors = jax.vmap(helpers.angle_to_vector)(sampled_rots)

        params, opt_state, loss = train_step(params, opt_state, sampled_points, sampled_vectors)
        if epoch % 1_000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
            helpers.plot_model_contours_on_mesh(model.apply, params, V, T)

def generate_training_example(key=jax.random.PRNGKey(0)):
    key, subkey = jax.random.split(key)
    V, T = helpers.generate_mesh(res_multiplier_range=(0.4, 0.8), key=subkey)

    # generate random dirichlet conditions
    num_dirichlet = 3
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, V.shape[0], shape=(num_dirichlet,), replace=False)
    dirichlet = []
    for i in indices:
        dirichlet.append((i, 0))
        dirichlet.append((i, 1))

    C = fem.construct_orthotropic_elasticity(
        E1 = 3500,
        E2 = 2000,
        nu12 = 0.33,
        G12 = 800
    )

    x0_min = V[:, 0].min()
    x0_max = V[:, 0].max()
    x1_min = V[:, 1].min()
    x1_max = V[:, 1].max()

    # generate random body forces
    num_body_sources = 2
    max_component_force = 120
    key, subkey = jax.random.split(key)
    body_sources = jax.random.uniform(subkey, (num_body_sources, 2), minval=jnp.array([x0_min, x1_min]), maxval=jnp.array([x0_max, x1_max]))
    key, subkey = jax.random.split(key)
    forces = jax.random.uniform(subkey, (num_body_sources, 2), minval=-max_component_force, maxval=max_component_force)

    def body_force_fn(v):
        total_force = jnp.zeros(2)
        for source, force in zip(body_sources, forces):
            dist = jnp.linalg.norm(v - source)
            total_force += force * jnp.exp(-dist**2 / (2*5**2))
        return total_force
    
    rots = jnp.array([0 for _ in range(len(T))], dtype=jnp.float32)
    F = fem.construct_load_vector(V, T, [], dirichlet, [], body_force=body_force_fn)

    @jax.jit
    def forward(rots):
        K = fem.construct_stiffness_matrix(V, T, C, dirichlet, rots=rots)

        u = jnp.linalg.solve(K, F)
        u = u.reshape(-1, 2)
        stresses = fem.compute_stresses(u, V, T, C, rots=rots)
        vm_stresses = jax.vmap(fem.von_mises_stresses)(stresses)
        loss = helpers.softmax_max(vm_stresses, beta=10)
        return loss

    for _ in range(2):
        gradient = jax.grad(forward)(rots)
        rots = rots - 0.01 * gradient

    return V, T, rots

def generate_and_save_training_data():
    chunk = []
    chunk_size = 500
    chunk_id = 0

    for i in range(chunk_size*20):

        key = jax.random.PRNGKey(i)
        V, T, rots = generate_training_example(key=key)
        centroids, edges, _ = helpers.generate_mesh_dual(V, T)

        example = {
            "V": np.array(jax.device_get(V)),
            "T": np.array(jax.device_get(T)),
            "rots": np.array(jax.device_get(rots)),
            "centroids": np.array(jax.device_get(centroids)),
            "edges": np.array(jax.device_get(edges)),
        }

        chunk.append(example)
        if i % 10 == 0:
            print("generated", i)
        if len(chunk) == chunk_size:

            np.savez_compressed(
                f"data/chunk_{chunk_id:04d}.npz",
                examples=np.array(chunk, dtype=object)
            )

            chunk = []
            chunk_id += 1

if __name__ == "__main__":
    generate_and_save_training_data()
    #load a chunk and visualize
    # data = np.load("data/chunk_0000.npz", allow_pickle=True)
    # examples = data["examples"]
    # example = examples[2]
    # V = example["V"]
    # T = example["T"]
    # rots = example["rots"]
    # helpers.plot_triangulation_with_angle(V, T, rots)
    # edges = example["edges"]
    # C = example["centroids"]
    # helpers.plot_primal_dual(V, T, C, edges)