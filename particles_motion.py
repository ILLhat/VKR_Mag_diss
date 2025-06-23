from timer import timeit


@timeit
def boris_push(
    pos: tuple,
    vel: tuple,
    efield: tuple,
    n_particles: int,
    c_over_m: float,
    dt: float,
) -> tuple:
    for p in range(n_particles):
        vel[0][p] += c_over_m * efield[0][p] * dt
        vel[1][p] += c_over_m * efield[1][p] * dt
        vel[2][p] += c_over_m * efield[2][p] * dt
        pos[0][p] += vel[0][p] * dt
        pos[1][p] += vel[1][p] * dt
        pos[2][p] += vel[2][p] * dt
        pos[0][p] = pos[0][p] % 1
        pos[0][p] = pos[0][p] % 1
        pos[0][p] = pos[0][p] % 1
    return (pos[0], pos[1], pos[2]), (vel[0], vel[1], vel[2])
