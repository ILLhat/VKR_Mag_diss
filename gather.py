import numpy as np
from math import floor
from timer import timeit


def wrap(ids: int, size: int) -> int:
    return ids % size


@timeit
def nearest_grid_point_gather(
    grid_efield_x: np.ndarray,
    grid_efield_y: np.ndarray,
    grid_efield_z: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
    n_particles: int,
    grid_size: tuple,
    cell_size: float,
    charge: float,
) -> np.ndarray:
    nx = grid_size[0]
    ny = grid_size[1]
    nz = grid_size[2]
    efield_x = np.zeros(n_particles)
    efield_y = np.zeros(n_particles)
    efield_z = np.zeros(n_particles)
    for n in range(n_particles):
        # Find nearest grid point for each particle
        # and add charge to the grid
        i = int(round(pos_x[n] / cell_size))
        j = int(round(pos_y[n] / cell_size))
        k = int(round(pos_z[n] / cell_size))
        efield_x += grid_efield_x[wrap(i, nx), wrap(j, ny), wrap(k, nz)]
        efield_y += grid_efield_y[wrap(i, nx), wrap(j, ny), wrap(k, nz)]
        efield_z += grid_efield_z[wrap(i, nx), wrap(j, ny), wrap(k, nz)]
    return (efield_x, efield_y, efield_z)


@timeit
def cloud_in_cell_gather(
    grid_efield: tuple,
    pos: tuple,
    n_particles: int,
    grid_size: tuple,
    cell_size: float,
    charge: float,
) -> np.ndarray:
    grid_efield_x = grid_efield[0]
    grid_efield_y = grid_efield[1]
    grid_efield_z = grid_efield[2]
    pos_x = pos[0]
    pos_y = pos[1]
    pos_z = pos[2]
    efield_x = np.zeros(n_particles)
    efield_y = np.zeros(n_particles)
    efield_z = np.zeros(n_particles)
    for n in range(n_particles):
        # Find cell coordinate
        i = floor(pos_x[n] / cell_size)
        j = floor(pos_y[n] / cell_size)
        k = floor(pos_z[n] / cell_size)
        # Find fractional distances
        d_x = pos_x[n] - i
        d_y = pos_y[n] - j
        d_z = pos_z[n] - k
        # Assign charge to grid
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    weight = (
                        (1 - d_x if di == 0 else d_x)
                        * (1 - d_y if dj == 0 else d_y)
                        * (1 - d_z if dk == 0 else d_z)
                    )
                    ii = (i + di) % grid_size[0]
                    jj = (j + dj) % grid_size[1]
                    kk = (k + dk) % grid_size[2]
                    efield_x += grid_efield_x[ii, jj, kk] * weight
                    efield_y += grid_efield_y[ii, jj, kk] * weight
                    efield_z += grid_efield_z[ii, jj, kk] * weight
    return (efield_x, efield_y, efield_z)


@timeit
def triangular_shaped_cloud_gather(
    grid_efield: tuple,
    pos: tuple,
    n_particles: int,
    grid_size: tuple,
    cell_size: float,
    charge: float,
) -> tuple:
    def weight_1d(pos):
        d = abs(pos)
        if d < 0.5:
            return 0.75 - d**2
        elif d < 1.5:
            return 0.5 * (1.5 - d) ** 2
        else:
            return 0.0

    grid_efield_x = grid_efield[0]
    grid_efield_y = grid_efield[1]
    grid_efield_z = grid_efield[2]
    pos_x = pos[0]
    pos_y = pos[1]
    pos_z = pos[2]
    efield_x = np.zeros(n_particles)
    efield_y = np.zeros(n_particles)
    efield_z = np.zeros(n_particles)
    for n in range(n_particles):
        # Find cell coordinate
        x = pos_x[n] / cell_size
        y = pos_y[n] / cell_size
        z = pos_z[n] / cell_size
        i = int(floor(x))
        j = int(floor(y))
        k = int(floor(z))
        # Assign charge to grid
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    wx = weight_1d(x - (i + di))
                    wy = weight_1d(y - (j + dj))
                    wz = weight_1d(z - (k + dk))
                    weight = wx * wy * wz
                    ii = (i + di) % grid_size[0]
                    jj = (j + dj) % grid_size[1]
                    kk = (k + dk) % grid_size[2]
                    efield_x += grid_efield_x[ii, jj, kk] * weight
                    efield_y += grid_efield_y[ii, jj, kk] * weight
                    efield_z += grid_efield_z[ii, jj, kk] * weight
    return (efield_x, efield_y, efield_z)
