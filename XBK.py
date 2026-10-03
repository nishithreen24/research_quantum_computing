from openfermion import *


def build_tV_hamiltonian_2d(nx, ny, t=1.0, V=1.0):

    def site(x, y):
        return x * ny + y

    H = FermionOperator()


    for x in range(nx):
        for y in range(ny):
            i = site(x, y)

            for dx, dy in [(1, 0), (0, 1)]:
                xp = (x + dx) % nx
                yp = (y + dy) % ny
                j = site(xp, yp)


                H += FermionOperator(((i, 1), (j, 0)), -t)

                H += FermionOperator(((j, 1), (i, 0)), -t)




                H += FermionOperator(((i, 1), (i, 0), (j, 1), (j, 0)), V)


                H += FermionOperator(((i, 1), (i, 0)), -V / 2)


                H += FermionOperator(((j, 1), (j, 0)), -V / 2)


                H += FermionOperator((), V / 4)

    return H



nx, ny = 2, 1
t, V = 1.0, 2.0
H_LFM = build_tV_hamiltonian_2d(nx, ny, t=t, V=V)

print(f"Lattice: {nx} x {ny} = {nx*ny} sites")
print(f"Parameters: t = {t}, V = {V}")
print(f"\nHamiltonian (number of terms before simplification): {len(list(H_LFM.get_operators()))}")
print("\nHamiltonian terms:")
print(H_LFM)

H_qubit = bravyi_kitaev(H_LFM)
print("\nHamiltonian terms after BK:")
print(H_qubit)