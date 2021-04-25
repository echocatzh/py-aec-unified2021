# shmzhang@aslp, 2021-04
import numpy as np
import cmath

# solve the generalized eigenvalue decomposition problem for 2x2
# Hermitian matrices for all frequency bins: C1*w = lambda*C0*w,
# the theoretical better solution
# diagload:             the diagonal loading to prevent singular
# C1:                   the matrix C1, 2 x 2
# C0:                   the matrix C0, 2 x 2
# Demix                 the demixing matrix, rows are the eigenvectors
#                       the scaling problem is solved


def heig2(diagload, C1, C0):
    Demix = np.zeros_like(C1)
    d = np.real(C1[0, 0]) + diagload
    er = np.real(C1[0, 1])
    ei = np.imag(C1[0, 1])
    f = np.real(C1[1, 1]) + diagload

    a = np.real(C0[0, 0]) + diagload
    br = np.real(C0[0, 1])
    bi = np.imag(C0[0, 1])
    c = np.real(C0[1, 1]) + diagload
    # calculate eigenvalues
    x = a * c - (br * br + bi * bi)
    y = 2.0 * (br * er + bi * ei) - d * c - a * f
    z = d * f - (er * er + ei * ei)
    delta = cmath.sqrt(y * y - 4.0 * x * z)
    lambda0 = (-y + delta) / (2.0 * x)
    lambda1 = (-y - delta) / (2.0 * x)
    # calculate the eigenvectors, as the row vectors of B
    b00 = f - lambda0 * c
    b01 = (-er + lambda0 * br) + (-ei + lambda0 * bi) * 1j
    b10 = (-er + lambda1 * br) + (ei - lambda1 * bi) * 1j
    b11 = d - lambda1 * a
    # solve the scaling ambiguity
    # calculate A = B^-1
    detbr = (np.real(b00) * np.real(b11) - np.imag(b00) * np.imag(b11)) - \
        (np.real(b01) * np.real(b10) - np.imag(b01) * np.imag(b10))
    detbi = (np.real(b00) * np.imag(b11) + np.imag(b00) * np.real(b11)) - \
        (np.real(b01) * np.imag(b10) + np.imag(b01) * np.real(b10))

    detbsq = detbr * detbr + detbi * detbi
    detb = (detbr / detbsq) - (detbi / detbsq) * 1j

    a00 = detb * b11
    a11 = detb * b00
    a01 = -detb * b01
    a10 = -detb * b10
    if np.abs(a00) >= np.abs(a10):
        Demix[0, 0] = a00 * b00
        Demix[0, 1] = a00 * b01
    else:
        Demix[0, 0] = a10 * b00
        Demix[0, 1] = a10 * b01
    if np.abs(a11) >= np.abs(a01):
        Demix[1, 0] = a11 * b10
        Demix[1, 1] = a11 * b11
    else:
        Demix[1, 0] = a01 * b10
        Demix[1, 1] = a01 * b11
    return Demix
