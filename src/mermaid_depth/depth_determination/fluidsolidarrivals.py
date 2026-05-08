import numpy as np
import matplotlib.pyplot as plt
"""Script Adapted from Ivy's original fluidsolidarrivals.mat"""

def fluidsolidcoefficients(rho_f, vp_f, rho_s, vp_s, vs_s, theta):
    """
    Calculate reflection/transmission coefficients for an acoustic wave in a fluid
    incident on a fluid-solid interface.

    Parameters
    ----------
    rho_f : float
        Density of the fluid.
    vp_f : float
        P-wave velocity in the fluid.
    rho_s : float
        Density of the solid.
    vp_s : float
        P-wave velocity in the solid.
    vs_s : float
        S-wave velocity in the solid.
    theta : float
        Incident angle in radians.

    Returns
    -------
    E : ndarray, shape (3,)
        Energy flux coefficients:
        [reflected, transmitted_P, transmitted_SV]
    D : ndarray, shape (3,)
        Displacement amplitude coefficients:
        [reflected, transmitted_P, transmitted_SV]
    P : ndarray, shape (3,)
        Potential amplitude coefficients:
        [reflected, transmitted_P, transmitted_SV]
    """

    theta_i = theta

    # Use complex dtype so critical-angle cases are handled like MATLAB.
    theta_t = np.arcsin(np.asarray(vp_s / vp_f * np.sin(theta_i), dtype=np.complex128))
    theta_s = np.arcsin(np.asarray(vs_s / vp_f * np.sin(theta_i), dtype=np.complex128))

    # Intermediate variables
    a = np.cos(theta_i) / vp_f
    b = np.cos(theta_t) / vp_s
    c = np.sin(theta_s) / vs_s
    d = rho_f
    f = rho_s * np.cos(2.0 * theta_s)
    g = rho_s * np.sin(2.0 * theta_s)
    m = np.sin(2.0 * theta_t) / (vp_s ** 2)
    n = np.cos(2.0 * theta_s) / (vs_s ** 2)
    p = a * (f * n + g * m)
    q = d * (b * n + c * m)

    # Potential amplitude ratios
    r_potential_amp_ratio = (p - q) / (p + q)
    tp_potential_amp_ratio = 2.0 * a * d * n / (p + q)
    tsv_potential_amp_ratio = -2.0 * a * d * m / (p + q)

    P = np.array(
        [
            r_potential_amp_ratio,
            tp_potential_amp_ratio,
            tsv_potential_amp_ratio,
        ],
        dtype=np.complex128,
    )

    # MATLAB isreal(...) behavior
    theta_t_is_real = np.isreal(theta_t).astype(float)
    theta_s_is_real = np.isreal(theta_s).astype(float)

    # Displacement amplitude ratios
    r_amp_ratio = r_potential_amp_ratio
    tp_amp_ratio = tp_potential_amp_ratio * (vp_f / vp_s) * theta_t_is_real
    tsv_amp_ratio = tsv_potential_amp_ratio * (vp_f / vs_s) * theta_s_is_real

    D = np.array(
        [
            r_amp_ratio,
            tp_amp_ratio,
            tsv_amp_ratio,
        ],
        dtype=np.complex128,
    )

    # Energy coefficients
    r = np.abs(r_amp_ratio) ** 2
    tp = (
        np.abs(tp_amp_ratio) ** 2
        * np.real(rho_s * vp_s * np.cos(theta_t))
        / (rho_f * vp_f * np.cos(theta_i))
    )
    tsv = (
        np.abs(tsv_amp_ratio) ** 2
        * np.real(rho_s * vs_s * np.cos(theta_s))
        / (rho_f * vp_f * np.cos(theta_i))
    )

    E = np.array([r, tp, tsv], dtype=float)

    return E, D, P

def fluidsolidcoefficients2(rho_f, vp_f, rho_s, vp_s, vs_s, theta):
    """
    Calculate reflection/transmission coefficients for a P wave in the solid
    incident on a fluid-solid interface from below.

    Parameters
    ----------
    rho_f : float
        Fluid density.
    vp_f : float
        Fluid P-wave velocity.
    rho_s : float
        Solid density.
    vp_s : float
        Solid P-wave velocity.
    vs_s : float
        Solid S-wave velocity.
    theta : float
        Incident angle in radians, measured for the incoming solid P wave.

    Returns
    -------
    E : ndarray, shape (3,)
        Energy coefficients:
        [reflected_P, reflected_SV, transmitted_acoustic]
    D : ndarray, shape (3,)
        Displacement amplitude coefficients:
        [reflected_P, reflected_SV, transmitted_acoustic]
    P : ndarray, shape (3,)
        Potential amplitude coefficients:
        [reflected_P, reflected_SV, transmitted_acoustic]
    """

    # Ray parameter
    p = np.sin(theta) / vp_s

    # Use complex dtype so critical-angle cases behave like MATLAB
    a = np.sqrt(np.asarray(1.0 / vp_s**2 - p**2, dtype=np.complex128))
    b = p
    c = np.sqrt(np.asarray(1.0 / vp_f**2 - p**2, dtype=np.complex128))

    # Intermediate variables
    d = rho_s * (1.0 - 2.0 * vs_s**2 * p**2)
    e = 2.0 * rho_s * vs_s**2 * p * np.sqrt(
        np.asarray(1.0 / vs_s**2 - p**2, dtype=np.complex128)
    )
    f = rho_f
    g = 2.0 * p * a
    h = (1.0 / vs_s**2) - (2.0 * p**2)

    U = (a * f * h) + (b * f * g) + (c * e * g)
    V = c * d * h

    # Potential amplitude ratios
    rp_potential_amp_ratio = (U - V) / (U + V)
    rsv_potential_amp_ratio = (2.0 * c * g * d) / (U + V)
    t_potential_amp_ratio = 2.0 * d * (a * h + b * g) / (U + V)

    P = np.array(
        [
            rp_potential_amp_ratio,
            rsv_potential_amp_ratio,
            t_potential_amp_ratio,
        ],
        dtype=np.complex128,
    )

    # MATLAB isreal(...) behavior
    a_is_real = np.isreal(a).astype(float)
    c_is_real = np.isreal(c).astype(float)

    # Displacement amplitude ratios
    rp_amp_ratio = rp_potential_amp_ratio
    rsv_amp_ratio = rsv_potential_amp_ratio * (vp_s / vs_s) * a_is_real
    t_amp_ratio = t_potential_amp_ratio * (vp_s / vp_f) * c_is_real

    D = np.array(
        [
            rp_amp_ratio,
            rsv_amp_ratio,
            t_amp_ratio,
        ],
        dtype=np.complex128,
    )

    # Energy coefficients
    rp = np.abs(rp_potential_amp_ratio) ** 2
    rsv = (
        np.abs(rsv_potential_amp_ratio) ** 2
        * (vp_s / vs_s)
        * np.real(np.sqrt((1.0 - vs_s**2 * p**2) / (1.0 - vp_s**2 * p**2)))
    )
    t = (
        np.abs(t_potential_amp_ratio) ** 2
        * (rho_f * vp_s) / (rho_s * vp_f)
        * np.real(np.sqrt((1.0 - vp_f**2 * p**2) / (1.0 - vp_s**2 * p**2)))
    )

    E = np.array([rp, rsv, t], dtype=float)

    return E, D, P

def fluidsolidarrivals(
    z_interface,
    z_station=None,
    rho_f=None,
    vp_f=None,
    rho_s=None,
    vp_s=None,
    vs_s=None,
    theta=None,
    t_max=None,
):
    """
    Compute arrival times and displacement amplitudes for waves arriving at a
    station in a fluid layer over a solid half-space, with a P wave entering
    from below.

    Parameters
    ----------
    z_interface : float or str
        Depth of the fluid-solid interface (top of fluid layer is 0), or 'demo'.
    z_station : float
        Depth of the station.
    rho_f : float
        Density of the fluid.
    vp_f : float
        P-wave velocity in the fluid.
    rho_s : float
        Density of the solid.
    vp_s : float
        P-wave velocity in the solid.
    vs_s : float
        S-wave velocity in the solid.
    theta : float
        Incidence angle of the P-wave entering from the bottom (radians).
    t_max : float
        Maximum time to include.

    Returns
    -------
    t : ndarray
        Arrival times.
    x : ndarray
        Horizontal displacement amplitudes.
    z : ndarray
        Vertical displacement amplitudes.
    ph : ndarray
        Phase labels:
            1  = P/upgoing acoustic
           -1  = P/downgoing acoustic
            2  = SV upgoing
           -2  = SV downgoing
    """

    # ------------------------------------------------------------------
    # Angles of incidence for each wave
    # ------------------------------------------------------------------
    theta_s = np.arcsin(vs_s / vp_s * np.sin(theta))
    theta_f = np.arcsin(vp_f / vp_s * np.sin(theta))

    # ------------------------------------------------------------------
    # Displacement amplitude coefficients
    # ------------------------------------------------------------------
    # solid -> fluid
    _, D_S2F, _ = fluidsolidcoefficients2(rho_f, vp_f, rho_s, vp_s, vs_s, theta)
    D_S2F_RP = D_S2F[0]   # reflected P
    D_S2F_RSV = D_S2F[1]  # reflected SV
    D_S2F_T = D_S2F[2]    # transmitted acoustic

    # free-surface reflection in acoustic medium
    D_FREE_R = -1.0

    # fluid -> solid
    _, D_F2S, _ = fluidsolidcoefficients(rho_f, vp_f, rho_s, vp_s, vs_s, theta_f)
    D_F2S_R = D_F2S[0]    # reflected acoustic
    D_F2S_TP = D_F2S[1]   # transmitted P
    D_F2S_TSV = D_F2S[2]  # transmitted SV

    # ------------------------------------------------------------------
    # Sign to convert downgoing wave to displacement
    # ------------------------------------------------------------------
    SGN_P_X = 1.0
    SGN_P_Z = -1.0
    SGN_SV_X = 1.0
    SGN_SV_Z = 1.0

    # ------------------------------------------------------------------
    # Determine which layer the station is in
    # ------------------------------------------------------------------
    layer = "fluid" if (z_interface > z_station) else "solid"

    t = np.array([0.0], dtype=float)
    x = np.array([1.0], dtype=float)
    z = np.array([1.0], dtype=float)
    ph = np.array([1], dtype=int)

    if layer == "solid":
        # travel time of first downgoing P-wave
        tPP = 2.0 * (z_station - z_interface) * np.cos(theta) / vp_s

        # travel time of first downgoing SV-wave
        tPSV = (z_station - z_interface) * (
            np.cos(theta) / vp_s + np.cos(theta_s) / vs_s
        )

        # round-trip travel time of acoustic wave in the fluid layer
        tFF = 2.0 * z_interface * np.cos(theta_f) / vp_f

        # number of rounds
        nP = int(np.floor((t_max - tPP) / tFF))
        nSV = int(np.floor((t_max - tPSV) / tFF))

        # --------------------------------------------------------------
        # downgoing P-wave arrivals
        # [P; P^bathP; P(A^A)P]
        # --------------------------------------------------------------
        if nP >= 0:
            t_P = tPP + np.arange(nP + 1) * tFF

            if nP == 0:
                amp_P = np.array([D_S2F_RP], dtype=float)
            else:
                tail = (
                    D_S2F_T
                    * (D_FREE_R ** np.arange(1, nP + 1))
                    * (D_F2S_R ** np.arange(0, nP))
                    * D_F2S_TP
                )
                amp_P = np.concatenate(([D_S2F_RP], tail))

            t = np.concatenate((t, t_P))
            x = np.concatenate((x, amp_P * SGN_P_X))
            z = np.concatenate((z, amp_P * SGN_P_Z))
            ph = np.concatenate((ph, -np.ones(nP + 1, dtype=int)))

        # --------------------------------------------------------------
        # downgoing SV-wave arrivals
        # [P; P^bathS; P(A^A)S]
        # --------------------------------------------------------------
        if nSV >= 0:
            t_SV = tPSV + np.arange(nSV + 1) * tFF

            if nSV == 0:
                amp_SV = np.array([D_S2F_RSV], dtype=float)
            else:
                tail = (
                    D_S2F_T
                    * (D_FREE_R ** np.arange(1, nSV + 1))
                    * (D_F2S_R ** np.arange(0, nSV))
                    * D_F2S_TSV
                )
                amp_SV = np.concatenate(([D_S2F_RSV], tail))

            t = np.concatenate((t, t_SV))
            x = np.concatenate(
                (x, amp_SV * SGN_SV_X * np.cos(theta_s) / np.sin(theta))
            )
            z = np.concatenate(
                (z, amp_SV * SGN_SV_Z * np.sin(theta_s) / np.cos(theta))
            )
            ph = np.concatenate((ph, -2 * np.ones(nSV + 1, dtype=int)))

    else:
        # travel time of the first downgoing acoustic wave
        tAA = 2.0 * z_station * np.cos(theta_f) / vp_f

        # round-trip travel time of acoustic wave in the fluid layer
        tFF = 2.0 * z_interface * np.cos(theta_f) / vp_f

        # number of rounds
        n_up = int(np.floor(t_max / tFF))
        n_down = int(np.floor((t_max - tAA) / tFF))

        # upgoing acoustic wave arrivals
        if n_up >= 1:
            t_up = np.arange(1, n_up + 1) * tFF
            amp_up = (D_FREE_R * D_F2S_R) ** np.arange(1, n_up + 1)

            t = np.concatenate((t, t_up))
            x = np.concatenate((x, amp_up))
            z = np.concatenate((z, amp_up))
            ph = np.concatenate((ph, np.ones(n_up, dtype=int)))

        # downgoing acoustic wave arrivals
        if n_down >= 0:
            t_down = tAA + np.arange(n_down + 1) * tFF
            amp_down = D_FREE_R * (D_F2S_R * D_FREE_R) ** np.arange(0, n_down + 1)

            t = np.concatenate((t, t_down))
            x = np.concatenate((x, amp_down * SGN_P_X))
            z = np.concatenate((z, amp_down * SGN_P_Z))
            ph = np.concatenate((ph, -np.ones(n_down + 1, dtype=int)))

    # ------------------------------------------------------------------
    # Sort arrivals by time
    # ------------------------------------------------------------------
    it = np.argsort(t)
    t = t[it]
    x = x[it]
    z = z[it]
    ph = ph[it]

    # vertical incidence has no horizontal displacement
    if abs(theta) == 0:
        x = np.zeros_like(t)

    return t, x, z, ph

if __name__ == "__main__":
    z_interface = 4000.0
    z_station = 1000.0
    rho_f = 1020.0
    vp_f = 1500.0
    rho_s = 2500.0
    vp_s = 3400.0
    vs_s = 1963.0
    theta = 20.0 * np.pi / 180.0
    t_max = 100.0

    t, x, z, ph = fluidsolidarrivals(
        z_interface, z_station, rho_f, vp_f, rho_s, vp_s, vs_s, theta, t_max
    )

    print(t, x, z, ph)

    theta_f = np.arcsin(vp_f / vp_s * np.sin(theta))

    fig, ax = plt.subplots(figsize=(8, 4))

    p_mask = np.abs(ph) == 1

    # For a fluid receiver, plot scalar acoustic pressure amplitude.
    # The direct arrival is normalized to 1.
    pressure_amp = np.real(x[p_mask])

    ax.stem(
        t[p_mask],
        pressure_amp,
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" ",
    )

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlim([-2, 52])
    ax.set_ylim(1.1 * np.min(pressure_amp), 1.1 * np.max(pressure_amp))
    ax.grid(True)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("relative pressure amplitude")
    ax.set_title("Fluid acoustic reverberations: relative pressure amplitudes")

    plt.show()