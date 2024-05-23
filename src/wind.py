import numpy as np
from scipy.optimize import fsolve
from typing import Union


def wind_profile(
    z: Union[int, float, list, np.ndarray],
    z0: Union[int, float],
    u_star: Union[int, float],
    t: Union[int, float, list, np.ndarray],
    q: Union[int, float, list, np.ndarray],
    H: Union[int, float] = None,
    H_turb: Union[int, float] = None,
    displacement_height: Union[int, float] = 0,
    kappa: float = 0.4,
    alpha: float = 1 / 4,
    beta: float = 4.7,
    gamma: float = 15.0,
) -> np.ndarray:
    """
    Calculate the wind profile based on the logarithmic wind profile equation.
    Give either the sensible heat flux H (in W/m2) or the turbulent sensible heat flux (in K m/s).
    Source: https://denning.atmos.colostate.edu/ats761/BonanBookFigures/Scans/Turbulent_fluxes_14.pdf

    Parameters:
    z (Union[int, float, list, np.ndarray]): Array of heights above the ground level.
    z0 (Union[int, float]): Roughness length.
    u_star (Union[int, float]): Friction velocity.
    t (Union[int, float, list, np.ndarray]): (Potential) temperature profile (in K).
    q (Union[int, float, list, np.ndarray]): Specific humidity profile (dimensionless, in kg/kg)..
    H (Union[int, float]): Sensible heat flux (W/m^2).
    H_turb (Union[int, float]): Turbulent sensible heat flux (K m/s).
    displacement_height (Union[int, float], optional): Zero-plane displacement height (m). Defaults to 0.
    kappa (float, optional): von Kármán constant. Defaults to 0.4.
    alpha (float, optional): Parameter for stability correction in unstable conditions. Defaults to 1/4.
    beta (float, optional): Parameter for stability correction in stable conditions. Defaults to 4.7.
    gamma (float, optional): Parameter for stability correction in unstable conditions. Defaults to 15.0.

    Returns:
    np.ndarray: Wind speed at each height in z.
    """
    z = np.array(z)

    # compute displacement height:
    # displacement_height = 10 * z0 * (3 / 4)

    displacement_height = 0

    # raise error if both H and H_turb are provided
    if (H is not None) and (H_turb is not None):
        raise ValueError(
            "Only one of sensible heat flux (H) or turbulent sensible heat flux (H_turb) should be provided, not both."
        )

    # convert sensible heat flux H to turbulent sensible heat flux H_turb, if H is provided
    elif H is not None:
        H_turb = calculate_virtual_turbulent_heat_flux(H)

    # raise error if both H and H_turb are not provided
    elif H_turb is None:
        raise ValueError(
            "Either sensible heat flux (H) or turbulent sensible heat flux (H_turb) should be provided."
        )

    T_v_mean = calculate_virtual_temperature(t, q).mean()
    L = calculate_monin_obukhov_length(u_star, T_v_mean, H_turb)

    stability = determine_stability(L)

    if stability == "stable":
        stability_correction = _correction_stable(z - displacement_height, L, beta)
    elif stability == "neutral":
        stability_correction = 0
    elif stability == "unstable":
        stability_correction = _correction_unstable(
            z - displacement_height, L, alpha, gamma
        )

    uz = (u_star / kappa) * (
        np.log((z - displacement_height) / z0) - stability_correction
    )

    return uz


def calculate_virtual_temperature(
    t: Union[int, float, list, np.ndarray], q: Union[int, float, list, np.ndarray]
) -> np.ndarray:
    """
    Calculate the virtual temperature T_v.

    Parameters:
    t (Union[int, float, list, np.ndarray]): (Potential) temperature (in K).
    q (Union[int, float, list, np.ndarray]): Specific humidity (dimensionless, in kg/kg).

    Returns:
    np.ndarray: Virtual temperature (in K).
    """
    t = np.array(t)
    q = np.array(q)
    T_v = t * (1 + 0.608 * q)
    return T_v


def calculate_virtual_turbulent_heat_flux(
    H: Union[int, float], rho: float = 1.225, cp: float = 1003.5
) -> float:
    """
    Calculate the virtual turbulent heat flux from sensible heat flux H.

    Parameters:
    H (Union[int, float]): Sensible heat flux (W/m^2).
    rho (float, optional): Density of air (kg/m^3). Defaults to 1.225.
    cp (float, optional): Specific heat capacity of air at constant pressure (J/(kg*K)). Defaults to 1003.5.

    Returns:
    float: Virtual turbulent heat flux (K m/s).
    """
    return H / (rho * cp)


def calculate_monin_obukhov_length(
    u_star: Union[int, float],
    tv_mean: Union[int, float],
    H_turb: Union[int, float],
    kappa: float = 0.4,
    g: float = 9.81,
) -> float:
    """
    Calculate the Monin-Obukhov length (L).

    Parameters:
    u_star (Union[int, float]): Friction velocity (m/s).
    tv_mean (Union[int, float]): Mean (potential) virtual temperature (K).
    H_turb (Union[int, float]): Virtual turbulent heat flux (K m/s).
    kappa (float, optional): von Kármán constant (dimensionless). Defaults to 0.4.
    g (float, optional): Acceleration due to gravity (m/s^2). Defaults to 9.81.

    Returns:
    float: Monin-Obukhov length (m).
    """

    L = -(u_star**3 * tv_mean) / (kappa * g * H_turb)
    return L


def determine_stability(L: float, threshold: float = 500) -> str:
    """
    Determine the stability regime based on the Monin-Obukhov length (L).

    Parameters:
    L (float): Monin-Obukhov length (m).
    threshold (float, optional): Threshold for neutral stability. Defaults to 1e5.

    Returns:
    str: Stability regime ("stable", "neutral", or "unstable").
    """
    if L > 0 and L < threshold:
        return "stable"
    elif L < 0 and L > -threshold:
        return "unstable"
    else:
        return "neutral"


def _correction_stable(
    z: Union[int, float, np.ndarray], L: float, beta: float = 4.7
) -> Union[int, float, np.ndarray]:
    """
    Stability correction for stable conditions.

    Parameters:
    z (Union[int, float, np.ndarray]): Height above the ground level (m).
    L (float): Monin-Obukhov length (m).
    beta (float, optional): Stability correction parameter. Defaults to 4.7.

    Returns:
    Union[int, float, np.ndarray]: Stability correction term.
    """
    return -beta * z / L


def _correction_unstable(
    z: Union[int, float, np.ndarray], L: float, alpha: float = 1 / 4, gamma: float = 15
) -> Union[int, float, np.ndarray]:
    """
    Stability correction for unstable conditions.

    Parameters:
    z (Union[int, float, np.ndarray]): Height above the ground level (m).
    L (float): Monin-Obukhov length (m).
    alpha (float, optional): Stability correction parameter. Defaults to 1/4.
    gamma (float, optional): Stability correction parameter. Defaults to 15.

    Returns:
    Union[int, float, np.ndarray]: Stability correction term.
    """
    x = (1 - gamma * z / L) ** alpha
    return (
        2 * np.log((1 + x) / 2)
        + np.log((1 + x**2) / 2)
        - (2 * np.arctan(x))
        + (np.pi / 2)
    )


def solve_wind_profile(
    u_star: Union[int, float],
    wind_speed: Union[int, float],
    z: Union[int, float, list, np.ndarray],
    z0: Union[int, float],
    t: Union[int, float, list, np.ndarray],
    q: Union[int, float, list, np.ndarray],
    H: Union[int, float] = None,
    H_turb: Union[int, float] = None,
) -> np.ndarray:
    """
    Equation to solve for finding friction velocity (u_star).

    Parameters:
    u_star (Union[int, float]): Initial guess for friction velocity (m/s).
    z (Union[int, float, list, np.ndarray]): Array of heights above the ground level (m).
    z0 (Union[int, float]): Roughness length (m).
    t (Union[int, float, list, np.ndarray]): (Potential) temperature profile (in K).
    q (Union[int, float, list, np.ndarray]): Specific humidity profile (dimensionless, kg/kg).
    H (Union[int, float]): Sensible heat flux (W/m^2).
    wind_speed (Union[int, float]): Measured wind speed (m/s).

    Returns:
    np.ndarray: Difference between calculated and measured wind speed.
    """
    return wind_profile(z, z0, u_star, t, q, H, H_turb) - wind_speed


def compute_u_star(
    wind_speed: Union[int, float],
    z: Union[int, float],
    z0: Union[int, float],
    t: Union[int, float],
    q: Union[int, float],
    H: Union[int, float] = None,
    H_turb: Union[int, float] = None,
    u_star_initial: Union[int, float] = None,
) -> float:
    """
    Compute the friction velocity u* with a correction term for stability numerically.
    Give either the sensible heat flux H (in W/m2) or the turbulent sensible heat flux (in K m/s).

    Parameters:
    wind_speed (Union[int, float]): Measured wind speed (m/s).
    z (Union[int, float]): Height above ground at which the wind speed was measured (m).
    z0 (Union[int, float]): Roughness length (m).
    t (Union[int, float]): (Potential) temperature (in K).
    q (Union[int, float]): Specific humidity (dimensionless, kg/kg).
    H (Union[int, float]): Sensible heat flux (W/m^2).
    H_turb (Union[int, float]): Turbulent sensible heat flux (K m/s).
    u_star_initial (Union[int, float]): Initial guess of u*. If not provided, it is calculated assuming neutrally stable conditions.
    """
    if u_star_initial is None:
        u_star_initial = compute_u_star_stable(wind_speed, z, z0, kappa=0.4)

    u_star_solution = fsolve(
        solve_wind_profile,
        u_star_initial,
        args=(wind_speed, z, z0, t, q, H, H_turb),
    )
    return u_star_solution[0]


def compute_u_star_stable(
    wind_speed: Union[int, float],
    z: Union[int, float],
    z0: Union[int, float],
    kappa: float = 0.4,
) -> float:
    """Compute u* for neutrally stable situations without a correction term for stability analytically.

    Args:
        wind_speed (Union[int, float]): Measured wind speed (m/s).
        z (Union[int, float]): Height above ground at which the wind speed was measured (m).
        z0 (Union[int, float]): Roughness length (m).
        kappa (float, optional): von Kármán constant (dimensionless). Defaults to 0.4.

    Returns:
        float: U* for neutrally stable cases without a correction term for atmospheric stability.
    """

    # Calculate friction velocity for neutrally stable case
    u_star = (wind_speed * kappa) / np.log(z / z0)

    return u_star
