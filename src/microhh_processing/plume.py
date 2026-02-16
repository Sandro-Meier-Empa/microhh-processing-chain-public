import numpy as np

__all__ = [
    "calculate_plume_centerline",
    "calculate_absolute_dispersion",
    "calculate_meandering_dispersion",
]


def calculate_plume_centerline(grid_y, concentration_2d):
    """
    Calculates the crosswind centerline position y_m for a given x-slice.
    Equation (2) in Raznevic et al., 2022, adapted for 2D.
    """
    # Numerator: sum(c * y), Denominator: sum(c)
    numerator = np.sum(concentration_2d * grid_y, axis=1)
    denominator = np.sum(concentration_2d, axis=1)
    
    # Avoid division by zero for slices with no plume
    return np.divide(numerator, denominator, 
                     out=np.zeros_like(numerator), 
                     where=denominator!=0)

def calculate_absolute_dispersion(grid_y, concentration_2d, y_mean_centerline):
    """
    Calculates absolute plume dispersion (sigma_ya^2).
    Equation (4) in Raznevic et al., 2022, adapted for 2D.
    """
    dy_squared = (grid_y - y_mean_centerline[:, np.newaxis])**2
    
    numerator = np.sum(concentration_2d * dy_squared, axis=1)
    denominator = np.sum(concentration_2d, axis=1)
    
    return np.divide(numerator, denominator, 
                     out=np.zeros_like(numerator), 
                     where=denominator!=0)


def calculate_meandering_dispersion(y_mean_centerline):
    """
    Calculates meandering plume dispersion (sigma_ym^2).
    Equation (5) in Raznevic et al., 2022.
    """
    return np.var(y_mean_centerline)

