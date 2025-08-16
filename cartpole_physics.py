import numpy as np

g = 1

def deg_from_rad(rad):
    return rad/(2*np.pi) * 360
def rad_from_deg(deg):
    return deg/360 * 2*np.pi


class CartPole:
    """
    theta = angle of the pole
    theta_dot = angular velocity
    theta_double_dot = angular acceleration
    (theta = 0 means the pole is upright, theta_dot > 0 means it is going clockwise)
    ---
    x = position of the cart
    x_dot = speed
    ---
    action = -1/1/0
    action = -1 means the user clicked or holds left arrow
    action = 1 means the user clicked or holds right arrow
    action = 0 means the user did not do anything 
    ---
    Core ideas:
        - 'action' corresponds to the force applied to the cart
        - the equatin governing the behavior of the cart is:
                theta_double_dot = g * np.sin(theta)
        - there is a damping (friction) term in both the equation for the car's acceleration and 
          the pole's angular acceleration
    """

    def __init__(self, pole_length=300, x=0, x_dot=0, theta=0, theta_dot=0):
        self.l = pole_length
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot

    def step(self, action, dt=0.02):
        if self.theta < np.pi/2 and self.theta > -np.pi/2:
            mult_fact = -action
        else:
            mult_fact = action

        x_double_dot = 100.0*action - 0.0025*self.x_dot / dt
        self.x_dot += x_double_dot * dt
        self.x += self.x_dot * dt

        theta_double_dot = g * np.sin(self.theta) + 0.5*mult_fact*np.abs(x_double_dot)/100
        theta_double_dot -= 0.002*self.theta_dot / dt
        self.theta_dot += theta_double_dot * dt
        self.theta += self.theta_dot * dt
