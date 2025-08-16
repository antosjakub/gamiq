import numpy as np

g = 1

def deg_from_rad(rad):
    return rad/(2*np.pi) * 360
def rad_from_deg(deg):
    return deg/360 * 2*np.pi


class CartPole:

    def __init__(self, pole_length, x, x_dot, theta, theta_dot):
        self.l = pole_length
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot

    def step(self, dt, action):
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

        print(self.theta, self.x)



