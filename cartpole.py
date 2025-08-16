import numpy as np

g = 1

def deg_from_rad(rad):
    return rad / (2 * np.pi) * 360
def rad_from_deg(deg):
    return deg / 360 * 2 * np.pi


class CartPole:
    def __init__(self, pole_length=300, x=0, x_dot=0, theta=0, theta_dot=0):
        self.l = pole_length
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot

    def reset(self):
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0

    def set_random_IC(self, mult_x=0, mult_x_dot=0, mult_theta=0, mult_theta_dot=0):
        r = np.random.uniform(-1,1, 4)
        self.x = mult_x * r[0]
        self.x_dot = mult_x_dot * r[1]
        self.theta = mult_theta * r[2]
        self.theta_dot = mult_theta_dot * r[3]

    def get_state(self):
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=np.float32)

    def step(self, action, dt=0.02):
        # action - one from -1, 0, 1
        if self.theta < np.pi/2 and self.theta > -np.pi/2:
            mult_fact = -action
        else:
            mult_fact = action

        x_double_dot = 100.0 * action - 0.0025 * self.x_dot / dt
        self.x_dot += x_double_dot * dt
        self.x += self.x_dot * dt

        theta_double_dot = g * np.sin(self.theta) + 0.5 * mult_fact * np.abs(x_double_dot) / 100
        theta_double_dot -= 0.002 * self.theta_dot / dt
        self.theta_dot += theta_double_dot * dt
        self.theta += self.theta_dot * dt

        # Termination conditions
        done = abs(self.theta) > np.pi/2 or abs(self.x) > 1000
        reward = 1.0 if not done else 0.0

        return self.get_state(), reward, done
