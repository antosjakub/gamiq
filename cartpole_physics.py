import numpy as np

g = 1

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

    def reset(self):
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0

    def set_random_IC(self, max_x=40, max_x_dot=10, max_theta=np.pi/90, max_theta_dot=np.pi/90):
        r = np.random.uniform(-1,1, 4)
        self.x = max_x * r[0]
        self.x_dot = max_x_dot * r[1]
        self.theta = max_theta * r[2]
        self.theta_dot = max_theta_dot * r[3]

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

    def step_wrapped(self, action, dt=0.02):
        self.step(action, dt)
        # Termination conditions
        done = abs(self.theta) > np.pi/2 or abs(self.x) > 1000
        reward = 1.0 if not done else 0.0
        return self.get_state(), reward, done