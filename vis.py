import pygame
from pygame.math import Vector2
import numpy as np

def deg_from_rad(rad):
    return rad/(2*np.pi) * 360
def rad_from_deg(deg):
    return deg/360 * 2*np.pi

def get_circle_pos(theta):
    return np.array([np.sin(theta), np.cos(theta)])

def np_to_pygame_arr(arr):
    assert len(arr) == 2
    return Vector2(arr[0], arr[1])

# Initialize Pygame
pygame.init()
WHITE = (255,255,255)
NIGGA = (0,0,0)
RED = (255,0,0)

SCREEN_W = 1800
SCREEN_H = 600

# Set up the game window
window = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Cartpole")
clock = pygame.time.Clock()


ground_h_offset = 150

cartpole_w = 200
cartpole_h = 20

cartpole_arm_w = 300
cartpole_arm_h = 15

# initial conditions
theta = rad_from_deg(10)
theta_dot = 0
x = 0

# constants
l = cartpole_arm_w
g = 1

def h_from_theta(theta):
    return l*np.cos(theta)
def v_from_E_and_theta(E, theta):
    return np.sqrt(2*(E-g*l*np.cos(theta)))

i = 0
# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                x += 5
                theta -= rad_from_deg(10)
            if event.key == pygame.K_LEFT:
                x -= 5
                theta += rad_from_deg(10)
    dt = clock.tick(100)
    
    delta_t = dt/500
    theta_double_dot = g * np.sin(theta)
    theta_dot += theta_double_dot * delta_t
    theta += theta_dot * delta_t
    
    # GRAPHICS
    pygame.display.flip()
    window.fill(NIGGA)
    pygame.draw.line(window, WHITE, Vector2(0,SCREEN_H//2+ground_h_offset), Vector2(SCREEN_W,SCREEN_H//2+ground_h_offset), 1)
    # -- cartpole base
    pygame.draw.rect(window, WHITE, [
        SCREEN_W//2 - cartpole_w//2 + x,
        SCREEN_H//2 - cartpole_h//2 + ground_h_offset,
        cartpole_w,
        cartpole_h
    ], 0)
    # -- cartpole arm
    arm_pnts = [
        Vector2(cartpole_arm_w*i, cartpole_arm_h*j)
        for i,j in [[1,1/2], [1,-1/2], [0,-1/2], [0,1/2]]
    ] 
    arm_coords = [
        Vector2(SCREEN_W//2 + x, SCREEN_H//2 + ground_h_offset) +
        p.rotate(-90+deg_from_rad(theta)) for p in arm_pnts
    ]
    pygame.draw.polygon(window, WHITE, arm_coords, 0)
    # -- cartpole center
    pygame.draw.circle(window, NIGGA, [
        SCREEN_W//2 + x,
        SCREEN_H//2 + ground_h_offset
    ], 5, 0)

    pygame.display.update()

    i+=1
    #if i > 100:
    #    break

# Quit Pygame
pygame.quit()


