import pygame

# Initialize Pygame
pygame.init()

# Set up the game window
window = pygame.display.set_mode((1800, 600))
pygame.display.set_caption("hi there")

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()