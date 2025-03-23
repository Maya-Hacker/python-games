import cProfile
import pstats
import pygame
import sys
from craters.config import WIDTH, HEIGHT, BACKGROUND_COLOR, FPS
from craters.simulation import CraterSimulation

def profile_simulation(frames=300):
    """Run the simulation for a fixed number of frames with profiling enabled"""
    # Initialize Pygame
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Crater Simulation Profiling")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 18)
    
    # Create the simulation
    simulation = CraterSimulation(font=font)
    
    # Always show sensors to profile their impact
    simulation.show_sensors = True
    
    # Run for a fixed number of frames
    for _ in range(frames):
        # Clear the screen
        screen.fill(BACKGROUND_COLOR)
        
        # Update and draw simulation
        simulation.update()
        simulation.draw(screen)
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    # Run the profiler
    print("Running profiler for crater simulation...")
    cProfile.run('profile_simulation()', 'profiler_stats')
    
    # Print sorted stats
    p = pstats.Stats('profiler_stats')
    
    print("\n--- Top 20 Time Consuming Functions ---")
    p.strip_dirs().sort_stats('cumulative').print_stats(20)
    
    print("\n--- Top 20 Most Called Functions ---")
    p.strip_dirs().sort_stats('calls').print_stats(20)
    
    sys.exit() 