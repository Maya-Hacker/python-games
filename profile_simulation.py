#!/usr/bin/env python3
"""
Profiling script for the crater simulation
"""
import cProfile
import pstats
import io
import pygame
from craters.simulation import CraterSimulation
from craters.config import WIDTH, HEIGHT, BACKGROUND_COLOR

def run_simulation_frames(num_frames=300):
    """Run the simulation for a specific number of frames"""
    # Initialize pygame but don't create a visible window
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.HIDDEN)
    font = pygame.font.SysFont(None, 24)
    
    # Create simulation
    simulation = CraterSimulation(font=font)
    
    # Create surface to draw on (needed for complete profiling)
    surface = pygame.Surface((WIDTH, HEIGHT))
    
    # Run for specified number of frames
    for _ in range(num_frames):
        surface.fill(BACKGROUND_COLOR)
        simulation.update()
        simulation.draw(surface)
    
    pygame.quit()

# Run the profiler
profiler = cProfile.Profile()
profiler.enable()
run_simulation_frames()
profiler.disable()

# Process and display results
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)  # Print top 30 functions by cumulative time
print(s.getvalue())

# Also save detailed results to a file
with open('profile_results.txt', 'w') as f:
    ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
    ps.print_stats(50)  # Top 50 functions
    
    # Print by internal time (time spent in the function itself)
    f.write("\n\n--- SORTED BY INTERNAL TIME ---\n\n")
    ps.sort_stats('time').print_stats(50)
    
    # Print highest number of calls
    f.write("\n\n--- SORTED BY CALL COUNT ---\n\n")
    ps.sort_stats('calls').print_stats(50)

print("Detailed profile results saved to 'profile_results.txt'") 