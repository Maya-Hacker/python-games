"""
Main simulation logic for crater simulation
"""
import random
import time
import pygame
from craters.config import (
    NUM_CRATERS, NUM_FOOD_PELLETS, TEXT_COLOR,
    FOOD_SPAWN_INTERVAL, ORANGE_FOOD_COLOR,
    AGE_TEEN, AGE_YOUNG, AGE_ADULT, AGE_MIDDLE, AGE_MATURE, AGE_SENIOR,
    DISTANCE_CUTOFF, USE_SPATIAL_HASH,
    BATCH_PROCESSING, SKIP_FRAMES_WHEN_LAGGING,
    FPS, MATING_DURATION, CELL_SIZE,
    FOOD_DETECTION_RANGE
)
from craters.models.crater import Crater
from craters.models.food import Food
from craters.spatial_hash import SpatialHash

class CraterSimulation:
    """
    Main simulation class that manages all entities and updates
    """
    def __init__(self, num_craters=NUM_CRATERS, num_food=NUM_FOOD_PELLETS, font=None):
        """
        Initialize the simulation with craters and food
        
        Args:
            num_craters (int): Number of craters to create
            num_food (int): Number of food pellets to create
            font: Pygame font for text display
        """
        self.font = font
        self.craters = [Crater(font=font) for _ in range(num_craters)]
        self.food_pellets = [Food() for _ in range(num_food)]
        self.show_sensors = True  # Toggle for sensor visualization
        self.food_spawn_timer = 0
        self.food_spawn_interval = FOOD_SPAWN_INTERVAL
        
        # Track statistics
        self.generation = 1
        self.mating_events = 0
        self.births = 0
        
        # Performance tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        self.avg_frame_time = 0
        self.skip_frame = False
        
        # Spatial hash for efficient collision detection
        if USE_SPATIAL_HASH:
            self.spatial_hash = SpatialHash(cell_size=CELL_SIZE)
    
    def _update_spatial_hash(self):
        """Update the spatial hash with current entity positions"""
        if not USE_SPATIAL_HASH:
            return
            
        self.spatial_hash.clear()
        for crater in self.craters:
            self.spatial_hash.insert(crater)
        
        for food in self.food_pellets:
            if food.active:
                self.spatial_hash.insert(food)
    
    def get_nearby_craters(self, crater):
        """Get craters near the given crater efficiently"""
        if USE_SPATIAL_HASH:
            nearby = self.spatial_hash.get_nearby_entities(crater, DISTANCE_CUTOFF)
            # Ensure we only return Crater instances and not the crater itself
            return [c for c in nearby if isinstance(c, Crater) and c is not crater]
        else:
            # Traditional O(n) approach
            return [c for c in self.craters if c is not crater]
            
    def get_nearby_food(self, crater):
        """Get active food pellets near the given crater efficiently"""
        if USE_SPATIAL_HASH:
            # Use the longer FOOD_DETECTION_RANGE for food detection
            nearby = self.spatial_hash.get_nearby_entities(crater, FOOD_DETECTION_RANGE)
            return [f for f in nearby if isinstance(f, Food) and f.active]
        else:
            # Traditional approach
            return [f for f in self.food_pellets if f.active]
    
    def update(self):
        """
        Update all entities in the simulation for one frame
        """
        # Performance tracking
        current_time = time.time()
        frame_duration = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Keep track of last 60 frames
        self.frame_times.append(frame_duration)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        # Calculate average frame time
        self.avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        # Skip updating if frame time is too high and skipping is enabled
        if SKIP_FRAMES_WHEN_LAGGING:
            target_frame_time = 1.0 / FPS
            self.skip_frame = not self.skip_frame and self.avg_frame_time > target_frame_time * 1.5
            if self.skip_frame:
                return
        
        # Update spatial hash if used
        if USE_SPATIAL_HASH:
            self._update_spatial_hash()
        
        # Update craters and handle energy depletion
        craters_to_remove = []
        mating_pairs = []  # Track craters that are mating this frame
        new_craters = []  # Track new craters born this frame
        
        # First pass: update all craters and check for mating
        if BATCH_PROCESSING:
            # Process craters in batches for better locality
            batch_size = max(1, min(20, len(self.craters)))  # Ensure batch size is at least 1
            for i in range(0, len(self.craters), batch_size):
                batch = self.craters[i:i+batch_size]
                self._update_crater_batch(batch, craters_to_remove, mating_pairs, new_craters)
        else:
            # Update each crater individually
            for i, crater in enumerate(self.craters):
                self._update_single_crater(i, crater, craters_to_remove, mating_pairs, new_craters)
        
        # Handle mating pairs
        for i in range(0, len(mating_pairs), 2):
            if i+1 < len(mating_pairs):  # Ensure we have a pair
                parent1 = mating_pairs[i]
                parent2 = mating_pairs[i+1]
                
                # Calculate total energy to distribute to offspring
                total_offspring_energy = (parent1.energy / 2) + (parent2.energy / 2)
                energy_per_offspring = total_offspring_energy / 2
                
                # Each parent loses half their energy
                parent1.energy /= 2
                parent2.energy /= 2
                
                # Reset mating state
                parent1.is_mating = False
                parent2.is_mating = False
                
                # Create two offspring
                for _ in range(2):
                    offspring = Crater.create_offspring(parent1, parent2, energy=energy_per_offspring)
                    offspring.font = self.font
                    new_craters.append(offspring)
                
                # Track mating statistics
                self.mating_events += 1
                self.births += 2
        
        # Add new craters
        self.craters.extend(new_craters)
        
        # Remove dead craters (in reverse order to avoid index issues)
        for i in sorted(craters_to_remove, reverse=True):
            del self.craters[i]
        
        # Count active food
        active_food = sum(1 for food in self.food_pellets if food.active)
        
        # Spawn new food periodically
        self.food_spawn_timer += 1
        if self.food_spawn_timer >= self.food_spawn_interval:
            # Reset timer
            self.food_spawn_timer = 0
            
            # Try to find an inactive food to reactivate
            inactive_food = None
            for food in self.food_pellets:
                if not food.active:
                    inactive_food = food
                    break
            
            # If we found an inactive food, try to place it
            if inactive_food:
                # Generate a random position
                inactive_food.set_position()
                
                # Check if this position overlaps with any food or craters
                position_valid = True
                
                # Check for overlap with other food
                for other_food in self.food_pellets:
                    if other_food is not inactive_food and other_food.active:
                        # Calculate distance between the two food items
                        dx = inactive_food.x - other_food.x
                        dy = inactive_food.y - other_food.y
                        distance = (dx*dx + dy*dy) ** 0.5
                        
                        # If food items are too close, this position is invalid
                        if distance < inactive_food.size + other_food.size:
                            position_valid = False
                            break
                
                # If position is still valid, check for overlap with craters
                if position_valid:
                    for crater in self.craters:
                        # Calculate distance between food and crater
                        dx = inactive_food.x - crater.x
                        dy = inactive_food.y - crater.y
                        distance = (dx*dx + dy*dy) ** 0.5
                        
                        # If food is too close to a crater, this position is invalid
                        if distance < inactive_food.size + crater.size:
                            position_valid = False
                            break
                
                # Only activate food if position is valid
                if position_valid:
                    inactive_food.active = True
                # Otherwise, skip this food spawn opportunity
    
    def _update_crater_batch(self, batch, craters_to_remove, mating_pairs, new_craters):
        """Process a batch of craters for better cache locality"""
        for i, crater in [(i, self.craters[i]) for i in range(len(self.craters)) if self.craters[i] in batch]:
            self._update_single_crater(i, crater, craters_to_remove, mating_pairs, new_craters)
    
    def _update_single_crater(self, i, crater, craters_to_remove, mating_pairs, new_craters):
        """Update a single crater and handle interactions"""
        # Get nearby entities efficiently
        nearby_craters = self.get_nearby_craters(crater)
        nearby_food = self.get_nearby_food(crater)
        
        # Update crater (returns other crater if mating occurred)
        other_crater = crater.update(nearby_craters, nearby_food)
        
        if other_crater and other_crater not in mating_pairs and crater not in mating_pairs:
            mating_pairs.append(crater)
            mating_pairs.append(other_crater)
        
        # Check if crater ran out of energy
        if crater.energy <= 0:
            # Mark for removal and create food pellet at its position
            craters_to_remove.append(i)
            # Create new orange food pellet at crater's position
            new_food = Food(crater.x, crater.y, color=ORANGE_FOOD_COLOR)
            self.food_pellets.append(new_food)
            
    
    def draw(self, surface):
        """
        Draw all entities to the given surface
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw food
        for food in self.food_pellets:
            food.draw(surface)
        
        # Draw craters
        for crater in self.craters:
            if crater.energy > 0:
                crater.draw(surface, self.show_sensors, self.craters)
        
        # Display information
        if self.font:
            active_food = sum(1 for food in self.food_pellets if food.active)
            active_craters = len(self.craters)
            mating_craters = sum(1 for crater in self.craters if crater.is_mating)
            
            # Calculate age statistics
            total_age = sum(crater.age for crater in self.craters) if self.craters else 0
            avg_age = total_age / len(self.craters) if self.craters else 0
            max_age = max((crater.age for crater in self.craters), default=0)
            
            teens = sum(1 for crater in self.craters if crater.age < AGE_TEEN)
            young_craters = sum(1 for crater in self.craters if AGE_TEEN <= crater.age < AGE_YOUNG)
            adult_craters = sum(1 for crater in self.craters if AGE_YOUNG <= crater.age < AGE_ADULT)
            middle_craters = sum(1 for crater in self.craters if AGE_ADULT <= crater.age < AGE_MIDDLE)
            mature_craters = sum(1 for crater in self.craters if AGE_MIDDLE <= crater.age < AGE_MATURE)
            senior_craters = sum(1 for crater in self.craters if AGE_MATURE <= crater.age < AGE_SENIOR)
            elder_craters = sum(1 for crater in self.craters if crater.age >= AGE_SENIOR)
            
            # Calculate generation depth statistics
            if self.craters:
                generation_depths = [crater.generation_depth for crater in self.craters]
                avg_generation = sum(generation_depths) / len(generation_depths)
                max_generation = max(generation_depths)
                min_generation = min(generation_depths)
            else:
                avg_generation = max_generation = min_generation = 0
            
            # Count inactive craters
            inactive_craters = sum(1 for crater in self.craters if crater.inactive_frames > crater.inactivity_threshold)
            avg_inactive_frames = sum(crater.inactive_frames for crater in self.craters) / len(self.craters) if self.craters else 0
            
            # Basic info
            info_text = f"Food: {active_food} | Craters: {active_craters}"
            text_surface = self.font.render(info_text, True, TEXT_COLOR)
            surface.blit(text_surface, (10, 10))
            
            # Generation stats info
            generation_info = f"Generation Stats - Avg: {avg_generation:.1f} | Max: {max_generation} | Min: {min_generation}"
            generation_surface = self.font.render(generation_info, True, TEXT_COLOR)
            surface.blit(generation_surface, (10, 30))
            
            # Mating info
            mating_info = f"Mating Craters: {mating_craters} | Mating Events: {self.mating_events} | Births: {self.births}"
            mating_surface = self.font.render(mating_info, True, TEXT_COLOR)
            surface.blit(mating_surface, (10, 50))
            
            # Age info
            age_info = f"Avg Age: {int(avg_age)} | Max Age: {max_age} | T: {teens} | Y: {young_craters} | A: {adult_craters} | M: {middle_craters} | Ma: {mature_craters} | S: {senior_craters} | E: {elder_craters}"
            age_surface = self.font.render(age_info, True, TEXT_COLOR)
            surface.blit(age_surface, (10, 70))
            
            # Inactivity info
            inactive_info = f"Inactive Craters: {inactive_craters} | Avg Inactive Frames: {int(avg_inactive_frames)}"
            inactive_surface = self.font.render(inactive_info, True, TEXT_COLOR)
            surface.blit(inactive_surface, (10, 90))
            
            # Performance info
            perf_info = f"Frame Time: {self.avg_frame_time*1000:.1f}ms | FPS: {1.0/max(self.avg_frame_time, 0.001):.1f}"
            perf_surface = self.font.render(perf_info, True, TEXT_COLOR)
            surface.blit(perf_surface, (10, 110))
    
    def toggle_sensors(self):
        """Toggle the visibility of sensor rays"""
        self.show_sensors = not self.show_sensors
        
    def force_mating(self, percentage=0.5):
        """
        Force the top percentage of highest-energy craters to mate
        
        Args:
            percentage (float): Percentage of population to mate (0.0-1.0)
        """
        # Get active craters and sort by energy
        active_craters = [crater for crater in self.craters if crater.energy > 0]
        if len(active_craters) < 2:
            return
            
        # Sort by energy (highest first)
        top_craters = sorted(active_craters, key=lambda c: c.energy, reverse=True)
        
        # Select top percentage
        num_to_mate = max(2, int(len(top_craters) * percentage))
        num_to_mate = num_to_mate if num_to_mate % 2 == 0 else num_to_mate - 1  # Ensure even number
        top_craters = top_craters[:num_to_mate]
        
        # Randomize pairings to avoid inbreeding
        random.shuffle(top_craters)
        
        # Create offspring
        new_craters = []
        
        # Increment generation when evolution occurs
        self.generation += 1
        
        # Create pairs and produce offspring
        for i in range(0, num_to_mate, 2):
            if i+1 >= len(top_craters):
                break  # Skip last unpaired crater if somehow we have an odd number
                
            parent1 = top_craters[i]
            parent2 = top_craters[i+1]
            
            # Calculate total energy to distribute to offspring
            total_offspring_energy = (parent1.energy / 2) + (parent2.energy / 2)
            energy_per_offspring = total_offspring_energy / 2
            
            # Each parent loses half their energy
            parent1.energy /= 2
            parent2.energy /= 2
            
            # Reset mating state if they were in it
            parent1.is_mating = False
            parent2.is_mating = False
            
            # Create two offspring
            for _ in range(2):
                offspring = Crater.create_offspring(parent1, parent2, energy=energy_per_offspring)
                offspring.font = self.font
                new_craters.append(offspring)
            
            # Flash the pair with mating color briefly for visual feedback
            parent1.is_mating = True
            parent2.is_mating = True
            parent1.mating_timer = 10  # Very short duration just for visual feedback
            parent2.mating_timer = 10
            
            # Track mating statistics
            self.mating_events += 1
            self.births += 2
            
        # Add new craters
        self.craters.extend(new_craters)
            
        # Print information about forced mating
        print(f"Forced {num_to_mate//2} pairs of craters to mate, creating {len(new_craters)} offspring") 