import random
import time
from src.utils import Direction

# Function to generate random sequences respecting maximum duration
def generate_random_sequence(max_duration):
    sequence = []
    remaining_duration = max_duration
    
    while remaining_duration > 0:
        # Calculate maximum possible duration for this direction
        max_duration_for_direction = min(remaining_duration, random.uniform(2, 3))  # Random 2 to 4 seconds per direction
        
        # Check if adding a direction is feasible
        if max_duration_for_direction >= 2:
            # Choose a random direction
            direction = random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])
            sequence.append((direction, max_duration_for_direction))
            remaining_duration -= max_duration_for_direction
            
            # Check if there's enough time for a straight segment
            if remaining_duration > 0:
                max_duration_for_straight = min(remaining_duration, random.uniform(2, 3))
                sequence.append((Direction.STRAIGHT, max_duration_for_straight))
                remaining_duration -= max_duration_for_straight
        else:
            break
    
    return sequence

# Example usage:
if __name__ == "__main__":
    max_duration = 10  # Maximum total duration of 10 seconds
    random_sequence = generate_random_sequence(max_duration)
    total_duration = sum(duration for _, duration in random_sequence)
    
    for direction, duration in random_sequence:
        print(direction.value, "- Duration:", round(duration, 2), "seconds")
        time.sleep(duration)
    
    print("Total Duration:", round(total_duration, 2), "seconds")
