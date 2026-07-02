import time
from zenmav.core import Zenmav

def main():
    # Connect to the drone (adjust for your connection)
    # Example for USB connection: drone = Zenmav(ip="/dev/ttyACM0")
    # Example for SITL simulation: drone = Zenmav()
    drone = Zenmav(ip="/dev/ttyACM0") 

    # Arm and set mode
    drone.arm()
    drone.set_mode("GUIDED")

    # 1. Take off to a safe altitude
    
    takeoff_alt = 1.0 
    drone.takeoff(altitude=takeoff_alt)
    time.sleep(2) # Wait for it to stabilize

    #START OF YOUR TEST SEQUENCE
    #X = North, Y = East, Z = Down (negative is up) 

    # 2. Fly 15 cm (0.15 m) up (which is negative down)
    print("Moving Up...")
    drone.local_target([0, 0, -0.15])
    time.sleep(1) # Wait for movement to complete

    # 3. Turn right then left
    print("Turning Right...")
    drone.set_yaw(90, 0, 0) # setyaw ANGLE ANGULAR_SPEED MODE
    time.sleep(1)
    print("Turning Left...")
    drone.set_yaw(-90, 0, 0)
    time.sleep(1)

    # 4. Go forward 15 cm 
    print("Moving Forward...")
    drone.local_target([0.15, 0, 0])
    time.sleep(1)

    # 5. Go back 15 cm 
    print("Moving Back...")
    drone.local_target([-0.15, 0, 0])
    time.sleep(1)

    # 6. Go down 15 cm e
    print("Moving Down...")
    drone.local_target([0, 0, 0.15])
    time.sleep(1)
    

    
    print("Returning to Launch...")
    drone.RTL()

if __name__ == "__main__":
    main()