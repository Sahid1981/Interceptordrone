import time
from zenmav.core import Zenmav

def main():
    drone = Zenmav(ip="/dev/ttyACM0") 
    
    # Arm and set mode
    drone.arm()
    drone.set_mode("GUIDED")
    print(" GUIDED mode set. Pilot can take over at any time by switching flight mode.")
    
    # 1. Take off to a safe altitude
    takeoff_alt = 1.0 
    drone.takeoff(altitude=takeoff_alt)
    time.sleep(3)  # Wait for it to stabilize


    # Fly 15 cm up (0.15m)
    print(" Moving Up...")
    drone.local_target([0, 0, -0.15])
    time.sleep(2)  # Wait for movement to complete

    # Turn right then left
    print(" Turning Right...")
    drone.set_yaw(90, 0, 0)
    time.sleep(2)
    
    print(" Turning Left...")
    drone.set_yaw(-90, 0, 0)
    time.sleep(2)

    # Go forward 15 cm
    print(" Moving Forward...")
    drone.local_target([0.15, 0, 0])
    time.sleep(2)

    # Go back 15 cm
    print(" Moving Back...")
    drone.local_target([-0.15, 0, 0])
    time.sleep(2)

    # Go down 15 cm
    print(" Moving Down...")
    drone.local_target([0, 0, 0.15])
    time.sleep(2)
    
    # --- END OF YOUR TEST SEQUENCE ---
    
    print(" Returning to Launch...")
    drone.RTL()

if __name__ == "__main__":
    main()