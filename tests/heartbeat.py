from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyTHS1', baud=921600)
msg = master.wait_heartbeat(timeout=5)
print("target_system:", master.target_system)
print("target_component:", master.target_component)
print("autopilot type:", msg.autopilot)
print("vehicle type:", msg.type)
