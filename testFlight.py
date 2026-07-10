import asyncio
from mavsdk import System

async def run():
  drone = System()
  await drone.connect(system_address="udp://:14550")

  print("Waiting connection")
  async for state in drone.core.connection_state():
    if state.is_connected:
      break
  print("Waiting GPS")
  async for health in drone.telemetry.health():
    if health.is_global_position_ok and health.is_home_position_ok:
      print("GPS ready")
      break

  async for position in drone.telemetry.position():
    home_lat = position.latitude_deg
    home_lon = position.longitude_deg
    home_alt = position.absolute_altitude_m
    print(f"Home: {home_lat}, {home_lon}, {home_alt}m")
    break

  print("Arming and taking off")
  await drone.action.set_takeoff_altitude(10)
  await drone.action.arm()
  await drone.action.takeoff()
  await asyncio.sleep(8)

  offset_lat = home_lat+(10/111320)

  print("Flying 10m north")
  await drone.action.goto_location(offset_lat, home_lon, home_alt + 10, 0)
  await asyncio.sleep(10)

  print("Returning home")
  await drone.action.goto_location(home_lat, home_lon, home_alt + 10, 0)
  await asyncio.sleep(10)

  print("Landing")
  await drone.action.land()

  async for in_air in drone.telemetry.in_air():
    if not in_air:
      print("Landed")
      break

asyncio.run(run())
