import asyncio
import sys
from mavsdk import System
from mavsdk.action import ActionError


async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14550")

    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break

    print("Waiting for GPS lock and home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position OK")
            break

    print("Checking current flight mode...")
    async for flight_mode in drone.telemetry.flight_mode():
        print(f"-- Current flight mode: {flight_mode}")
        break

    try:
        print("Setting flight mode to Hold (pre-arm safe state)...")
        await drone.action.hold()
        print("-- Mode set to Hold")
    except ActionError as e:
        print(f"!! Failed to set Hold mode: {e}")
        sys.exit(1)

    await asyncio.sleep(1)

    try:
        print("Arming...")
        await drone.action.arm()
    except ActionError as e:
        print(f"!! Arm command rejected: {e}")
        sys.exit(1)

    armed_confirmed = False
    async for is_armed in drone.telemetry.armed():
        if is_armed:
            armed_confirmed = True
            print("-- Armed confirmed via telemetry")
            break
        else:
            print("-- Waiting for armed confirmation...")
        await asyncio.sleep(0.5)

    if not armed_confirmed:
        print("!! Arm never confirmed - aborting")
        sys.exit(1)

    try:
        print("Setting takeoff altitude to 2m...")
        await drone.action.set_takeoff_altitude(1.0)
        print("-- Takeoff altitude set")
    except ActionError as e:
        print(f"!! Failed to set takeoff altitude: {e}")
        sys.exit(1)

    try:
        print("Taking off...")
        await drone.action.takeoff()
    except ActionError as e:
        print(f"!! Takeoff command rejected: {e}")
        sys.exit(1)

    print("Waiting for confirmed takeoff (in_air)...")
    took_off = False
    for _ in range(20):
        async for in_air in drone.telemetry.in_air():
            if in_air:
                took_off = True
                print("-- Confirmed in air")
            break
        if took_off:
            break
        await asyncio.sleep(0.5)

    if not took_off:
        print("!! Never left the ground - aborting, attempting disarm for safety")
        try:
            await drone.action.disarm()
        except ActionError:
            pass
        sys.exit(1)

    print("Holding at 2m for 2 seconds...")
    await asyncio.sleep(2)

    try:
        print("Landing...")
        await drone.action.land()
    except ActionError as e:
        print(f"!! Land command rejected: {e}")
        sys.exit(1)

    print("Waiting for landing to complete...")
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("-- Landed!")
            break

    try:
        print("Disarming...")
        await drone.action.disarm()
        print("-- Disarmed")
    except ActionError as e:
        print(f"-- Disarm note: {e} (may already be disarmed automatically)")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(run())