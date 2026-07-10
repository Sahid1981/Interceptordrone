#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <chrono>
#include <thread>
#include <iostream>

using namespace mavsdk;
using std::chrono::seconds;
using std::this_thread::sleep_for;

int main() {
    Mavsdk::Configuration config{Mavsdk::ComponentType::GroundStation};
    config.set_compatibility_mode(Mavsdk::ComponentType::ArduPilot);
    Mavsdk mavsdk(config);

    ConnectionResult connection_result = mavsdk.add_serial_connection("/dev/ttyTHS1", 921600);
    if (connection_result != ConnectionResult::Success) {
        std::cout << "Adding connection failed: " << connection_result << '\n';
        return 1;
    }

    // Wait for the system to connect via heartbeat
    while (mavsdk.systems().size() == 0) {
        sleep_for(seconds(1));
    }
    // System got discovered.
    System system = mavsdk.systems()[0];
    auto telemetry = Telemetry{system};
    auto action = Action{system};

    // Exit if calibration is required
    Telemetry::Health check_health = telemetry.health();
    bool calibration_required = false;
    if (!check_health.gyrometer_calibration_ok) {
        std::cout << "Gyro requires calibration.\n";
        calibration_required = true;
    }
    if (!check_health.accelerometer_calibration_ok) {
        std::cout << "Accelerometer requires calibration.\n";
        calibration_required = true;
    }
    if (!check_health.magnetometer_calibration_ok) {
        std::cout << "Magnetometer (compass) requires calibration.\n";
        calibration_required = true;
    }
    if (!check_health.level_calibration_ok) {
        std::cout << "Level calibration required.\n";
        calibration_required = true;
    }
    if (calibration_required) {
        return 1;
    }

    // Check if ready to arm (reporting status)
    while (!telemetry.health_all_ok()) {
        std::cout << "Vehicle not ready to arm. Waiting on:\n";
        Telemetry::Health current_health = telemetry.health();
        if (!current_health.global_position_ok) {
            std::cout << "  - GPS fix.\n";
        }
        if (!current_health.local_position_ok) {
            std::cout << "  - Local position estimate.\n";
        }
        if (!current_health.home_position_ok) {
            std::cout << "  - Home position to be set.\n";
        }
        sleep_for(seconds(1));
    }

    // Arm drone
    std::cout << "Arming...\n";
    const Action::Result arm_result = action.arm();
    if (arm_result != Action::Result::Success) {
        std::cout << "Arming failed: " << arm_result << '\n';
        return 1;
    }

    action.set_takeoff_altitude(2.0f);

    // Take off
    std::cout << "Taking off...\n";
    const Action::Result takeoff_result = action.takeoff();
    if (takeoff_result != Action::Result::Success) {
        std::cout << "Takeoff failed: " << takeoff_result << '\n';
        return 1;
    }

    sleep_for(seconds(5));


    Action::Result land_result = action.land();
if (land_result != Action::Result::Success) {
    sleep_for(seconds(5));
    land_result = action.land();
    if (land_result != Action::Result::Success) {
        const Action::Result kill_result = action.kill();
        if (kill_result != Action::Result::Success) {
            std::cout << "Failed to kill drone: " << kill_result << '\n';
            return 1;
        }
    }
}

const int max_wait_seconds = 60;
int elapsed_seconds = 0;

while (telemetry.armed()) {
    if (elapsed_seconds >= max_wait_seconds) {
        std::cout << "Timed out waiting for disarm after " << max_wait_seconds << "s.\n";
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    elapsed_seconds++;
}
std::cout << "Disarmed, exiting.\n";
return 0;

}