#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <condition_variable>
#include <mutex>

using namespace mavsdk;
using std::chrono::seconds;
using std::this_thread::sleep_for;

int main() {
    Mavsdk::Configuration config{ComponentType::GroundStation};
    Mavsdk mavsdk(config);

    ConnectionResult connection_result =
        mavsdk.add_any_connection("serial:///dev/ttyTHS1:921600");
    if (connection_result != ConnectionResult::Success) {
        std::cout << "Adding connection failed: " << connection_result << '\n';
        return 1;
    }

    // Wait for the system to connect via heartbeat
    while (mavsdk.systems().size() == 0) {
        sleep_for(seconds(1));
    }
    // System got discovered.
    auto system = mavsdk.systems()[0];
    auto telemetry = Telemetry{system};
    auto action = Action{system};
    
    std::cout << "Waiting for first telemetry health update...\n";
    {
        std::mutex health_mutex;
        std::condition_variable health_cv;
        bool health_received = false;

        Telemetry::HealthHandle handle = telemetry.subscribe_health(
            [&](Telemetry::Health) {
                std::lock_guard<std::mutex> lock(health_mutex);
                health_received = true;
                health_cv.notify_one();
            });

        std::unique_lock<std::mutex> lock(health_mutex);
        if (!health_cv.wait_for(lock, seconds(10), [&] { return health_received; })) {
            std::cout << "No telemetry health data received after 10s.\n";
        }
        telemetry.unsubscribe_health(handle);
    }

    
    // Exit if calibration is required
    Telemetry::Health check_health = telemetry.health();
    bool calibration_required = false;
    if (!check_health.is_gyrometer_calibration_ok) {
        std::cout << "Gyro requires calibration.\n";
        calibration_required = true;
    }
    if (!check_health.is_accelerometer_calibration_ok) {
        std::cout << "Accelerometer requires calibration.\n";
        calibration_required = true;
    }
    if (!check_health.is_magnetometer_calibration_ok) {
        std::cout << "Magnetometer (compass) requires calibration.\n";
        calibration_required = true;
    }
    if (calibration_required) {
        return 1;
    }

    // Check if ready to arm (reporting status)
    while (!telemetry.health_all_ok()) {
        std::cout << "Vehicle not ready to arm. Waiting on:\n";
        Telemetry::Health current_health = telemetry.health();
        if (!current_health.is_global_position_ok) {
            std::cout << "  - GPS fix.\n";
        }
        if (!current_health.is_local_position_ok) {
            std::cout << "  - Local position estimate.\n";
        }
        if (!current_health.is_home_position_ok) {
            std::cout << "  - Home position to be set.\n";
        }
        if (!current_health.is_armable) {
            std::cout << "  - Vehicle not yet armable.\n";
        }
        sleep_for(seconds(1));
    }

    // Final armable check before sending the arm command
    if (!telemetry.health().is_armable) {
        std::cout << "Vehicle reports not armable, aborting.\n";
        return 1;
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
        sleep_for(seconds(1));
        elapsed_seconds++;
    }
    std::cout << "Disarmed, exiting.\n";
    return 0;
}
