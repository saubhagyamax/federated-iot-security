"""
Multi-Device Simulator for Testing Federated Learning IoT Security Platform
Simulates multiple IoT devices with different behaviors and attack scenarios
"""

import time
import random
from datetime import datetime
from typing import List, Dict
import sys
import os

# Add parent directory to path to import the SDK
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from device_client_sdk import FLIoTClient, DeviceConfig, DeviceMetrics


class DeviceSimulator:
    """Simulates a single IoT device with configurable behavior"""
    
    def __init__(self, client: FLIoTClient, behavior: str = "normal"):
        self.client = client
        self.behavior = behavior
        self.attack_started = False
        self.attack_start_time = None
        
    def get_simulated_metrics(self) -> DeviceMetrics:
        """Generate simulated metrics based on device behavior"""
        config = self.client.config
        
        if self.behavior == "normal":
            # Normal operation with small variations
            cpu = config.baseline_cpu + random.uniform(-3, 3)
            memory = config.baseline_memory + random.uniform(-5, 5)
            network = config.baseline_network + random.uniform(-10, 10)
            requests = int(config.baseline_requests + random.uniform(-3, 3))
            connections = random.randint(0, 5)
            
        elif self.behavior == "botnet":
            # Simulate botnet activity - high scanning, high network
            cpu = config.baseline_cpu + random.uniform(20, 40)
            memory = config.baseline_memory + random.uniform(10, 20)
            network = config.baseline_network + random.uniform(100, 200)
            requests = int(config.baseline_requests + random.uniform(50, 100))
            connections = random.randint(50, 200)
            
        elif self.behavior == "cryptomining":
            # Simulate cryptomining - very high CPU, moderate network
            cpu = config.baseline_cpu + random.uniform(60, 80)
            memory = config.baseline_memory + random.uniform(30, 50)
            network = config.baseline_network + random.uniform(20, 40)
            requests = int(config.baseline_requests + random.uniform(-2, 5))
            connections = random.randint(5, 15)
            
        elif self.behavior == "data_exfiltration":
            # Simulate data theft - very high network, high connections
            cpu = config.baseline_cpu + random.uniform(10, 20)
            memory = config.baseline_memory + random.uniform(15, 25)
            network = config.baseline_network + random.uniform(150, 300)
            requests = int(config.baseline_requests + random.uniform(10, 30))
            connections = random.randint(20, 50)
            
        elif self.behavior == "gradual_compromise":
            # Slowly increasing anomalous behavior (harder to detect)
            if not self.attack_started:
                self.attack_started = True
                self.attack_start_time = time.time()
            
            elapsed = time.time() - self.attack_start_time
            factor = min(1.0, elapsed / 300)  # Ramp up over 5 minutes
            
            cpu = config.baseline_cpu + random.uniform(0, 30) * factor
            memory = config.baseline_memory + random.uniform(0, 20) * factor
            network = config.baseline_network + random.uniform(0, 100) * factor
            requests = int(config.baseline_requests + random.uniform(0, 50) * factor)
            connections = random.randint(0, int(50 * factor))
            
        else:  # intermittent
            # Randomly switch between normal and suspicious
            if random.random() < 0.3:  # 30% chance of anomaly
                cpu = config.baseline_cpu + random.uniform(30, 50)
                memory = config.baseline_memory + random.uniform(20, 40)
                network = config.baseline_network + random.uniform(50, 100)
                requests = int(config.baseline_requests + random.uniform(20, 40))
                connections = random.randint(20, 50)
            else:
                cpu = config.baseline_cpu + random.uniform(-3, 3)
                memory = config.baseline_memory + random.uniform(-5, 5)
                network = config.baseline_network + random.uniform(-10, 10)
                requests = int(config.baseline_requests + random.uniform(-3, 3))
                connections = random.randint(0, 5)
        
        return DeviceMetrics(
            cpu_usage=max(0, cpu),
            memory_usage=max(0, memory),
            network_traffic=max(0, network),
            request_count=max(0, requests),
            connection_attempts=max(0, connections)
        )


class SimulationOrchestrator:
    """Manages multiple device simulators"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.devices: List[tuple[FLIoTClient, DeviceSimulator]] = []
        self.threats_detected = []
        self.model_updates = []
        
    def add_device(
        self,
        device_id: str,
        device_type: str,
        name: str,
        behavior: str = "normal",
        baseline_cpu: float = None,
        baseline_network: float = None
    ):
        """Add a simulated device to the fleet"""
        
        # Set baselines based on device type if not specified
        if baseline_cpu is None:
            baselines = {
                "camera": (15.0, 50.0),
                "sensor": (10.0, 30.0),
                "thermostat": (8.0, 20.0),
                "lock": (5.0, 15.0),
                "medical": (20.0, 40.0),
                "industrial": (25.0, 60.0)
            }
            baseline_cpu, baseline_network = baselines.get(device_type, (10.0, 30.0))
        
        config = DeviceConfig(
            device_id=device_id,
            device_type=device_type,
            name=name,
            organization_id="org-simulation-001",
            api_base_url=self.api_base_url,
            baseline_cpu=baseline_cpu,
            baseline_network=baseline_network if baseline_network else baseline_cpu * 3,
            metrics_interval=15,  # 15 seconds
            model_update_interval=120  # 2 minutes
        )
        
        client = FLIoTClient(config)
        simulator = DeviceSimulator(client, behavior)
        
        # Set up callbacks
        client.on_threat_detected = self._on_threat_detected
        client.on_model_updated = self._on_model_updated
        
        # Override metrics collection to use simulator
        original_collect = client._collect_metrics
        client._collect_metrics = lambda: simulator.get_simulated_metrics()
        
        self.devices.append((client, simulator))
        
        print(f"✅ Added device: {name} ({device_type}) - Behavior: {behavior}")
    
    def _on_threat_detected(self, threat):
        """Callback when threat is detected"""
        self.threats_detected.append({
            **threat,
            "detected_at": datetime.utcnow()
        })
        print(f"\n🚨 THREAT DETECTED: {threat['threat_type']}")
        print(f"   Device: {threat['device_name']}")
        print(f"   Severity: {threat['severity']}")
        print(f"   Confidence: {threat['confidence']}\n")
    
    def _on_model_updated(self, model):
        """Callback when model is updated"""
        self.model_updates.append({
            "version": model['version'],
            "accuracy": model['accuracy'],
            "updated_at": datetime.utcnow()
        })
        print(f"\n📊 GLOBAL MODEL UPDATED: Version {model['version']}")
        print(f"   Accuracy: {model['accuracy']:.3f}")
        print(f"   Devices contributed: {model['devices_contributed']}\n")
    
    def start_all(self):
        """Start all simulated devices"""
        print("\n🚀 Starting all devices...")
        for client, _ in self.devices:
            if client.start():
                print(f"   ✓ {client.config.name} started")
            else:
                print(f"   ✗ {client.config.name} failed to start")
        print()
    
    def stop_all(self):
        """Stop all simulated devices"""
        print("\n🛑 Stopping all devices...")
        for client, _ in self.devices:
            client.stop()
            print(f"   ✓ {client.config.name} stopped")
        print()
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        total_devices = len(self.devices)
        running_devices = sum(1 for c, _ in self.devices if c.is_running)
        
        behaviors = {}
        for _, sim in self.devices:
            behaviors[sim.behavior] = behaviors.get(sim.behavior, 0) + 1
        
        return {
            "total_devices": total_devices,
            "running_devices": running_devices,
            "threats_detected": len(self.threats_detected),
            "model_updates": len(self.model_updates),
            "behaviors": behaviors,
            "current_model_version": self.devices[0][0].current_model_version if self.devices else 0
        }
    
    def print_status(self):
        """Print current simulation status"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("SIMULATION STATUS")
        print("="*60)
        print(f"Devices Running: {stats['running_devices']}/{stats['total_devices']}")
        print(f"Threats Detected: {stats['threats_detected']}")
        print(f"Model Updates: {stats['model_updates']}")
        print(f"Current Model Version: {stats['current_model_version']}")
        print("\nDevice Behaviors:")
        for behavior, count in stats['behaviors'].items():
            print(f"  - {behavior}: {count} devices")
        print("="*60 + "\n")


# ============================================================================
# PRE-CONFIGURED SIMULATION SCENARIOS
# ============================================================================

def scenario_basic():
    """Basic scenario: Mix of normal devices"""
    print("\n📋 SCENARIO: Basic Mixed Fleet")
    print("   - 5 normal devices")
    print("   - 1 compromised device (botnet)")
    print()
    
    sim = SimulationOrchestrator()
    
    # Normal devices
    sim.add_device("camera-001", "camera", "Front Door Camera", "normal")
    sim.add_device("sensor-001", "sensor", "Temperature Sensor", "normal")
    sim.add_device("thermostat-001", "thermostat", "Living Room Thermostat", "normal")
    sim.add_device("lock-001", "lock", "Main Door Lock", "normal")
    sim.add_device("camera-002", "camera", "Garage Camera", "normal")
    
    # Compromised device
    sim.add_device("camera-003", "camera", "Back Door Camera", "botnet")
    
    return sim


def scenario_industrial():
    """Industrial IoT scenario with multiple attack types"""
    print("\n📋 SCENARIO: Industrial IoT Facility")
    print("   - 8 industrial sensors (normal)")
    print("   - 1 cryptomining attack")
    print("   - 1 data exfiltration attack")
    print()
    
    sim = SimulationOrchestrator()
    
    # Normal industrial sensors
    for i in range(1, 9):
        sim.add_device(
            f"plc-{i:03d}",
            "industrial",
            f"Industrial Sensor {i}",
            "normal"
        )
    
    # Attacks
    sim.add_device("plc-009", "industrial", "Industrial Sensor 9", "cryptomining")
    sim.add_device("plc-010", "industrial", "Industrial Sensor 10", "data_exfiltration")
    
    return sim


def scenario_gradual_compromise():
    """Scenario with hard-to-detect gradual compromise"""
    print("\n📋 SCENARIO: Gradual Compromise (APT Simulation)")
    print("   - 6 normal devices")
    print("   - 2 devices with gradual compromise")
    print("   - Watch how FL adapts to detect slow attacks")
    print()
    
    sim = SimulationOrchestrator()
    
    # Normal devices
    for i in range(1, 7):
        sim.add_device(
            f"device-{i:03d}",
            "sensor",
            f"Sensor {i}",
            "normal"
        )
    
    # Gradual compromise (APT-style)
    sim.add_device("device-007", "sensor", "Sensor 7", "gradual_compromise")
    sim.add_device("device-008", "sensor", "Sensor 8", "gradual_compromise")
    
    return sim


def scenario_intermittent():
    """Scenario with intermittent attacks (hardest to detect)"""
    print("\n📋 SCENARIO: Intermittent Attacks")
    print("   - 5 normal devices")
    print("   - 3 devices with intermittent suspicious behavior")
    print()
    
    sim = SimulationOrchestrator()
    
    # Normal devices
    for i in range(1, 6):
        sim.add_device(f"cam-{i:03d}", "camera", f"Camera {i}", "normal")
    
    # Intermittent attacks
    for i in range(6, 9):
        sim.add_device(f"cam-{i:03d}", "camera", f"Camera {i}", "intermittent")
    
    return sim


# ============================================================================
# MAIN SIMULATION RUNNER
# ============================================================================

def run_simulation(scenario_func, duration_minutes: int = 10):
    """Run a simulation scenario for specified duration"""
    
    print("\n" + "="*60)
    print("FEDERATED LEARNING IoT SECURITY - DEVICE SIMULATOR")
    print("="*60)
    
    # Create scenario
    sim = scenario_func()
    
    # Start all devices
    sim.start_all()
    
    print(f"⏱️  Simulation will run for {duration_minutes} minutes")
    print("   Press Ctrl+C to stop early\n")
    
    # Run for specified duration
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            remaining = int((end_time - time.time()) / 60)
            print(f"⏱️  Time remaining: {remaining} minutes", end="\r")
            time.sleep(30)  # Update every 30 seconds
            
            # Print status every 2 minutes
            if int(time.time() - start_time) % 120 == 0:
                sim.print_status()
        
        print("\n\n⏱️  Simulation time completed!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation stopped by user")
    
    finally:
        # Stop all devices
        sim.stop_all()
        
        # Final statistics
        print("\n" + "="*60)
        print("FINAL SIMULATION RESULTS")
        print("="*60)
        
        stats = sim.get_statistics()
        print(f"\nTotal Runtime: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Devices Simulated: {stats['total_devices']}")
        print(f"Threats Detected: {stats['threats_detected']}")
        print(f"Model Updates: {stats['model_updates']}")
        print(f"Final Model Version: {stats['current_model_version']}")
        
        if sim.threats_detected:
            print("\n🚨 Detected Threats:")
            for threat in sim.threats_detected[-10:]:  # Last 10
                print(f"   - {threat['threat_type']} ({threat['severity']}) "
                      f"on {threat['device_name']}")
        
        print("\n✅ Simulation complete!")
        print("="*60 + "\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IoT Device Fleet Simulator")
    parser.add_argument(
        "--scenario",
        choices=["basic", "industrial", "gradual", "intermittent"],
        default="basic",
        help="Simulation scenario to run"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Simulation duration in minutes"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL"
    )
    
    args = parser.parse_args()
    
    # Map scenario names to functions
    scenarios = {
        "basic": scenario_basic,
        "industrial": scenario_industrial,
        "gradual": scenario_gradual_compromise,
        "intermittent": scenario_intermittent
    }
    
    # Run the simulation
    run_simulation(scenarios[args.scenario], args.duration)