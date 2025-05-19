import psutil
import requests
import time
import statistics
from datetime import datetime

class FastAPIMonitor:
    def __init__(self, api_process_name, api_url, api_payload=None):
        """
        Initialize the monitoring tool.
        
        Args:
            api_process_name: Name pattern to identify the FastAPI process (e.g., "uvicorn")
            api_url: URL of the API endpoint to monitor
            api_payload: JSON payload for POST requests
        """
        self.api_process_name = api_process_name
        self.api_url = api_url
        self.api_payload = api_payload or {}
        self.pid = None
        self.request_times = []
        
    def find_api_process(self):
        """Find the PID of the FastAPI process."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if process name or command line contains our API process name
                if (self.api_process_name in proc.info['name'] or 
                    any(self.api_process_name in cmd for cmd in proc.info['cmdline'] if cmd)):
                    self.pid = proc.info['pid']
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False
    
    def get_process_metrics(self):
        """Get CPU and memory usage of the FastAPI process."""
        if not self.pid:
            if not self.find_api_process():
                return None, None
        
        try:
            process = psutil.Process(self.pid)
            cpu_percent = process.cpu_percent(interval=0.5)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            return cpu_percent, memory_mb
        except psutil.NoSuchProcess:
            self.pid = None
            return self.get_process_metrics()
    
    def measure_latency(self, num_requests=3):
        """Measure API latency by making test requests."""
        latencies = []
        total_size = 0
        
        for _ in range(num_requests):
            start_time = time.time()
            try:
                response = requests.post(self.api_url, json=self.api_payload, timeout=10)
                end_time = time.time()
                
                if response.status_code in [200, 201]:
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    latencies.append(latency)
                    total_size += len(response.content)
                    self.request_times.append(end_time)
            except requests.RequestException as e:
                pass
        
        # Clean up old request times (keep only last minute)
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t <= 60]
        
        # Calculate metrics
        avg_latency = statistics.mean(latencies) if latencies else 0
        requests_per_second = len(self.request_times) / 60 if self.request_times else 0
        throughput = (total_size / 1024) / (num_requests * avg_latency / 1000) if avg_latency > 0 else 0
        
        # Calculate inference time (approximated as latency minus network overhead)
        # This is an approximation as true inference time would need to be measured inside the API
        inference_time = avg_latency * 0.85 if avg_latency > 0 else 0  # Assume ~85% of latency is inference
        
        return avg_latency, inference_time, requests_per_second, throughput
    
    def get_metrics(self):
        """Get all metrics as a dictionary."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu_percent, memory_mb = self.get_process_metrics()
        
        if cpu_percent is None:
            return {"error": "Process not found"}
                
        latency, inference_time, rps, throughput = self.measure_latency()
        
        # Create metrics dict
        metrics = {
            "timestamp": timestamp,
            "cpu_percent": round(cpu_percent, 2),
            "memory_mb": round(memory_mb, 2),
            "api_latency_ms": round(latency, 2),
            "inference_time_ms": round(inference_time, 2),
            "requests_per_second": round(rps, 2),
            "throughput_kb_per_sec": round(throughput, 2)
        }
        
        return metrics

# When this file is run directly, get metrics and print them
if __name__ == "__main__":
    PROCESS_NAME = "uvicorn"  # Change this if your FastAPI app uses a different server
    API_URL = "http://localhost:8000/predict"
    API_PAYLOAD = {
        "start_date": "2023-11-01",
        "end_date": "2024-02-01"
    }
    
    monitor = FastAPIMonitor(PROCESS_NAME, API_URL, API_PAYLOAD)
    metrics = monitor.get_metrics()
    print(metrics)
