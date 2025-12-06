"""
Performance benchmarking for DQN TicTacToe v2.
Measures training speed, inference time, and memory usage.
"""

import time
import torch
import numpy as np
import psutil
import os
from datetime import datetime
from typing import Dict, List, Any
import json
import gc

from models.ReinforcementLearning.DeepQ_TicTacToe_v2.DeepQAgent import DeepQAgent
from models.ReinforcementLearning.DeepQ_TicTacToe_v2.TicTacToeGame import TicTacToeGame, OPPONENT_LEVEL
from models.ReinforcementLearning.Utils import train_agent, test_agent
from models.common import get_root_directory


class PerformanceBenchmark:
    """Performance benchmarking suite for DQN TicTacToe."""
    
    def __init__(self, device: torch.device = None):
        """Initialize benchmark suite."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.process = psutil.Process(os.getpid())
        
    def measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            "percent": self.process.memory_percent()
        }
    
    def measure_gpu_memory(self) -> Dict[str, float]:
        """Measure GPU memory usage if available."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1024 / 1024
        }
    
    def benchmark_initialization(self, num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark agent initialization time."""
        print("Benchmarking initialization...")
        times = []
        memory_before = self.measure_memory()
        
        for _ in range(num_runs):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            start_time = time.time()
            agent = DeepQAgent(
                device=self.device,
                epsilon=1.0,
                gamma=0.99,
                state_space=9,
                action_space=9,
                hidden_size=256,
                dropout=0.1,
                memory_max_len=10000
            )
            end_time = time.time()
            times.append(end_time - start_time)
            
            del agent
        
        memory_after = self.measure_memory()
        
        return {
            "mean_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "memory_increase_mb": memory_after["rss_mb"] - memory_before["rss_mb"]
        }
    
    def benchmark_forward_pass(self, batch_sizes: List[int] = [1, 32, 64, 128]) -> Dict[str, Any]:
        """Benchmark forward pass performance."""
        print("Benchmarking forward pass...")
        agent = DeepQAgent(
            device=self.device,
            epsilon=0.0,  # No exploration for consistent results
            gamma=0.99,
            state_space=9,
            action_space=9
        )
        agent.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            states = torch.randn(batch_size, 9).to(self.device)
            
            # Warmup
            for _ in range(10):
                _ = agent.forward(states)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(100):
                start_time = time.time()
                with torch.no_grad():
                    _ = agent.forward(states)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[f"batch_{batch_size}"] = {
                "mean_time_ms": np.mean(times) * 1000,
                "std_time_ms": np.std(times) * 1000,
                "throughput_samples_per_sec": batch_size / np.mean(times)
            }
        
        return results
    
    def benchmark_action_selection(self, num_runs: int = 1000) -> Dict[str, Any]:
        """Benchmark action selection performance."""
        print("Benchmarking action selection...")
        agent = DeepQAgent(
            device=self.device,
            epsilon=0.5,
            gamma=0.99,
            state_space=9,
            action_space=9
        )
        agent.eval()
        
        state = torch.zeros(9).to(self.device)
        mask = torch.ones(9).to(self.device)
        indices = list(range(9))
        
        # Warmup
        for _ in range(100):
            _ = agent.select_action(state, mask, indices)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = agent.select_action(state, mask, indices)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "mean_time_us": np.mean(times) * 1e6,  # microseconds
            "std_time_us": np.std(times) * 1e6,
            "actions_per_second": 1 / np.mean(times)
        }
    
    def benchmark_memory_operations(self, num_experiences: int = 10000) -> Dict[str, Any]:
        """Benchmark experience replay memory operations."""
        print("Benchmarking memory operations...")
        agent = DeepQAgent(
            device=self.device,
            epsilon=1.0,
            gamma=0.99,
            state_space=9,
            action_space=9,
            memory_max_len=10000
        )
        
        # Benchmark remember operation
        state = torch.zeros(9).to(self.device)
        next_state = torch.zeros(9).to(self.device)
        
        start_time = time.time()
        for i in range(num_experiences):
            action = i % 9
            reward = float(i % 3 - 1)
            done = i % 10 == 0
            agent.remember(state, action, reward, next_state, done)
        remember_time = time.time() - start_time
        
        # Benchmark replay operation
        optimizer = torch.optim.Adam(agent.policy_network.parameters())
        criterion = torch.nn.MSELoss()
        
        replay_times = []
        for _ in range(100):
            start_time = time.time()
            agent.replay(optimizer, criterion)
            end_time = time.time()
            replay_times.append(end_time - start_time)
        
        return {
            "remember_throughput_per_sec": num_experiences / remember_time,
            "replay_mean_time_ms": np.mean(replay_times) * 1000,
            "replay_std_time_ms": np.std(replay_times) * 1000,
            "memory_size_mb": agent.memory.__sizeof__() / 1024 / 1024
        }
    
    def benchmark_game_performance(self, num_games: int = 100) -> Dict[str, Any]:
        """Benchmark game playing performance."""
        print("Benchmarking game performance...")
        agent = DeepQAgent(
            device=self.device,
            epsilon=0.1,
            gamma=0.99,
            state_space=9,
            action_space=9
        )
        
        # Load a trained model if available
        model_path = os.path.join(
            get_root_directory(),
            "trained_models/ReinforcementLearning/TicTacToeV2/FINAL_MODEL.pt"
        )
        if os.path.exists(model_path):
            try:
                agent.load_model(model_path)
            except:
                pass
        
        agent.eval()
        
        results = {
            "naive": {"times": [], "moves": []},
            "optimal": {"times": [], "moves": []}
        }
        
        for opponent in ["naive", "optimal"]:
            level = OPPONENT_LEVEL.NAIVE if opponent == "naive" else OPPONENT_LEVEL.OPTIMAL
            
            for _ in range(num_games):
                game = TicTacToeGame(self.device, agent, level)
                
                start_time = time.time()
                moves = 0
                done = False
                
                while not done:
                    state = game.get_state()
                    _, mask, indices = game.get_valid_moves()
                    action = agent.select_action(state, mask, indices)
                    _, _, done = game.take_action(action)
                    
                    if not done:
                        _, _, done = game.move()
                    
                    moves += 1
                    
                    if moves > 50:  # Prevent infinite games
                        break
                
                end_time = time.time()
                
                results[opponent]["times"].append(end_time - start_time)
                results[opponent]["moves"].append(moves)
        
        return {
            "naive_opponent": {
                "mean_game_time_ms": np.mean(results["naive"]["times"]) * 1000,
                "mean_moves_per_game": np.mean(results["naive"]["moves"]),
                "games_per_second": 1 / np.mean(results["naive"]["times"])
            },
            "optimal_opponent": {
                "mean_game_time_ms": np.mean(results["optimal"]["times"]) * 1000,
                "mean_moves_per_game": np.mean(results["optimal"]["moves"]),
                "games_per_second": 1 / np.mean(results["optimal"]["times"])
            }
        }
    
    def benchmark_training_speed(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """Benchmark training speed."""
        print(f"Benchmarking training speed ({num_episodes} episodes)...")
        
        agent = DeepQAgent(
            device=self.device,
            epsilon=1.0,
            gamma=0.99,
            state_space=9,
            action_space=9,
            train_start=100,
            batch_size=32
        )
        
        environment = TicTacToeGame(self.device, agent, OPPONENT_LEVEL.NAIVE)
        optimizer = torch.optim.Adam(agent.policy_network.parameters())
        criterion = torch.nn.MSELoss()
        
        memory_before = self.measure_memory()
        gpu_memory_before = self.measure_gpu_memory()
        
        start_time = time.time()
        metrics = train_agent(
            agent=agent,
            environment=environment,
            num_episodes=num_episodes,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            save_every=-1  # Don't save during benchmark
        )
        end_time = time.time()
        
        memory_after = self.measure_memory()
        gpu_memory_after = self.measure_gpu_memory()
        
        training_time = end_time - start_time
        
        return {
            "total_time_seconds": training_time,
            "episodes_per_second": num_episodes / training_time,
            "final_win_rate": metrics["wins"] / num_episodes,
            "memory_increase_mb": memory_after["rss_mb"] - memory_before["rss_mb"],
            "gpu_memory_increase_mb": gpu_memory_after["allocated_mb"] - gpu_memory_before["allocated_mb"]
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print(f"Running full benchmark on {self.device}...")
        print("=" * 60)
        
        results = {
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
            }
        }
        
        # Run benchmarks
        results["initialization"] = self.benchmark_initialization()
        results["forward_pass"] = self.benchmark_forward_pass()
        results["action_selection"] = self.benchmark_action_selection()
        results["memory_operations"] = self.benchmark_memory_operations()
        results["game_performance"] = self.benchmark_game_performance()
        results["training_speed"] = self.benchmark_training_speed(num_episodes=500)
        
        return results
    
    def save_results(self, results: Dict[str, Any], filepath: str = None):
        """Save benchmark results to file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"benchmark_results_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        print(f"\nDevice: {results['device']}")
        print(f"Timestamp: {results['timestamp']}")
        
        print("\nInitialization Performance:")
        print(f"  Mean time: {results['initialization']['mean_time_ms']:.2f} ms")
        
        print("\nForward Pass Performance:")
        for batch_size, metrics in results['forward_pass'].items():
            print(f"  {batch_size}: {metrics['throughput_samples_per_sec']:.0f} samples/sec")
        
        print("\nAction Selection:")
        print(f"  Mean time: {results['action_selection']['mean_time_us']:.1f} μs")
        print(f"  Actions/sec: {results['action_selection']['actions_per_second']:.0f}")
        
        print("\nTraining Performance:")
        print(f"  Episodes/sec: {results['training_speed']['episodes_per_second']:.1f}")
        print(f"  Final win rate: {results['training_speed']['final_win_rate']:.2%}")
        
        print("\nGame Performance:")
        print(f"  vs NAIVE: {results['game_performance']['naive_opponent']['games_per_second']:.1f} games/sec")
        print(f"  vs OPTIMAL: {results['game_performance']['optimal_opponent']['games_per_second']:.1f} games/sec")


def main():
    """Run benchmark with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance benchmark for DQN TicTacToe v2")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to run benchmark on")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save benchmark results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark (fewer iterations)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Create and run benchmark
    benchmark = PerformanceBenchmark(device)
    
    if args.quick:
        # Quick benchmark with reduced iterations
        results = {
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
            "initialization": benchmark.benchmark_initialization(num_runs=3),
            "forward_pass": benchmark.benchmark_forward_pass(batch_sizes=[1, 32]),
            "action_selection": benchmark.benchmark_action_selection(num_runs=100),
            "training_speed": benchmark.benchmark_training_speed(num_episodes=100)
        }
    else:
        results = benchmark.run_full_benchmark()
    
    # Print and save results
    benchmark.print_summary(results)
    
    if args.save:
        benchmark.save_results(results, args.save)


if __name__ == "__main__":
    main()