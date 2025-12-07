"""
Benchmark script to find optimal num_envs for your system.

Tests different parallel environment configurations while keeping
total_transitions constant to see which is fastest.
"""
import gymnasium as gym
import time
from stable_baselines3.common.env_util import make_vec_env

def benchmark_config(n_envs, n_steps, total_steps=50000):
    """
    Benchmark a specific configuration.

    Args:
        n_envs: Number of parallel environments
        n_steps: Steps per rollout
        total_steps: Total steps to simulate
    """
    print(f"\nTesting: {n_envs} envs × {n_steps} steps = {n_envs * n_steps} transitions/rollout")

    # Create vectorized environment
    env = make_vec_env("LunarLander-v2", n_envs=n_envs)

    # Reset environments
    obs = env.reset()

    # Time the rollout collection
    start_time = time.time()

    steps_done = 0
    episodes_completed = 0

    while steps_done < total_steps:
        # Random actions (we're just testing speed, not training)
        actions = [env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, infos = env.step(actions)

        steps_done += n_envs

        # Count completed episodes
        for done in dones:
            if done:
                episodes_completed += 1

    elapsed = time.time() - start_time
    env.close()

    # Calculate metrics
    steps_per_sec = total_steps / elapsed
    episodes_per_sec = episodes_completed / elapsed

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {steps_per_sec:.0f} steps/sec")
    print(f"  Episodes completed: {episodes_completed}")
    print(f"  Episodes/sec: {episodes_per_sec:.1f}")

    return {
        'n_envs': n_envs,
        'n_steps': n_steps,
        'time': elapsed,
        'steps_per_sec': steps_per_sec,
        'episodes': episodes_completed
    }

if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL ENVIRONMENTS BENCHMARK")
    print("=" * 60)
    print("\nGoal: Find fastest configuration while maintaining quality")
    print("Constraint: n_steps should be ≥ 250 (typical episode length)")
    print("\nAll configs collect same total transitions (16,384)")

    # Test different configurations
    # All have same total_transitions = 16,384
    configs = [
        (4, 4096),    # Very few envs, long rollouts
        (8, 2048),    # Few envs, long rollouts
        (16, 1024),   # Current setup
        (32, 512),    # More parallel
        (64, 256),    # Lots of parallel (at minimum n_steps)
        # (128, 128), # Too many - n_steps too small!
    ]

    results = []

    for n_envs, n_steps in configs:
        try:
            result = benchmark_config(n_envs, n_steps)
            results.append(result)
            time.sleep(1)  # Cool down between tests
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<20} {'Time (s)':<12} {'Steps/sec':<12} {'Speedup':<10}")
    print("-" * 60)

    baseline_time = results[0]['time'] if results else 1

    for r in results:
        speedup = baseline_time / r['time']
        config_str = f"{r['n_envs']} × {r['n_steps']}"
        print(f"{config_str:<20} {r['time']:<12.2f} {r['steps_per_sec']:<12.0f} {speedup:<10.2f}×")

    # Find best
    if results:
        best = max(results, key=lambda x: x['steps_per_sec'])
        print(f"\n🏆 WINNER: {best['n_envs']} envs × {best['n_steps']} steps")
        print(f"   Speed: {best['steps_per_sec']:.0f} steps/sec")
        print(f"\nRecommendation for Unit1.py:")
        print(f"   n_envs={best['n_envs']}, n_steps={best['n_steps']}")
