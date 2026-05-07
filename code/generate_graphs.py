import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def generate_presentation_graphs(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    continuous_path = os.path.join(results_dir, "continuous.csv")
    events_path = os.path.join(results_dir, "events.csv")
    
    if not os.path.exists(continuous_path) or not os.path.exists(events_path):
        print(f"Error: Could not find CSV files in {results_dir}")
        return

    df_cont = pd.read_csv(continuous_path)
    df_events = pd.read_csv(events_path)

    cols_to_fix = ['team_fill', 'enemy_fill', 'health_pct']
    df_cont[cols_to_fix] = df_cont[cols_to_fix].replace(0, float('nan')).ffill()

    # Set the style for presentation-ready charts
    sns.set_theme(style="darkgrid")
    
    # ==========================================
    # CHART 1: Match Momentum (The Story Slide)
    # ==========================================
    plt.figure(figsize=(12, 6))
    plt.plot(df_cont['timestamp'], df_cont['team_fill'] * 100, label="Team Domination %", color='dodgerblue', linewidth=2)
    plt.plot(df_cont['timestamp'], df_cont['enemy_fill'] * 100, label="Enemy Domination %", color='crimson', linewidth=2)
    
    # Overlay Kills
    team_kills = df_events[df_events['kill_type'] == 'team_kill']
    enemy_kills = df_events[df_events['kill_type'] == 'enemy_kill']
    self_kills = df_events[df_events['kill_type'] == 'self_kill']

    plt.scatter(team_kills['timestamp'], [105]*len(team_kills), color='dodgerblue', marker='v', s=100, label='Team Kill')
    plt.scatter(enemy_kills['timestamp'], [105]*len(enemy_kills), color='crimson', marker='v', s=100, label='Enemy Kill')
    plt.scatter(self_kills['timestamp'], [105]*len(self_kills), color='gold', marker='*', s=200, label='Self Kill (Tap13)')

    plt.title(f"Match Momentum & Key Events ({os.path.basename(results_dir)})", fontsize=16, fontweight='bold')
    plt.xlabel("Match Time (Seconds)", fontsize=12)
    plt.ylabel("Objective Fill %", fontsize=12)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.basename(results_dir)}_momentum.png"), dpi=300)
    plt.close()

    # ==========================================
    # CHART 2: The "Clutch" Heatmap (Event Weights)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Sort by weight so bigger dots are drawn on top
    df_events_sorted = df_events.sort_values(by='event_weight')
    
    scatter = plt.scatter(
        df_events_sorted['timestamp'], 
        df_events_sorted['health_pct_at_kill'] * 100, 
        s=(df_events_sorted['event_weight'] + 2) * 50, # Scale size by weight
        c=df_events_sorted['event_weight'], 
        cmap='viridis', 
        alpha=0.8,
        edgecolors='white'
    )
    
    plt.colorbar(scatter, label="Event Weight (Impact)")
    plt.axhline(30, color='red', linestyle='--', alpha=0.5, label='Low Health Threshold (30%)')
    
    plt.title(f"Lethality vs Survivability: The Clutch Index", fontsize=16, fontweight='bold')
    plt.xlabel("Match Time (Seconds)", fontsize=12)
    plt.ylabel("Player Health at Time of Kill (%)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.basename(results_dir)}_clutch_heatmap.png"), dpi=300)
    plt.close()

    # ==========================================
    # CHART 3: Model Extraction Validation (Hero Counts)
    # ==========================================
    plt.figure(figsize=(12, 5))
    
    # Count how many times each hero appeared as a killer or victim
    all_heroes = pd.concat([df_events['killer_hero'], df_events['victim_hero']])
    hero_counts = all_heroes.value_counts().reset_index()
    hero_counts.columns = ['Hero', 'Involvements']
    
    sns.barplot(data=hero_counts, x='Hero', y='Involvements', palette='magma')
    plt.title(f"Hero Detection Frequencies ({os.path.basename(results_dir)})", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Times Detected", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.basename(results_dir)}_hero_distribution.png"), dpi=300)
    plt.close()

    print(f"✅ Generated 3 graphs for {results_dir} in {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results", help="Folder containing the CSVs")
    parser.add_argument("--output", type=str, default="presentation_graphs", help="Where to save the PNGs")
    args = parser.parse_args()
    
    generate_presentation_graphs(args.input, args.output)