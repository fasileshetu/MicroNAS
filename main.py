import csv
from search.astar import astar_search
from train.trainer import evaluate_architecture

def save_results(results, path='results/log.csv'):
    fieldnames = ['layers', 'activations', 'dropout_rates', 'learning_rate', 
                  'val_acc', 'train_time', 'param_count']
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            arch = r['architecture']
            writer.writerow({
                'layers': arch.hidden_layers,
                'activations': arch.activations,
                'dropout_rates': arch.dropout_rates,
                'learning_rate': arch.learning_rate,
                'val_acc': round(r['val_acc'], 4),
                'train_time': round(r['train_time'], 2),
                'param_count': r['param_count']
            })

if __name__ == '__main__':
    print("Starting MicroNAS search...")
    
    results = astar_search(
        evaluate_fn=evaluate_architecture,
        budget=10
    )
    
    save_results(results)
    
    print("\nSearch complete.")
    print(f"Best architecture found:")
    best = results[0]
    print(f"  Layers: {best['architecture'].hidden_layers}")
    print(f"  Activations: {best['architecture'].activations}")
    print(f"  Val accuracy: {best['val_acc']:.4f}")
    print(f"  Param count: {best['param_count']}")
    print(f"\nAll results saved to results/log.csv")