#!/usr/bin/env python3
"""
Script to fix plot saving in bayes_ordinal package.
"""

import re

def fix_plot_saving():
    """Replace plt.show() calls with show_and_save() calls in plotting functions."""
    
    # Files to fix
    files_to_fix = [
        "../bayes_ordinal/workflow/diagnostics.py",
        "../bayes_ordinal/workflow/sensitivity.py", 
        "../bayes_ordinal/workflow/cross_validation.py",
        "../bayes_ordinal/plotting.py"
    ]
    
    # Plot function names for better filenames
    plot_names = {
        "diagnostics.py": [
            "diagnostics_summary",
            "diagnostics_trace", 
            "diagnostics_energy",
            "diagnostics_pair",
            "diagnostics_loo_pit"
        ],
        "sensitivity.py": [
            "prior_sensitivity",
            "influence_diagnostics",
            "sensitivity_analysis"
        ],
        "cross_validation.py": [
            "model_comparison_stacking"
        ],
        "plotting.py": [
            "contrast_analysis",
            "model_structure",
            "causal_graph",
            "category_probabilities", 
            "model_comparison",
            "model_comparison_interpretation",
            "prior_posterior"
        ]
    }
    
    for filepath in files_to_fix:
        print(f"Fixing {filepath}...")
        
        # Read file
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Get plot names for this file
        filename = filepath.split('/')[-1]
        names = plot_names.get(filename, ["plot"] * 10)  # Default fallback
        
        # Find all plt.show() calls
        pattern = r'plt\.tight_layout\(\)\s*\n\s*plt\.show\(\)'
        matches = list(re.finditer(pattern, content))
        
        print(f"  Found {len(matches)} plt.show() calls")
        
        # Replace them in reverse order to maintain positions
        for i, match in enumerate(reversed(matches)):
            if i < len(names):
                plot_name = names[i]
            else:
                plot_name = f"plot_{i+1}"
            
            replacement = f'show_and_save("{plot_name}")'
            content = content[:match.start()] + replacement + content[match.end():]
        
        # Write back
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"  Fixed {len(matches)} calls")

if __name__ == "__main__":
    fix_plot_saving() 