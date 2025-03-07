#!/usr/bin/env python3
"""
Command-line script to analyze agent responses to scenarios.

This script runs the AgentResponseAnalyzer to analyze results from your AI governance experiment,
generating visualizations and reports to help understand how different agents approach scenarios.

Examples:
  python analyze_responses.py --results_dir results
  python analyze_responses.py --scenario trolley --report
  python analyze_responses.py --all_scenarios --output_dir analysis_output
"""

import os
import sys
import argparse
import glob
import matplotlib.pyplot as plt
from agent_analyzer import AgentResponseAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Analyze agent responses to scenarios")
    parser.add_argument("--results_dir", type=str, default="results",
                      help="Directory containing result files")
    parser.add_argument("--scenario", type=str, default=None,
                      help="Filter for a specific scenario")
    parser.add_argument("--all_scenarios", action="store_true",
                      help="Generate separate analyses for each scenario")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                      help="Directory to save analysis results")
    parser.add_argument("--report", action="store_true",
                      help="Generate a comprehensive HTML report")
    parser.add_argument("--no_plots", action="store_true",
                      help="Skip generating individual plot files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and initialize the analyzer
    analyzer = AgentResponseAnalyzer(results_dir=args.results_dir)
    
    # Load all results
    loaded_results = analyzer.load_results()
    if not loaded_results:
        print("No results found to analyze.")
        return 1
    
    # Get list of available scenarios
    scenarios = list(analyzer.scenario_results.keys())
    print(f"Found {len(scenarios)} scenarios: {', '.join(scenarios)}")
    
    if args.all_scenarios:
        # Process each scenario separately
        for scenario in scenarios:
            print(f"\nProcessing scenario: {scenario}")
            
            # Create a scenario-specific output directory
            scenario_output_dir = os.path.join(args.output_dir, scenario)
            os.makedirs(scenario_output_dir, exist_ok=True)
            
            # Run analysis for this scenario
            analyzer.run_analysis(scenario_filter=scenario)
            
            # Generate visualizations if requested
            if not args.no_plots:
                print("Generating visualizations...")
                
                # Political compass plot
                analyzer.plot_political_compass(scenario)
                plt.savefig(os.path.join(scenario_output_dir, "political_compass.png"))
                plt.close()
                
                # Alignment matrix
                analyzer.plot_alignment_matrix(scenario)
                plt.savefig(os.path.join(scenario_output_dir, "alignment_matrix.png"))
                plt.close()
                
                # Framework adherence
                analyzer.plot_framework_adherence(scenario)
                plt.savefig(os.path.join(scenario_output_dir, "framework_adherence.png"))
                plt.close()
                
                # Cross-framework influence
                analyzer.plot_cross_framework_influence(scenario)
                plt.savefig(os.path.join(scenario_output_dir, "cross_framework_influence.png"))
                plt.close()
                
                # Decision distribution
                analyzer.plot_decision_distribution(scenario)
                plt.savefig(os.path.join(scenario_output_dir, "decision_distribution.png"))
                plt.close()
            
            # Generate comprehensive report if requested
            if args.report:
                print("Generating HTML report...")
                report_file = analyzer.generate_comprehensive_report(scenario_output_dir, scenario)
                print(f"Report saved to: {report_file}")
    else:
        # Process a single scenario or all together
        specific_scenario = args.scenario
        
        # Run analysis
        analyzer.run_analysis(scenario_filter=specific_scenario)
        
        # Generate visualizations if requested
        if not args.no_plots:
            print("Generating visualizations...")
            
            # Political compass plot
            analyzer.plot_political_compass(specific_scenario)
            plt.savefig(os.path.join(args.output_dir, "political_compass.png"))
            plt.close()
            
            # Alignment matrix
            analyzer.plot_alignment_matrix(specific_scenario)
            plt.savefig(os.path.join(args.output_dir, "alignment_matrix.png"))
            plt.close()
            
            # Framework adherence
            analyzer.plot_framework_adherence(specific_scenario)
            plt.savefig(os.path.join(args.output_dir, "framework_adherence.png"))
            plt.close()
            
            # Cross-framework influence
            analyzer.plot_cross_framework_influence(specific_scenario)
            plt.savefig(os.path.join(args.output_dir, "cross_framework_influence.png"))
            plt.close()
            
            # Decision distribution
            analyzer.plot_decision_distribution(specific_scenario)
            plt.savefig(os.path.join(args.output_dir, "decision_distribution.png"))
            plt.close()
        
        # Generate comprehensive report if requested
        if args.report:
            print("Generating HTML report...")
            report_file = analyzer.generate_comprehensive_report(args.output_dir, specific_scenario)
            print(f"Report saved to: {report_file}")
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())