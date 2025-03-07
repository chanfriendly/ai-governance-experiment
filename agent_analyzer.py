#!/usr/bin/env python3
"""
Agent Response Analyzer for the AI Governance Experiment.

This module provides tools to analyze how different philosophical and governmental
frameworks approach ethical scenarios, creating visualizations and metrics to help
understand their reasoning patterns and decision-making.
"""

import os
import json
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from collections import Counter, defaultdict
from datetime import datetime

# Initialize NLTK and spaCy
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

class AgentResponseAnalyzer:
    """Analyzes responses from different agents in the AI Governance Experiment."""
    
    # Framework-specific terminology and concepts
    FRAMEWORK_TERMS = {
        "effective_altruism": [
            "utility", "welfare", "maximize", "consequences", "outcomes", "evidence", 
            "quantify", "impact", "future generations", "long-term", "expected value",
            "benefit", "cost", "efficient", "tractable", "neglected", "scale"
        ],
        "deontological": [
            "duty", "obligation", "principle", "universal law", "categorical imperative",
            "dignity", "respect", "autonomy", "rights", "intentions", "means", "ends",
            "maxim", "universalizability", "moral law", "unconditional"
        ],
        "care_ethics": [
            "care", "compassion", "relationships", "connection", "context", "needs",
            "vulnerability", "interdependence", "empathy", "nurture", "particular",
            "concrete", "responsibility", "attentiveness", "responsiveness"
        ],
        "democratic_process": [
            "participation", "vote", "consensus", "deliberation", "representation",
            "transparency", "majority", "minority rights", "discussion", "public",
            "citizen", "voice", "inclusion", "equality", "accountability", "procedural"
        ],
        "checks_and_balances": [
            "separation", "powers", "oversight", "accountability", "veto", "review",
            "balance", "distributed", "authority", "institutional", "constraint",
            "procedure", "transparency", "corruption", "conflict of interest", "judicial"
        ]
    }
    
    # Moral foundation dimensions
    MORAL_FOUNDATIONS = {
        "care_harm": ["harm", "care", "suffering", "compassion", "cruel", "hurt", "protect"],
        "fairness_cheating": ["fair", "unfair", "justice", "rights", "equity", "equal", "inequality"],
        "loyalty_betrayal": ["loyal", "betray", "solidarity", "unity", "group", "collective", "community"],
        "authority_subversion": ["authority", "obedience", "respect", "tradition", "order", "chaos", "discipline"],
        "sanctity_degradation": ["purity", "sacred", "disgust", "dignity", "integrity", "corrupt", "defile"],
        "liberty_oppression": ["liberty", "freedom", "oppression", "control", "autonomy", "choice", "coercion"]
    }
    
    # Political dimensions terms
    POLITICAL_DIMENSIONS = {
        "libertarian_authoritarian": {
            "libertarian": ["freedom", "liberty", "choice", "autonomy", "individual", "consent", "privacy"],
            "authoritarian": ["order", "control", "authority", "collective", "obedience", "security", "structure"]
        },
        "progressive_conservative": {
            "progressive": ["change", "reform", "progress", "innovation", "future", "flexibility", "adapt"],
            "conservative": ["tradition", "stability", "preserve", "caution", "tested", "heritage", "maintain"]
        }
    }
    
    # Decision patterns to look for
    DECISION_PATTERNS = {
        "pull_lever": [
            r"pull the lever",
            r"divert the trolley",
            r"switch the tracks",
            r"save the five",
            r"sacrifice the one"
        ],
        "dont_pull_lever": [
            r"not pull the lever",
            r"don't pull the lever",
            r"should not pull",
            r"allow the five",
            r"avoid taking action"
        ]
    }
    
    def __init__(self, results_dir="results"):
        """Initialize the analyzer with the directory containing result files."""
        self.results_dir = results_dir
        self.scenario_results = {}
        self.analysis_results = {}
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def load_results(self):
        """Load all results from the results directory."""
        print(f"Loading results from {self.results_dir}...")
        
        # Walk through all subdirectories in the results directory
        for root, dirs, files in os.walk(self.results_dir):
            # Skip the root directory itself
            if root == self.results_dir:
                continue
            
            scenario_dir = os.path.basename(root)
            print(f"Processing directory: {scenario_dir}")
            
            # Check if this is a valid scenario directory with results
            agent_files = [f for f in files if f.endswith('.txt') and not f.startswith('scenario')]
            json_files = [f for f in files if f.endswith('.json')]
            
            if not agent_files and not json_files:
                continue
            
            # This appears to be a valid scenario directory
            self.scenario_results[scenario_dir] = {
                "agent_responses": {},
                "metadata": {}
            }
            
            # Try to load scenario file if it exists
            scenario_file = os.path.join(root, "scenario.txt")
            if os.path.exists(scenario_file):
                with open(scenario_file, 'r', encoding='utf-8') as f:
                    self.scenario_results[scenario_dir]["scenario_text"] = f.read()
            
            # Load individual agent response files
            for agent_file in agent_files:
                agent_name = os.path.splitext(agent_file)[0]
                file_path = os.path.join(root, agent_file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Try to extract response section
                        response_section = re.search(r'RESPONSE:(.*?)$', content, re.DOTALL)
                        if response_section:
                            response_text = response_section.group(1).strip()
                        else:
                            # If there's no RESPONSE: marker, use the whole file
                            response_text = content
                        
                        self.scenario_results[scenario_dir]["agent_responses"][agent_name] = response_text
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            
            # Load JSON results file if it exists
            if json_files:
                json_path = os.path.join(root, json_files[0])
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        self.scenario_results[scenario_dir]["metadata"] = metadata
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
        
        print(f"Loaded results for {len(self.scenario_results)} scenarios")
        return len(self.scenario_results) > 0
    
    def run_analysis(self, scenario_filter=None):
        """
        Run a comprehensive analysis on all loaded results.
        
        Args:
            scenario_filter: If provided, only analyze this specific scenario
        """
        print("Running comprehensive analysis...")
        
        # Filter scenarios if requested
        scenarios = [scenario_filter] if scenario_filter else list(self.scenario_results.keys())
        if scenario_filter and scenario_filter not in self.scenario_results:
            print(f"Scenario '{scenario_filter}' not found in results.")
            return False
        
        # Initialize analysis results
        self.analysis_results = {}
        
        # Process each scenario
        for scenario in scenarios:
            self.analysis_results[scenario] = {}
            
            # Load responses for this scenario
            responses = self.scenario_results[scenario]["agent_responses"]
            
            # Skip scenarios with no responses
            if not responses:
                continue
            
            # Run various analyses
            framework_analysis = self._analyze_framework_adherence(responses)
            sentiment_analysis = self._analyze_sentiment(responses)
            decision_analysis = self._analyze_decisions(responses, scenario)
            moral_foundation_analysis = self._analyze_moral_foundations(responses)
            political_dimension_analysis = self._analyze_political_dimensions(responses)
            
            # Store analysis results
            self.analysis_results[scenario] = {
                "framework_adherence": framework_analysis["framework_adherence"],
                "cross_framework_influence": framework_analysis.get("cross_framework_influence", {}),
                "sentiment": sentiment_analysis,
                "decisions": decision_analysis,
                "moral_foundations": moral_foundation_analysis,
                "political_dimensions": political_dimension_analysis
            }
        
        return True
    
    def _analyze_framework_adherence(self, responses):
        """
        Analyze how well each agent adheres to its expected framework.
        
        Args:
            responses: Dictionary of agent responses
            
        Returns:
            Dictionary with framework adherence scores and cross-framework influence
        """
        results = {
            "framework_adherence": {},
            "cross_framework_influence": {}
        }
        
        # For each agent, calculate adherence to expected framework
        for agent, response in responses.items():
            if not response:
                continue
                
            # Determine expected framework from agent name
            expected_framework = None
            for framework in self.FRAMEWORK_TERMS.keys():
                if framework in agent:
                    expected_framework = framework
                    break
            
            # If we couldn't determine the framework, use agent name directly
            if not expected_framework:
                expected_framework = agent
            
            # Count occurrences of framework-specific terms
            framework_counts = {}
            for framework, terms in self.FRAMEWORK_TERMS.items():
                count = sum(1 for term in terms if re.search(r'\b' + re.escape(term.lower()) + r'\b', response.lower()))
                framework_counts[framework] = count
            
            # Calculate adherence as normalized counts
            total_framework_terms = sum(framework_counts.values())
            if total_framework_terms > 0:
                adherence_scores = {framework: count / total_framework_terms for framework, count in framework_counts.items()}
            else:
                adherence_scores = {framework: 0 for framework in framework_counts}
            
            # Store results
            results["framework_adherence"][agent] = adherence_scores
            
            # Calculate cross-framework influence (% of terms from other frameworks)
            if expected_framework in adherence_scores and adherence_scores[expected_framework] > 0:
                other_frameworks = {f: s for f, s in adherence_scores.items() if f != expected_framework}
                results["cross_framework_influence"][agent] = other_frameworks
            else:
                print(f"Warning: No framework terms found for agent {agent}")
                results["cross_framework_influence"][agent] = {f: 0 for f in self.FRAMEWORK_TERMS if f != expected_framework}
        
        return results
    
    def _analyze_sentiment(self, responses):
        """
        Analyze sentiment of each agent's response.
        
        Args:
            responses: Dictionary of agent responses
            
        Returns:
            Dictionary with sentiment scores
        """
        results = {}
        
        for agent, response in responses.items():
            if not response:
                continue
                
            # Get sentiment scores using VADER
            sentiment = self.sentiment_analyzer.polarity_scores(response)
            
            results[agent] = {
                "compound": sentiment["compound"],
                "positive": sentiment["pos"],
                "negative": sentiment["neg"],
                "neutral": sentiment["neu"]
            }
            
        return results
    
    def _analyze_decisions(self, responses, scenario):
        """
        Analyze decision patterns in responses based on the scenario.
        
        Args:
            responses: Dictionary of agent responses
            scenario: Name of the scenario
            
        Returns:
            Dictionary with decision analysis
        """
        results = {"raw_decisions": {}, "categories": {}}
        
        # For the trolley problem, identify if they chose to pull the lever
        if "trolley" in scenario.lower():
            for agent, response in responses.items():
                if not response:
                    continue
                    
                # Look for decision patterns
                pull_lever = any(re.search(pattern, response.lower()) for pattern in self.DECISION_PATTERNS["pull_lever"])
                dont_pull = any(re.search(pattern, response.lower()) for pattern in self.DECISION_PATTERNS["dont_pull_lever"])
                
                if pull_lever and not dont_pull:
                    decision = "pull"
                elif dont_pull and not pull_lever:
                    decision = "dont_pull"
                else:
                    # Look for more evidence if both or neither were found
                    pull_count = sum(1 for pattern in self.DECISION_PATTERNS["pull_lever"] 
                                     if re.search(pattern, response.lower()))
                    dont_count = sum(1 for pattern in self.DECISION_PATTERNS["dont_pull_lever"] 
                                     if re.search(pattern, response.lower()))
                    
                    if pull_count > dont_count:
                        decision = "pull"
                    elif dont_count > pull_count:
                        decision = "dont_pull"
                    else:
                        decision = "ambiguous"
                
                results["raw_decisions"][agent] = decision
            
            # Categorize results
            pull_count = sum(1 for d in results["raw_decisions"].values() if d == "pull")
            dont_pull_count = sum(1 for d in results["raw_decisions"].values() if d == "dont_pull")
            ambiguous_count = sum(1 for d in results["raw_decisions"].values() if d == "ambiguous")
            
            results["categories"] = {
                "pull": pull_count,
                "dont_pull": dont_pull_count,
                "ambiguous": ambiguous_count
            }
        
        return results
    
    def _analyze_moral_foundations(self, responses):
        """
        Analyze responses through the lens of moral foundations theory.
        
        Args:
            responses: Dictionary of agent responses
            
        Returns:
            Dictionary with moral foundation scores for each agent
        """
        results = {}
        
        for agent, response in responses.items():
            if not response:
                continue
                
            # Count occurrences of moral foundation terms
            foundation_scores = {}
            for foundation, terms in self.MORAL_FOUNDATIONS.items():
                count = sum(1 for term in terms if re.search(r'\b' + re.escape(term.lower()) + r'\b', response.lower()))
                foundation_scores[foundation] = count
            
            # Normalize scores
            total = sum(foundation_scores.values())
            if total > 0:
                foundation_scores = {f: s/total for f, s in foundation_scores.items()}
            
            results[agent] = foundation_scores
            
        return results
    
    def _analyze_political_dimensions(self, responses):
        """
        Analyze responses on political dimensions.
        
        Args:
            responses: Dictionary of agent responses
            
        Returns:
            Dictionary with political dimension scores
        """
        results = {}
        
        for agent, response in responses.items():
            if not response:
                continue
                
            # Calculate scores for each dimension
            dimension_scores = {}
            
            for dimension, poles in self.POLITICAL_DIMENSIONS.items():
                pole1, pole2 = list(poles.keys())
                terms1, terms2 = poles[pole1], poles[pole2]
                
                count1 = sum(1 for term in terms1 if re.search(r'\b' + re.escape(term.lower()) + r'\b', response.lower()))
                count2 = sum(1 for term in terms2 if re.search(r'\b' + re.escape(term.lower()) + r'\b', response.lower()))
                
                # Calculate a normalized position between -1 and 1
                if count1 + count2 > 0:
                    position = (count1 - count2) / (count1 + count2)
                else:
                    position = 0
                
                dimension_scores[dimension] = position
            
            results[agent] = dimension_scores
            
        return results
    
    def plot_framework_adherence(self, scenario=None):
        """
        Plot how well each agent adheres to different frameworks.
        
        Args:
            scenario: Name of specific scenario to plot (None for all)
        """
        if not self.analysis_results:
            print("No analysis results available. Run analysis first.")
            return
        
        # Get scenarios to plot
        scenarios = [scenario] if scenario else list(self.analysis_results.keys())
        if scenario and scenario not in self.analysis_results:
            print(f"Scenario '{scenario}' not found in analysis results.")
            return
        
        # For each scenario, create a grouped bar chart
        for s in scenarios:
            if "framework_adherence" not in self.analysis_results[s]:
                continue
                
            adherence = self.analysis_results[s]["framework_adherence"]
            if not adherence:
                continue
            
            # Convert data to DataFrame for easier plotting
            data = []
            for agent, scores in adherence.items():
                for framework, score in scores.items():
                    data.append({
                        "Agent": agent,
                        "Framework": framework,
                        "Adherence Score": score
                    })
            
            df = pd.DataFrame(data)
            
            # Plot
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x="Agent", y="Adherence Score", hue="Framework", data=df)
            plt.title(f"Framework Adherence by Agent: {s}", fontsize=14)
            plt.xlabel("Agent", fontsize=12)
            plt.ylabel("Adherence Score (normalized)", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Framework", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            return plt.gcf()  # Return the figure for saving
    
    def plot_political_compass(self, scenario=None):
        """
        Plot agents on a political compass based on their responses.
        
        Args:
            scenario: Name of specific scenario to plot (None for all)
        """
        if not self.analysis_results:
            print("No analysis results available. Run analysis first.")
            return
        
        # Get scenarios to plot
        scenarios = [scenario] if scenario else list(self.analysis_results.keys())
        if scenario and scenario not in self.analysis_results:
            print(f"Scenario '{scenario}' not found in analysis results.")
            return
        
        # For each scenario, create a political compass plot
        for s in scenarios:
            if "political_dimensions" not in self.analysis_results[s]:
                continue
                
            dimensions = self.analysis_results[s]["political_dimensions"]
            if not dimensions:
                continue
            
            # Extract x and y coordinates for each agent
            x_coords = {}  # Progressive (left) vs Conservative (right)
            y_coords = {}  # Libertarian (top) vs Authoritarian (bottom)
            
            for agent, scores in dimensions.items():
                if "progressive_conservative" in scores:
                    # Flip sign so progressive is negative (left), conservative is positive (right)
                    x_coords[agent] = scores["progressive_conservative"]
                
                if "libertarian_authoritarian" in scores:
                    # Negate so libertarian is positive (top), authoritarian is negative (bottom)
                    y_coords[agent] = -scores["libertarian_authoritarian"]
            
            # Create plot
            plt.figure(figsize=(10, 10))
            
            # Draw quadrant lines
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Label quadrants
            plt.text(-0.9, 0.9, "Progressive\nLibertarian", ha='center', fontsize=12)
            plt.text(0.9, 0.9, "Conservative\nLibertarian", ha='center', fontsize=12)
            plt.text(-0.9, -0.9, "Progressive\nAuthoritarian", ha='center', fontsize=12)
            plt.text(0.9, -0.9, "Conservative\nAuthoritarian", ha='center', fontsize=12)
            
            # Plot each agent
            for agent in x_coords:
                if agent in y_coords:
                    plt.scatter(x_coords[agent], y_coords[agent], s=100, label=agent)
                    plt.text(x_coords[agent]+0.05, y_coords[agent]+0.05, agent, fontsize=10)
            
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.title(f"Political Compass: {s}", fontsize=14)
            plt.xlabel("Progressive (-1) to Conservative (+1)", fontsize=12)
            plt.ylabel("Authoritarian (-1) to Libertarian (+1)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()  # Return the figure for saving
    
    def plot_cross_framework_influence(self, scenario=None):
        """
        Plot how much each agent is influenced by other frameworks.
        
        Args:
            scenario: Name of specific scenario to plot (None for all)
        """
        if not self.analysis_results:
            print("No analysis results available. Run analysis first.")
            return
        
        # Get scenarios to plot
        scenarios = [scenario] if scenario else list(self.analysis_results.keys())
        if scenario and scenario not in self.analysis_results:
            print(f"Scenario '{scenario}' not found in analysis results.")
            return
        
        # For each scenario, create a heatmap
        for s in scenarios:
            if "cross_framework_influence" not in self.analysis_results[s]:
                # Initialize cross-framework influence if it doesn't exist
                if "framework_adherence" in self.analysis_results[s]:
                    adherence = self.analysis_results[s]["framework_adherence"]
                    cross_influence = {}
                    
                    # Calculate cross-framework influence manually
                    for agent, scores in adherence.items():
                        # Determine expected framework from agent name
                        expected_framework = None
                        for framework in self.FRAMEWORK_TERMS.keys():
                            if framework in agent:
                                expected_framework = framework
                                break
                        
                        # If we couldn't determine the framework, skip
                        if not expected_framework:
                            continue
                            
                        # Calculate influence from other frameworks
                        other_frameworks = {f: s for f, s in scores.items() if f != expected_framework}
                        cross_influence[agent] = other_frameworks
                    
                    self.analysis_results[s]["cross_framework_influence"] = cross_influence
                else:
                    continue
            
            influence = self.analysis_results[s]["cross_framework_influence"]
            if not influence:
                continue
            
            # Convert data to DataFrame for heatmap
            data = []
            agents = list(influence.keys())
            frameworks = list(set(framework for agent_data in influence.values() for framework in agent_data))
            
            for agent in agents:
                for framework in frameworks:
                    score = influence[agent].get(framework, 0)
                    data.append({
                        "Agent": agent,
                        "Framework": framework,
                        "Influence Score": score
                    })
            
            df = pd.DataFrame(data)
            pivot_df = df.pivot(index="Agent", columns="Framework", values="Influence Score")
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Influence Score'})
            plt.title(f"Cross-Framework Influence: {s}", fontsize=14)
            plt.tight_layout()
            
            return plt.gcf()  # Return the figure for saving
    
    def plot_alignment_matrix(self, scenario=None):
        """
        Plot alignment between agents using cosine similarity.
        
        Args:
            scenario: Name of specific scenario to plot (None for all)
        """
        if not self.scenario_results:
            print("No scenario results available. Load results first.")
            return
        
        # Get scenarios to plot
        scenarios = [scenario] if scenario else list(self.scenario_results.keys())
        if scenario and scenario not in self.scenario_results:
            print(f"Scenario '{scenario}' not found in scenario results.")
            return
        
        # For each scenario, create an alignment matrix
        for s in scenarios:
            responses = self.scenario_results[s]["agent_responses"]
            if not responses:
                continue
            
            # Calculate TF-IDF vectors for responses
            agents = list(responses.keys())
            texts = [responses[agent] for agent in agents]
            
            # Skip if fewer than 2 responses
            if len(texts) < 2:
                continue
            
            # Calculate TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
            except:
                # If vectorization fails (e.g., empty strings), skip
                continue
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", 
                        xticklabels=agents, yticklabels=agents,
                        cbar_kws={'label': 'Cosine Similarity'})
            plt.title(f"Agent Alignment Matrix: {s}", fontsize=14)
            plt.tight_layout()
            
            return plt.gcf()  # Return the figure for saving
    
    def plot_decision_distribution(self, scenario=None):
        """
        Plot the distribution of decisions made by agents.
        
        Args:
            scenario: Name of specific scenario to plot (None for all)
        """
        if not self.analysis_results:
            print("No analysis results available. Run analysis first.")
            return
        
        # Get scenarios to plot
        scenarios = [scenario] if scenario else list(self.analysis_results.keys())
        if scenario and scenario not in self.analysis_results:
            print(f"Scenario '{scenario}' not found in analysis results.")
            return
        
        # For each scenario, create a distribution plot
        for s in scenarios:
            if "decisions" not in self.analysis_results[s] or "categories" not in self.analysis_results[s]["decisions"]:
                continue
                
            categories = self.analysis_results[s]["decisions"]["categories"]
            if not categories:
                continue
            
            # Create pie chart
            plt.figure(figsize=(10, 8))
            labels = list(categories.keys())
            values = list(categories.values())
            colors = ['#ff9999','#66b3ff','#99ff99']
            
            # Only plot non-zero values
            non_zero_labels = [labels[i] for i in range(len(values)) if values[i] > 0]
            non_zero_values = [v for v in values if v > 0]
            
            if non_zero_values:
                plt.pie(non_zero_values, labels=non_zero_labels, colors=colors[:len(non_zero_values)], 
                        autopct='%1.1f%%', startangle=90, shadow=True)
                plt.title(f"Decision Distribution: {s}", fontsize=14)
                
            return plt.gcf()  # Return the figure for saving
    
    def generate_comprehensive_report(self, output_dir, scenario=None):
        """
        Generate a comprehensive HTML report of all analyses.
        
        Args:
            output_dir: Directory to save the report
            scenario: Specific scenario to include (None for all)
            
        Returns:
            Path to the generated report file
        """
        if not self.analysis_results:
            print("No analysis results available. Run analysis first.")
            return None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get scenarios to include
        scenarios = [scenario] if scenario else list(self.analysis_results.keys())
        if scenario and scenario not in self.analysis_results:
            print(f"Scenario '{scenario}' not found in analysis results.")
            return None
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_text = scenario if scenario else "all_scenarios"
        report_filename = f"analysis_report_{scenario_text}_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)
        
        # Create figure paths
        figure_paths = {}
        for s in scenarios:
            # Generate figures and save them
            figure_paths[s] = {}
            
            # Framework adherence plot
            fig = self.plot_framework_adherence(s)
            if fig:
                path = os.path.join(output_dir, f"{s}_framework_adherence.png")
                fig.savefig(path)
                plt.close(fig)
                figure_paths[s]["framework_adherence"] = os.path.basename(path)
            
            # Political compass plot
            fig = self.plot_political_compass(s)
            if fig:
                path = os.path.join(output_dir, f"{s}_political_compass.png")
                fig.savefig(path)
                plt.close(fig)
                figure_paths[s]["political_compass"] = os.path.basename(path)
            
            # Cross-framework influence plot
            fig = self.plot_cross_framework_influence(s)
            if fig:
                path = os.path.join(output_dir, f"{s}_cross_framework_influence.png")
                fig.savefig(path)
                plt.close(fig)
                figure_paths[s]["cross_framework_influence"] = os.path.basename(path)
            
            # Alignment matrix plot
            fig = self.plot_alignment_matrix(s)
            if fig:
                path = os.path.join(output_dir, f"{s}_alignment_matrix.png")
                fig.savefig(path)
                plt.close(fig)
                figure_paths[s]["alignment_matrix"] = os.path.basename(path)
            
            # Decision distribution plot
            fig = self.plot_decision_distribution(s)
            if fig:
                path = os.path.join(output_dir, f"{s}_decision_distribution.png")
                fig.savefig(path)
                plt.close(fig)
                figure_paths[s]["decision_distribution"] = os.path.basename(path)
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Governance Experiment Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .scenario {{
                    margin-bottom: 40px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .footer {{
                    margin-top: 40px;
                    text-align: center;
                    font-size: 0.8em;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Governance Experiment Analysis Report</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """
        
        # Add scenario-specific sections
        for s in scenarios:
            html_content += f"""
                <div class="scenario">
                    <h2>Scenario: {s}</h2>
            """
            
            # Add scenario text if available
            if s in self.scenario_results and "scenario_text" in self.scenario_results[s]:
                scenario_text = self.scenario_results[s]["scenario_text"]
                html_content += f"""
                    <h3>Scenario Description</h3>
                    <div class="scenario-text">
                        <p>{scenario_text}</p>
                    </div>
                """
            
            # Add visualizations if available
            if s in figure_paths:
                html_content += f"""
                    <h3>Visualizations</h3>
                """
                
                # Framework adherence
                if "framework_adherence" in figure_paths[s]:
                    html_content += f"""
                    <div class="visualization">
                        <h4>Framework Adherence</h4>
                        <p>This chart shows how much each agent uses terminology from different ethical frameworks.</p>
                        <img src="{figure_paths[s]['framework_adherence']}" alt="Framework Adherence">
                    </div>
                    """
                
                # Political compass
                if "political_compass" in figure_paths[s]:
                    html_content += f"""
                    <div class="visualization">
                        <h4>Political Compass</h4>
                        <p>This plot shows where each agent falls on the political compass based on their response.</p>
                        <img src="{figure_paths[s]['political_compass']}" alt="Political Compass">
                    </div>
                    """
                
                # Cross-framework influence
                if "cross_framework_influence" in figure_paths[s]:
                    html_content += f"""
                    <div class="visualization">
                        <h4>Cross-Framework Influence</h4>
                        <p>This heatmap shows how much each agent is influenced by frameworks other than their own.</p>
                        <img src="{figure_paths[s]['cross_framework_influence']}" alt="Cross-Framework Influence">
                    </div>
                    """
                
                # Alignment matrix
                if "alignment_matrix" in figure_paths[s]:
                    html_content += f"""
                    <div class="visualization">
                        <h4>Agent Alignment Matrix</h4>
                        <p>This matrix shows the similarity between each agent's reasoning, based on cosine similarity.</p>
                        <img src="{figure_paths[s]['alignment_matrix']}" alt="Agent Alignment Matrix">
                    </div>
                    """
                
                # Decision distribution
                if "decision_distribution" in figure_paths[s]:
                    html_content += f"""
                    <div class="visualization">
                        <h4>Decision Distribution</h4>
                        <p>This chart shows the distribution of decisions made by the agents.</p>
                        <img src="{figure_paths[s]['decision_distribution']}" alt="Decision Distribution">
                    </div>
                    """
            
            # Add analytical findings
            html_content += f"""
                <h3>Key Findings</h3>
            """
            
            # Add sentiment analysis
            if "sentiment" in self.analysis_results[s]:
                html_content += f"""
                <h4>Sentiment Analysis</h4>
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Positive</th>
                        <th>Negative</th>
                        <th>Neutral</th>
                        <th>Compound</th>
                    </tr>
                """
                
                for agent, scores in self.analysis_results[s]["sentiment"].items():
                    html_content += f"""
                    <tr>
                        <td>{agent}</td>
                        <td>{scores["positive"]:.2f}</td>
                        <td>{scores["negative"]:.2f}</td>
                        <td>{scores["neutral"]:.2f}</td>
                        <td>{scores["compound"]:.2f}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
            
            # Add moral foundations analysis
            if "moral_foundations" in self.analysis_results[s]:
                html_content += f"""
                <h4>Moral Foundations Analysis</h4>
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Care/Harm</th>
                        <th>Fairness/Cheating</th>
                        <th>Loyalty/Betrayal</th>
                        <th>Authority/Subversion</th>
                        <th>Sanctity/Degradation</th>
                        <th>Liberty/Oppression</th>
                    </tr>
                """
                
                for agent, scores in self.analysis_results[s]["moral_foundations"].items():
                    html_content += f"""
                    <tr>
                        <td>{agent}</td>
                        <td>{scores.get("care_harm", 0):.2f}</td>
                        <td>{scores.get("fairness_cheating", 0):.2f}</td>
                        <td>{scores.get("loyalty_betrayal", 0):.2f}</td>
                        <td>{scores.get("authority_subversion", 0):.2f}</td>
                        <td>{scores.get("sanctity_degradation", 0):.2f}</td>
                        <td>{scores.get("liberty_oppression", 0):.2f}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
            
            # Add decision analysis
            if "decisions" in self.analysis_results[s] and "raw_decisions" in self.analysis_results[s]["decisions"]:
                html_content += f"""
                <h4>Decision Analysis</h4>
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Decision</th>
                    </tr>
                """
                
                for agent, decision in self.analysis_results[s]["decisions"]["raw_decisions"].items():
                    html_content += f"""
                    <tr>
                        <td>{agent}</td>
                        <td>{decision}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
            
            html_content += """
                </div>
            """
        
        # Close HTML content
        html_content += """
                <div class="footer">
                    <p>AI Governance Experiment Analysis Report</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report generated: {report_path}")
        return report_path