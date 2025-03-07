#!/usr/bin/env python3
"""
Agent Response Analyzer

This tool analyzes responses from different AI agents (based on different
philosophical and governance frameworks) to understand their reasoning patterns,
alignment, and positioning on various spectrums of thought.
"""

import os
import json
import re
import glob
import time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import nltk
import spacy
from matplotlib.colors import LinearSegmentedColormap

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class AgentResponseAnalyzer:
    """A tool for analyzing agent responses to ethical scenarios."""
    
    def __init__(self, results_dir="results"):
        """
        Initialize the analyzer with framework information and load results.
        
        Args:
            results_dir (str): Directory containing the result files
        """
        self.results_dir = results_dir
        self.frameworks = [
            "effective_altruism", 
            "deontological", 
            "care_ethics", 
            "democratic_process", 
            "checks_and_balances"
        ]
        
        # Dictionary mapping agent frameworks to their key concepts
        self.framework_keywords = {
            "effective_altruism": [
                "utility", "outcomes", "maximize", "consequentialist", 
                "evidence", "calculation", "impact", "greater good", "quantify",
                "well-being", "effectiveness", "cost-benefit", "efficiency",
                "future generations", "long-term", "scale", "tractability"
            ],
            "deontological": [
                "duty", "obligation", "rights", "universal law", "dignity", 
                "categorical", "principle", "inherent", "moral law", "duty-based",
                "autonomy", "respect", "intention", "universalizability", "moral worth",
                "imperative", "never merely means"
            ],
            "care_ethics": [
                "relationship", "care", "vulnerability", "context", 
                "connection", "responsibility", "attentiveness", "compassion",
                "nurturing", "interdependence", "emotional", "particular",
                "concrete", "needs", "empathy", "narrative", "listening"
            ],
            "democratic_process": [
                "participation", "stakeholder", "transparency", 
                "representation", "vote", "deliberation", "consent", "citizen",
                "public", "inclusion", "debate", "majority", "minority rights",
                "discussion", "discourse", "equality", "voice", "assembly"
            ],
            "checks_and_balances": [
                "oversight", "power", "distribution", "accountability", 
                "authority", "procedure", "institutional", "balance", "restraint",
                "limited", "separated", "veto", "review", "courts", "transparency",
                "corruption", "rule of law", "constitution", "judiciary"
            ]
        }
        
        # For analyzing positioning on progressive/conservative axis
        self.progressive_terms = [
            "change", "reform", "progress", "equality", "diversity", 
            "inclusion", "marginalized", "innovation", "future", "adaptation",
            "collective", "shared", "public good", "regulation", "intervention",
            "support", "aid", "assistance", "welfare", "protection", "equity"
        ]
        
        self.conservative_terms = [
            "tradition", "stability", "order", "preservation", "continuity", 
            "tested", "proven", "caution", "restraint", "individual", "private",
            "self-reliance", "autonomy", "market", "liberty", "freedom", 
            "responsibility", "virtue", "merit", "hierarchy"
        ]
        
        # For analyzing positioning on authoritarian/distributed power axis
        self.authoritarian_terms = [
            "central", "authority", "control", "enforce", "mandate", "require",
            "compliance", "obedience", "order", "command", "direct", "efficiency",
            "decisive", "strong", "firm", "leadership", "unity", "security",
            "threat", "protection", "defense", "standard", "uniform"
        ]
        
        self.distributed_power_terms = [
            "decentralized", "diverse", "voluntary", "choice", "consensus", 
            "dialogue", "participatory", "community", "local", "democratic", 
            "representation", "voice", "autonomy", "self-governance", "federation",
            "subsidiarity", "pluralism", "dissent", "deliberation", "checks"
        ]
        
        # Terms indicating decision direction
        self.decision_terms = {
            "pull_lever": ["pull the lever", "divert the trolley", "sacrifice one", "save five", "utilitarian choice"],
            "dont_pull": ["don't pull", "not pull", "refrain from", "against pulling", "not morally permissible"]
        }
        
        # Load sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Will hold all results after loading
        self.all_results = []
        self.scenario_results = {}
        
        # DataFrame to store analysis results
        self.results_df = pd.DataFrame()

    def load_results(self, scenario_filter=None):
        """
        Load all result files from the result directory.
        
        Args:
            scenario_filter (str, optional): Only load results for this scenario
        
        Returns:
            dict: A dictionary of loaded results grouped by scenario
        """
        print(f"Loading results from {self.results_dir}...")
        
        # Get all scenario directories
        scenario_dirs = glob.glob(os.path.join(self.results_dir, "*"))
        scenario_dirs = [d for d in scenario_dirs if os.path.isdir(d)]
        
        if not scenario_dirs:
            print(f"No scenario directories found in {self.results_dir}")
            return {}
        
        # Process each scenario directory
        for scenario_dir in scenario_dirs:
            scenario_name = os.path.basename(scenario_dir)
            
            # Skip if we're filtering for a specific scenario
            if scenario_filter and scenario_filter not in scenario_name.lower():
                continue
                
            # Check for JSON results file
            json_files = glob.glob(os.path.join(scenario_dir, "results.json"))
            
            if json_files:
                # We have a JSON results file
                with open(json_files[0], 'r') as f:
                    scenario_data = json.load(f)
                    
                # Extract the scenario from the results
                scenario_text = scenario_data.get("scenario_text", "")
                
                # Create an entry for this scenario
                if scenario_name not in self.scenario_results:
                    self.scenario_results[scenario_name] = {
                        "scenario_text": scenario_text,
                        "runs": []
                    }
                
                # Add this run to the scenario
                run_data = {
                    "run_id": os.path.basename(scenario_dir),
                    "agent_responses": {}
                }
                
                # Extract agent responses
                for result in scenario_data.get("results", []):
                    agent_name = result.get("agent", "")
                    response = result.get("response", "")
                    run_data["agent_responses"][agent_name] = response
                
                self.scenario_results[scenario_name]["runs"].append(run_data)
                self.all_results.append(scenario_data)
            else:
                # Individual text files for each agent
                print(f"Processing directory: {scenario_dir}")
                agent_files = glob.glob(os.path.join(scenario_dir, "*.txt"))
                
                if not agent_files:
                    print(f"No agent response files found in {scenario_dir}")
                    continue
                
                # Read the scenario.txt file if it exists
                scenario_file = os.path.join(scenario_dir, "scenario.txt")
                scenario_text = ""
                if os.path.exists(scenario_file):
                    with open(scenario_file, 'r') as f:
                        scenario_text = f.read()
                
                # Create an entry for this scenario
                if scenario_name not in self.scenario_results:
                    self.scenario_results[scenario_name] = {
                        "scenario_text": scenario_text,
                        "runs": []
                    }
                
                # Create a run for this directory
                run_data = {
                    "run_id": os.path.basename(scenario_dir),
                    "agent_responses": {}
                }
                
                # Load each agent response
                for agent_file in agent_files:
                    agent_name = os.path.splitext(os.path.basename(agent_file))[0]
                    
                    # Skip non-agent files
                    if agent_name.lower() == "scenario":
                        continue
                    
                    with open(agent_file, 'r') as f:
                        response = f.read()
                    
                    # Extract the actual response content (after the metadata)
                    response_parts = response.split("RESPONSE:")
                    if len(response_parts) > 1:
                        response = response_parts[1].strip()
                    
                    run_data["agent_responses"][agent_name] = response
                
                self.scenario_results[scenario_name]["runs"].append(run_data)
        
        print(f"Loaded results for {len(self.scenario_results)} scenarios")
        return self.scenario_results

    def extract_key_concepts(self, text):
        """
        Extract key concepts and themes from text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Dictionary with key concepts and their counts
        """
        # Process with spaCy
        doc = nlp(text)
        
        # Extract entities, noun phrases, and important keywords
        entities = [ent.text.lower() for ent in doc.ents]
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        # Extract important words (nouns, verbs, adjectives)
        important_words = [token.lemma_.lower() for token in doc 
                          if token.pos_ in ('NOUN', 'VERB', 'ADJ') 
                          and not token.is_stop
                          and len(token.text) > 1]
        
        # Count occurrences
        concept_counts = Counter(entities + noun_phrases + important_words)
        
        return dict(concept_counts.most_common(20))

    def analyze_framework_adherence(self, agent_name, text):
        """
        Analyze how well a response adheres to its framework.
        
        Args:
            agent_name (str): The name of the agent/framework
            text (str): The agent's response text
            
        Returns:
            dict: Metrics on framework adherence
        """
        # Get the keywords for this framework
        if agent_name in self.framework_keywords:
            framework_terms = self.framework_keywords[agent_name]
        else:
            # Try to match partial name
            for framework in self.framework_keywords:
                if framework in agent_name:
                    framework_terms = self.framework_keywords[framework]
                    break
            else:
                print(f"Warning: No framework terms found for agent {agent_name}")
                return {"adherence_score": 0, "keyword_matches": {}}
        
        # Count framework-specific terms
        text_lower = text.lower()
        keyword_counts = {}
        
        for term in framework_terms:
            count = text_lower.count(term)
            if count > 0:
                keyword_counts[term] = count
        
        # Calculate adherence score (percentage of framework terms used)
        total_possible = len(framework_terms)
        terms_used = len(keyword_counts)
        adherence_score = (terms_used / total_possible) * 100
        
        # Also check for cross-framework influence
        cross_framework = {}
        for framework, terms in self.framework_keywords.items():
            if framework != agent_name and framework not in agent_name:
                matches = sum(1 for term in terms if text_lower.count(term) > 0)
                if matches > 0:
                    cross_framework[framework] = (matches / len(terms)) * 100
        
        return {
            "adherence_score": adherence_score,
            "keyword_matches": keyword_counts,
            "cross_framework_influence": cross_framework
        }

    def measure_alignment(self, responses, method="tfidf"):
        """
        Measure how aligned different agent responses are.
        
        Args:
            responses (dict): Dictionary of agent responses
            method (str): Method to use for measuring alignment
                          (options: "tfidf", "count", "decision")
            
        Returns:
            dict: Alignment scores between agents
        """
        agents = list(responses.keys())
        texts = [responses[agent] for agent in agents]
        
        if not texts or not agents:
            return {}
        
        if method == "tfidf":
            # TF-IDF based similarity
            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
            except ValueError as e:
                print(f"Error calculating TF-IDF: {e}")
                return {}
                
        elif method == "count":
            # Simple count-based similarity
            vectorizer = CountVectorizer(stop_words='english')
            try:
                count_matrix = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(count_matrix)
            except ValueError as e:
                print(f"Error calculating count vectors: {e}")
                return {}
                
        elif method == "decision":
            # Decision-based alignment
            # This is a simpler approach focused just on the final decision
            decisions = {}
            for agent, text in responses.items():
                # For trolley problem
                pull_count = sum(text.lower().count(term) for term in self.decision_terms["pull_lever"])
                dont_pull_count = sum(text.lower().count(term) for term in self.decision_terms["dont_pull"])
                
                if pull_count > dont_pull_count:
                    decisions[agent] = "pull"
                elif dont_pull_count > pull_count:
                    decisions[agent] = "dont_pull"
                else:
                    decisions[agent] = "unclear"
            
            # Calculate a simple alignment matrix
            similarity_matrix = np.zeros((len(agents), len(agents)))
            for i, agent1 in enumerate(agents):
                for j, agent2 in enumerate(agents):
                    if decisions[agent1] == decisions[agent2]:
                        similarity_matrix[i, j] = 1.0
                    else:
                        similarity_matrix[i, j] = 0.0
        else:
            print(f"Unknown alignment method: {method}")
            return {}
        
        # Create a dictionary of alignment scores
        alignment_scores = {}
        for i, agent1 in enumerate(agents):
            alignment_scores[agent1] = {}
            for j, agent2 in enumerate(agents):
                if i != j:  # Skip self-comparison
                    alignment_scores[agent1][agent2] = similarity_matrix[i, j]
        
        return alignment_scores

    def analyze_political_positioning(self, text):
        """
        Analyze where a response falls on progressive/conservative and
        authoritarian/distributed power axes.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Scores on both axes
        """
        text_lower = text.lower()
        
        # Count term occurrences on both axes
        prog_count = sum(text_lower.count(term) for term in self.progressive_terms)
        cons_count = sum(text_lower.count(term) for term in self.conservative_terms)
        auth_count = sum(text_lower.count(term) for term in self.authoritarian_terms)
        dist_count = sum(text_lower.count(term) for term in self.distributed_power_terms)
        
        # Calculate the relative positions on each axis
        # Scale from -1 to 1, where:
        # -1 = conservative or authoritarian
        # 1 = progressive or distributed power
        
        # Avoid division by zero
        prog_cons_total = prog_count + cons_count
        auth_dist_total = auth_count + dist_count
        
        if prog_cons_total > 0:
            prog_cons_score = (prog_count - cons_count) / prog_cons_total
        else:
            prog_cons_score = 0
            
        if auth_dist_total > 0:
            auth_dist_score = (dist_count - auth_count) / auth_dist_total
        else:
            auth_dist_score = 0
        
        # Sentiment analysis can also inform positioning
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        return {
            "progressive_conservative": prog_cons_score,
            "authoritarian_distributed": auth_dist_score,
            "progressive_terms": prog_count,
            "conservative_terms": cons_count,
            "authoritarian_terms": auth_count,
            "distributed_terms": dist_count,
            "sentiment": sentiment
        }

    def extract_decision(self, text, scenario_type=None):
        """
        Extract the final decision or recommendation from a response.
        
        Args:
            text (str): The agent's response
            scenario_type (str, optional): Type of scenario for specialized extraction
            
        Returns:
            str: The extracted decision
        """
        # Look for common concluding phrases
        conclusion_markers = [
            "conclusion", "therefore", "thus", "in summary", "ultimately",
            "to conclude", "in conclusion", "my recommendation", "my analysis",
            "my decision", "final decision", "I would recommend", "I recommend"
        ]
        
        # Try to find a conclusion paragraph
        paragraphs = text.split('\n\n')
        
        # Check if any conclusion markers are in the last 3 paragraphs
        for paragraph in paragraphs[-3:]:
            for marker in conclusion_markers:
                if marker.lower() in paragraph.lower():
                    return paragraph.strip()
        
        # If no markers found, just take the last paragraph
        if paragraphs:
            return paragraphs[-1].strip()
        
        return "No clear decision found"

    def run_analysis(self, scenario_filter=None):
        """
        Run comprehensive analysis on all loaded results.
        
        Args:
            scenario_filter (str, optional): Only analyze results for this scenario
            
        Returns:
            pandas.DataFrame: DataFrame with analysis results
        """
        print("Running comprehensive analysis...")
        
        # Load results if not already loaded
        if not self.scenario_results:
            self.load_results(scenario_filter)
        
        if not self.scenario_results:
            print("No results to analyze")
            return pd.DataFrame()
        
        # Prepare data structure for analysis results
        analysis_results = []
        
        # Analyze each scenario and run
        for scenario_name, scenario_data in self.scenario_results.items():
            scenario_text = scenario_data.get("scenario_text", "")
            
            for run in scenario_data.get("runs", []):
                run_id = run.get("run_id", "")
                agent_responses = run.get("agent_responses", {})
                
                # Calculate alignment between agents
                alignment_scores = self.measure_alignment(agent_responses, method="tfidf")
                decision_alignment = self.measure_alignment(agent_responses, method="decision")
                
                # Analyze each agent's response
                for agent_name, response_text in agent_responses.items():
                    # Skip non-agent responses
                    if agent_name.lower() == "scenario":
                        continue
                        
                    # Extract key concepts
                    key_concepts = self.extract_key_concepts(response_text)
                    
                    # Analyze framework adherence
                    framework_analysis = self.analyze_framework_adherence(agent_name, response_text)
                    
                    # Analyze political positioning
                    positioning = self.analyze_political_positioning(response_text)
                    
                    # Extract decision
                    decision = self.extract_decision(response_text)
                    
                    # Add result to analysis results
                    result = {
                        "scenario": scenario_name,
                        "run_id": run_id,
                        "agent": agent_name,
                        "prog_cons_score": positioning["progressive_conservative"],
                        "auth_dist_score": positioning["authoritarian_distributed"],
                        "framework_adherence": framework_analysis["adherence_score"],
                        "decision": decision,
                        "response_text": response_text,
                        "key_concepts": key_concepts,
                        "framework_matches": framework_analysis["keyword_matches"],
                        "cross_framework": framework_analysis["cross_framework_influence"],
                        "sentiment_compound": positioning["sentiment"]["compound"]
                    }
                    
                    # Add alignment scores
                    if agent_name in alignment_scores:
                        for other_agent, score in alignment_scores[agent_name].items():
                            result[f"align_{other_agent}"] = score
                    
                    # Add decision alignment
                    if agent_name in decision_alignment:
                        for other_agent, score in decision_alignment[agent_name].items():
                            result[f"decision_align_{other_agent}"] = score
                    
                    analysis_results.append(result)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(analysis_results)
        print(f"Analysis complete: {len(self.results_df)} agent responses analyzed")
        
        return self.results_df

    def plot_alignment_matrix(self, scenario=None, method="tfidf"):
        """
        Plot a matrix showing alignment between different agents.
        
        Args:
            scenario (str, optional): Focus on a specific scenario
            method (str): Method to use for measuring alignment
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.results_df.empty:
            print("No analysis results available. Run analysis first.")
            return None
        
        # Filter for the specified scenario if provided
        if scenario:
            scenario_df = self.results_df[self.results_df['scenario'].str.contains(scenario, case=False)]
            if scenario_df.empty:
                print(f"No data for scenario: {scenario}")
                return None
        else:
            # Use all data but take the first run of each scenario
            scenario_df = self.results_df.drop_duplicates(subset=['scenario', 'agent'])
        
        # Get unique agents
        agents = scenario_df['agent'].unique()
        
        # Create empty alignment matrix
        alignment_matrix = np.zeros((len(agents), len(agents)))
        
        # Fill the matrix
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i == j:
                    # Perfect alignment with self
                    alignment_matrix[i, j] = 1.0
                else:
                    # Get the alignment score column
                    align_col = f"align_{agent2}"
                    if align_col in scenario_df.columns:
                        # Get records for agent1
                        agent1_records = scenario_df[scenario_df['agent'] == agent1]
                        if not agent1_records.empty and align_col in agent1_records:
                            alignment_matrix[i, j] = agent1_records[align_col].mean()
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            alignment_matrix, 
            annot=True, 
            xticklabels=agents,
            yticklabels=agents,
            cmap="YlGnBu",
            vmin=0,
            vmax=1
        )
        
        # Set title and labels
        title = f"Agent Alignment Matrix ({method.upper()} similarity)"
        if scenario:
            title += f" - {scenario}"
        plt.title(title, fontsize=15)
        plt.xlabel("Agent", fontsize=12)
        plt.ylabel("Agent", fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()

    def plot_political_compass(self, scenario=None):
        """
        Plot agents on a political compass based on their responses.
        
        Args:
            scenario (str, optional): Focus on a specific scenario
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.results_df.empty:
            print("No analysis results available. Run analysis first.")
            return None
        
        # Filter for the specified scenario if provided
        if scenario:
            scenario_df = self.results_df[self.results_df['scenario'].str.contains(scenario, case=False)]
            if scenario_df.empty:
                print(f"No data for scenario: {scenario}")
                return None
        else:
            # Use all data
            scenario_df = self.results_df
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Create a custom colormap for different agents
        agent_colors = {
            "effective_altruism": "#FF5733",  # Orange
            "deontological": "#33FF57",      # Green
            "care_ethics": "#3357FF",        # Blue
            "democratic_process": "#FF33A8",  # Pink
            "checks_and_balances": "#33A8FF"  # Light Blue
        }
        
        # Map for short names on the plot
        agent_short_names = {
            "effective_altruism": "Effective Altruism",
            "deontological": "Deontological",
            "care_ethics": "Care Ethics",
            "democratic_process": "Democratic",
            "checks_and_balances": "Checks & Balances"
        }
        
        # Create scatter plot
        # First clean up agent names to group them properly

        agent_groups = {}
        for agent in scenario_df['agent'].unique():
            base_name = agent.replace('_response', '')
            if base_name not in agent_groups:
                agent_groups[base_name] = []
            agent_groups[base_name].append(agent)

        # Now plot each base agent type with a single legend entry
        for base_agent, agent_list in agent_groups.items():
            # Get the color for this agent
            color = agent_colors.get(base_agent, "#888888")  # Default gray
            
            # Get the short name for this agent
            short_name = agent_short_names.get(base_agent, base_agent)
            
            # Plot the first one with a label (for legend)
            first_agent = True
            for agent in agent_list:
                agent_data = scenario_df[scenario_df['agent'] == agent]
                plt.scatter(
                    agent_data['prog_cons_score'], 
                    agent_data['auth_dist_score'],
                    color=color,
                    s=100,
                    alpha=0.7,
                    label=short_name if first_agent else None  # Only add to legend once
                )
                first_agent = False
        
        # Add quadrant labels
        plt.text(0.85, 0.85, "Progressive\nDistributed Power", 
                 ha='center', va='center', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.5))
        
        plt.text(-0.85, 0.85, "Conservative\nDistributed Power", 
                 ha='center', va='center', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5))
        
        plt.text(0.85, -0.85, "Progressive\nAuthoritarian", 
                 ha='center', va='center', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5))
        
        plt.text(-0.85, -0.85, "Conservative\nAuthoritarian", 
                 ha='center', va='center', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5))
        
        # Add grid lines
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set limits, labels, and title
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.xlabel("Progressive (1.0) vs Conservative (-1.0)", fontsize=14)
        plt.ylabel("Distributed Power (1.0) vs Authoritarian (-1.0)", fontsize=14)
        
        title = "Political Compass of Agent Responses"
        if scenario:
            title += f" - {scenario}"
        plt.title(title, fontsize=16)
        
        # Add legend
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  ncol=3, fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()

    def plot_framework_adherence(self, scenario=None):
        """
        Plot how well each agent adheres to its framework.
        
        Args:
            scenario (str, optional): Focus on a specific scenario
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.results_df.empty:
            print("No analysis results available. Run analysis first.")
            return None
        
        # Filter for the specified scenario if provided
        if scenario:
            scenario_df = self.results_df[self.results_df['scenario'].str.contains(scenario, case=False)]
            if scenario_df.empty:
                print(f"No data for scenario: {scenario}")
                return None
        else:
            # Use all data
            scenario_df = self.results_df
        
        # Group by agent and calculate mean adherence
        adherence_data = scenario_df.groupby('agent')['framework_adherence'].mean().reset_index()
        
        # Create the bar chart
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='agent', y='framework_adherence', data=adherence_data)
        
        # Add value labels on top of bars
        for i, v in enumerate(adherence_data['framework_adherence']):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
        
        # Set title and labels
        title = "Framework Adherence by Agent"
        if scenario:
            title += f" - {scenario}"
        plt.title(title, fontsize=15)
        plt.xlabel("Agent", fontsize=12)
        plt.ylabel("Framework Adherence Score (%)", fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)  # Set y-axis from 0 to 100%
        plt.tight_layout()
        
        return plt.gcf()

    def plot_cross_framework_influence(self, scenario=None):
        """
        Plot how much each framework influences others.
        
        Args:
            scenario (str, optional): Focus on a specific scenario
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.results_df.empty:
            print("No analysis results available. Run analysis first.")
            return None
        
        # Filter for the specified scenario if provided
        if scenario:
            scenario_df = self.results_df[self.results_df['scenario'].str.contains(scenario, case=False)]
            if scenario_df.empty:
                print(f"No data for scenario: {scenario}")
                return None
        else:
            # Use all data
            scenario_df = self.results_df
        
        # Create an influence matrix
        agents = scenario_df['agent'].unique()
        influence_matrix = np.zeros((len(agents), len(agents)))
        
        # Fill the matrix with cross-framework influence scores
        for i, agent in enumerate(agents):
            agent_data = scenario_df[scenario_df['agent'] == agent]
            
            # Get all cross-framework influences
            for row_idx, row in agent_data.iterrows():
                cross_framework = row.get('cross_framework', {})
                if isinstance(cross_framework, dict):
                    for influenced_by, score in cross_framework.items():
                        # Find the index of the influencing framework
                        for j, potential_influencer in enumerate(agents):
                            if influenced_by in potential_influencer:
                                influence_matrix[i, j] += score / len(agent_data)
                                break
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            influence_matrix, 
            annot=True, 
            xticklabels=agents,
            yticklabels=agents,
            cmap="YlOrRd", 
            vmin=0
        )
        
        # Set title and labels
        title = "Cross-Framework Influence"
        if scenario:
            title += f" - {scenario}"
        plt.title(title, fontsize=15)
        plt.xlabel("Influencing Framework", fontsize=12)
        plt.ylabel("Influenced Framework", fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()

    def plot_decision_distribution(self, scenario=None):
        """
        Plot the distribution of decisions across agents.
        
        Args:
            scenario (str, optional): Focus on a specific scenario
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.results_df.empty:
            print("No analysis results available. Run analysis first.")
            return None
        
        # Filter for the specified scenario if provided
        if scenario:
            scenario_df = self.results_df[self.results_df['scenario'].str.contains(scenario, case=False)]
            if scenario_df.empty:
                print(f"No data for scenario: {scenario}")
                return None
        else:
            # Use all data but filter to a single scenario type
            # (decisions aren't comparable across different scenario types)
            scenarios = self.results_df['scenario'].unique()
            if len(scenarios) > 0:
                scenario_df = self.results_df[self.results_df['scenario'] == scenarios[0]]
            else:
                scenario_df = self.results_df
        
        # Try to categorize decisions
        # This is just a simple example for the trolley problem
        decision_categories = []
        
        for idx, row in scenario_df.iterrows():
            decision = row['decision'].lower()
            
            # Check for trolley-specific decisions
            pull_count = sum(decision.count(term) for term in self.decision_terms["pull_lever"])
            dont_pull_count = sum(decision.count(term) for term in self.decision_terms["dont_pull"])
            
            if pull_count > dont_pull_count:
                decision_categories.append("Pull the lever")
            elif dont_pull_count > pull_count:
                decision_categories.append("Don't pull the lever")
            else:
                decision_categories.append("Unclear/Other")
        
        # Add the categorization back to the dataframe
        scenario_df = scenario_df.copy()
        scenario_df['decision_category'] = decision_categories
        
        # Create the count plot
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='agent', hue='decision_category', data=scenario_df)
        
        # Set title and labels
        title = "Decision Distribution by Agent"
        if scenario:
            title += f" - {scenario}"
        plt.title(title, fontsize=15)
        plt.xlabel("Agent", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Decision")
        plt.tight_layout()
        
        return plt.gcf()

    def generate_comprehensive_report(self, output_dir="analysis_results", scenario=None):
        """
        Generate a comprehensive analysis report with all visualizations.
        
        Args:
            output_dir (str): Directory to save the report files
            scenario (str, optional): Focus on a specific scenario
            
        Returns:
            str: Path to the generated report
        """
        if self.results_df.empty:
            print("No analysis results available. Run analysis first.")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate an appropriate filename
        if scenario:
            report_file = os.path.join(output_dir, f"analysis_report_{scenario}_{int(time.time())}.html")
        else:
            report_file = os.path.join(output_dir, f"analysis_report_{int(time.time())}.html")
        
        # Start building the HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                .figure {{ margin: 20px 0; text-align: center; }}
                .figure img {{ max-width: 100%; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                th {{ background-color: #333366; color: white; }}
            </style>
        </head>
        <body>
            <h1>Agent Analysis Report</h1>
        """
        
        # Add report generation time
        import datetime
        html_content += f"<p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        
        # Add scenario information if applicable
        if scenario:
            html_content += f"<h2>Scenario: {scenario}</h2>"
            
            # Try to find the scenario text
            scenario_df = self.results_df[self.results_df['scenario'].str.contains(scenario, case=False)]
            if not scenario_df.empty:
                scenario_text = self.scenario_results.get(scenario_df['scenario'].iloc[0], {}).get('scenario_text', '')
                if scenario_text:
                    html_content += f"<h3>Scenario Text:</h3><p>{scenario_text}</p>"
        
        # Generate and save visualizations
        
        # 1. Political Compass
        compass_fig = self.plot_political_compass(scenario)
        if compass_fig:
            compass_file = os.path.join(output_dir, "political_compass.png")
            compass_fig.savefig(compass_file)
            html_content += f"""
            <h2>Political Compass Analysis</h2>
            <div class="figure">
                <img src="{os.path.basename(compass_file)}" alt="Political Compass">
                <p>This chart shows where each agent's response falls on the progressive/conservative
                and authoritarian/distributed power axes.</p>
            </div>
            """
            plt.close(compass_fig)
        
        # 2. Alignment Matrix
        alignment_fig = self.plot_alignment_matrix(scenario)
        if alignment_fig:
            alignment_file = os.path.join(output_dir, "alignment_matrix.png")
            alignment_fig.savefig(alignment_file)
            html_content += f"""
            <h2>Agent Alignment Analysis</h2>
            <div class="figure">
                <img src="{os.path.basename(alignment_file)}" alt="Alignment Matrix">
                <p>This heatmap shows how closely aligned different agents' responses are.
                Higher values (darker colors) indicate greater similarity between responses.</p>
            </div>
            """
            plt.close(alignment_fig)
        
        # 3. Framework Adherence
        adherence_fig = self.plot_framework_adherence(scenario)
        if adherence_fig:
            adherence_file = os.path.join(output_dir, "framework_adherence.png")
            adherence_fig.savefig(adherence_file)
            html_content += f"""
            <h2>Framework Adherence Analysis</h2>
            <div class="figure">
                <img src="{os.path.basename(adherence_file)}" alt="Framework Adherence">
                <p>This chart shows how well each agent adheres to its assigned framework.
                Higher percentages indicate stronger adherence to framework-specific reasoning.</p>
            </div>
            """
            plt.close(adherence_fig)
        
        # 4. Cross-Framework Influence
        influence_fig = self.plot_cross_framework_influence(scenario)
        if influence_fig:
            influence_file = os.path.join(output_dir, "cross_framework_influence.png")
            influence_fig.savefig(influence_file)
            html_content += f"""
            <h2>Cross-Framework Influence Analysis</h2>
            <div class="figure">
                <img src="{os.path.basename(influence_file)}" alt="Cross-Framework Influence">
                <p>This heatmap shows how much each framework influences others.
                Higher values indicate stronger influence.</p>
            </div>
            """
            plt.close(influence_fig)
        
        # 5. Decision Distribution
        decision_fig = self.plot_decision_distribution(scenario)
        if decision_fig:
            decision_file = os.path.join(output_dir, "decision_distribution.png")
            decision_fig.savefig(decision_file)
            html_content += f"""
            <h2>Decision Distribution Analysis</h2>
            <div class="figure">
                <img src="{os.path.basename(decision_file)}" alt="Decision Distribution">
                <p>This chart shows the distribution of decisions across different agents.</p>
            </div>
            """
            plt.close(decision_fig)
        
        # Add key metrics table
        html_content += """
        <h2>Key Metrics Summary</h2>
        <table>
            <tr>
                <th>Agent</th>
                <th>Progressive/Conservative Score</th>
                <th>Authoritarian/Distributed Score</th>
                <th>Framework Adherence</th>
                <th>Sentiment</th>
            </tr>
        """
        
        # Filter for the specified scenario if provided
        summary_df = self.results_df
        if scenario:
            summary_df = summary_df[summary_df['scenario'].str.contains(scenario, case=False)]
        
        # Group by agent and calculate means
        summary_df = summary_df.groupby('agent').agg({
            'prog_cons_score': 'mean',
            'auth_dist_score': 'mean',
            'framework_adherence': 'mean',
            'sentiment_compound': 'mean'
        }).reset_index()
        
        # Add rows to the table
        for _, row in summary_df.iterrows():
            html_content += f"""
            <tr>
                <td>{row['agent']}</td>
                <td>{row['prog_cons_score']:.3f}</td>
                <td>{row['auth_dist_score']:.3f}</td>
                <td>{row['framework_adherence']:.1f}%</td>
                <td>{row['sentiment_compound']:.3f}</td>
            </tr>
            """
        
        html_content += "</table>"
        
       # Add key concepts in a two-column layout
        html_content += "<h2>Key Concepts by Agent</h2>"

        # Group agents by base name (without _response)
        agent_pairs = {}
        for agent in summary_df['agent']:
            if '_response' in agent:
                continue  # Skip response agents, we'll handle them with their base pair
            
            base_name = agent
            response_name = f"{agent}_response"
            
            # Only add pairs where both exist
            if response_name in summary_df['agent'].values:
                agent_pairs[base_name] = response_name

        # Create a table for side-by-side display
        html_content += """
        <style>
        .concepts-table {
            width: 100%;
            border-collapse: collapse;
        }
        .concepts-table td {
            vertical-align: top;
            width: 50%;
            padding: 10px;
        }
        .concepts-table h3 {
            margin-top: 0;
        }
        </style>
        <table class="concepts-table">
        """

        # Process each agent pair
        for base_agent, response_agent in agent_pairs.items():
            html_content += "<tr><td>"  # Left column
            
            # Base agent concepts
            html_content += f"<h3>{base_agent}</h3>"
            
            # Filter for this agent
            agent_df = self.results_df[self.results_df['agent'] == base_agent]
            if scenario:
                agent_df = agent_df[agent_df['scenario'].str.contains(scenario, case=False)]
            
            # Collect all key concepts
            base_concepts = {}
            for _, row in agent_df.iterrows():
                concepts = row.get('key_concepts', {})
                if isinstance(concepts, dict):
                    for concept, count in concepts.items():
                        if concept in base_concepts:
                            base_concepts[concept] += count
                        else:
                            base_concepts[concept] = count
            
            # Sort and display top 10
            sorted_concepts = sorted(base_concepts.items(), key=lambda x: x[1], reverse=True)[:10]
            html_content += "<ul>"
            for concept, count in sorted_concepts:
                html_content += f"<li>{concept}: {count}</li>"
            html_content += "</ul>"
            
            html_content += "</td><td>"  # Right column
            
            # Response agent concepts
            html_content += f"<h3>{response_agent}</h3>"
            
            # Filter for response agent
            resp_df = self.results_df[self.results_df['agent'] == response_agent]
            if scenario:
                resp_df = resp_df[resp_df['scenario'].str.contains(scenario, case=False)]
            
            # Collect response concepts
            resp_concepts = {}
            for _, row in resp_df.iterrows():
                concepts = row.get('key_concepts', {})
                if isinstance(concepts, dict):
                    for concept, count in concepts.items():
                        if concept in resp_concepts:
                            resp_concepts[concept] += count
                        else:
                            resp_concepts[concept] = count
            
            # Sort and display top 10
            sorted_concepts = sorted(resp_concepts.items(), key=lambda x: x[1], reverse=True)[:10]
            html_content += "<ul>"
            for concept, count in sorted_concepts:
                html_content += f"<li>{concept}: {count}</li>"
            html_content += "</ul>"
            
            html_content += "</td></tr>"  # End row

        # Handle any unpaired agents
        for agent in summary_df['agent']:
            if '_response' in agent:
                base_name = agent.replace('_response', '')
                if base_name not in summary_df['agent'].values:
                    # This is a response agent without a base agent
                    html_content += f"<tr><td></td><td>"  # Empty left column
                    html_content += f"<h3>{agent}</h3>"
                    # Similar concept collection as above
                    html_content += "</td></tr>"
            elif f"{agent}_response" not in summary_df['agent'].values:
                # This is a base agent without a response agent
                html_content += f"<tr><td>"
                html_content += f"<h3>{agent}</h3>"
                # Similar concept collection as above
                html_content += "</td><td></td></tr>"  # Empty right column

        html_content += "</table>"  # Close the table
        
        # Finish HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write the HTML file
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive report saved to: {report_file}")
        return report_file

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze agent responses to scenarios")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing result files")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Filter for a specific scenario")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                        help="Directory to save analysis results")
    parser.add_argument("--report", action="store_true",
                        help="Generate a comprehensive HTML report")
    
    args = parser.parse_args()
    
    # Create and run the analyzer
    analyzer = AgentResponseAnalyzer(results_dir=args.results_dir)
    analyzer.load_results(scenario_filter=args.scenario)
    analyzer.run_analysis(scenario_filter=args.scenario)
    
    # Generate visualizations
    analyzer.plot_political_compass(args.scenario)
    plt.savefig(os.path.join(args.output_dir, "political_compass.png"))
    plt.close()
    
    analyzer.plot_alignment_matrix(args.scenario)
    plt.savefig(os.path.join(args.output_dir, "alignment_matrix.png"))
    plt.close()
    
    analyzer.plot_framework_adherence(args.scenario)
    plt.savefig(os.path.join(args.output_dir, "framework_adherence.png"))
    plt.close()
    
    # Generate report if requested
    if args.report:
        analyzer.generate_comprehensive_report(args.output_dir, args.scenario)