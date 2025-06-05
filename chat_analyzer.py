import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from collections import Counter
import re
import plotly.express as px
import plotly.graph_objects as go

class ChatDataAnalyzer:
    def __init__(self, data_directory="./json_files"):
        """
        Initialize the analyzer with directory containing JSON files
        """
        self.data_directory = data_directory
        self.all_data = []
        self.processed_df = None
        self.output_base_dir = "visualization_outputs"
        
    def load_json_files(self):
        """
        Load all JSON files from the specified directory
        """
        json_files = [f for f in os.listdir(self.data_directory) if f.endswith('.json')]
        
        for file_name in json_files:
            file_path = os.path.join(self.data_directory, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['source_file'] = file_name
                    self.all_data.append(data)
                print(f" Successfully loaded: {file_name}")
            except Exception as e:
                print(f" Error loading {file_name}: {e}")
        
        print(f" Total files loaded: {len(self.all_data)}")
        return self.all_data
    
    def process_data(self):
        """
        Process all loaded JSON data into a structured DataFrame
        """
        processed_rows = []
        
        for session_data in self.all_data:
            chat_history = session_data.get('chat_history', [])
            
            for chat_session in chat_history:
                user_id = chat_session.get('user_id', 'Unknown')
                project_id = chat_session.get('project_id', 'Unknown')
                session_id = chat_session.get('session_id', 'Unknown')
                session_datetime = chat_session.get('datetime', '')
                session_total_tokens = chat_session.get('session_total_tokens', 0)
                source_file = session_data.get('source_file', 'Unknown')
                
                for chat in chat_session.get('chat_data', []):
                    row = {
                        'user_id': user_id,
                        'project_id': project_id,
                        'session_id': session_id,
                        'session_datetime': session_datetime,
                        'session_total_tokens': session_total_tokens,
                        'source_file': source_file,
                        'input_prompt': chat.get('input_prompt', ''),
                        'output_response': chat.get('output_response', ''),
                        'timestamp': chat.get('timestamp', ''),
                        'input_tokens': chat.get('input_tokens', 0),
                        'output_tokens': chat.get('output_tokens', 0),
                        'total_tokens': chat.get('total_tokens', 0)
                    }
                    processed_rows.append(row)
        
        self.processed_df = pd.DataFrame(processed_rows)
        
        # here we need to Convert timestamps to datetime
        self.processed_df['timestamp'] = pd.to_datetime(self.processed_df['timestamp'])
        self.processed_df['session_datetime'] = pd.to_datetime(self.processed_df['session_datetime'])
        
        return self.processed_df

    def create_visualizations(self, source_file):
        """
        Create all visualizations for a specific JSON file
        """
        # here we need to Create file-specific directory
        file_dir = os.path.join(self.output_base_dir, os.path.splitext(source_file)[0])
        os.makedirs(file_dir, exist_ok=True)
        
        # here we need to Filter data for this file
        file_data = self.processed_df[self.processed_df['source_file'] == source_file]
        
        # 1. Multi-Session Interaction Frequency Over Time
        plt.figure(figsize=(15, 8))
        for session_id in file_data['session_id'].unique():
            session_data = file_data[file_data['session_id'] == session_id]
            hourly_interactions = session_data.groupby(session_data['timestamp'].dt.hour).size()
            plt.plot(hourly_interactions.index, hourly_interactions.values, 
                    marker='o', linestyle='-', label=f'Session {session_id}')
        
        plt.title('Interaction Frequency Over Time Across Sessions', fontsize=14, pad=20)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Number of Interactions', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(file_dir, 'multi_session_interaction_frequency.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Token Distribution Across Sessions
        plt.figure(figsize=(12, 8))
        session_tokens = file_data.groupby('session_id').agg({
            'input_tokens': 'sum',
            'output_tokens': 'sum'
        })
        
        x = np.arange(len(session_tokens))
        width = 0.35
        
        plt.bar(x - width/2, session_tokens['input_tokens'], width, label='Input Tokens', color='#3498db')
        plt.bar(x + width/2, session_tokens['output_tokens'], width, label='Output Tokens', color='#e74c3c')
        
        plt.title('Token Distribution Across Sessions', fontsize=14, pad=20)
        plt.xlabel('Session', fontsize=12)
        plt.ylabel('Total Tokens', fontsize=12)
        plt.xticks(x, [f'Session {sid}' for sid in session_tokens.index])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(file_dir, 'multi_session_token_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Most Common Words Analysis
        plt.figure(figsize=(15, 8))
        # Get all words and their frequencies
        all_text = ' '.join(file_data['input_prompt'].astype(str))
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'it', 'on', 'you', 'for', 'i', 'with', 'as', 'at', 'this', 'but', 'be', 'are'}
        for word in stop_words:
            word_freq.pop(word, None)
        
        # Get top 20 most common words
        common_words = word_freq.most_common(20)
        words, counts = zip(*common_words)
        
        # Create horizontal bar chart
        plt.barh(range(len(words)), counts, color='#3498db')
        plt.yticks(range(len(words)), words)
        plt.title('Top 20 Most Common Words Across All Sessions', fontsize=14, pad=20)
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Words', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on the bars
        for i, count in enumerate(counts):
            plt.text(count, i, f' {count}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, 'common_words_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Response Length Distribution
        plt.figure(figsize=(15, 8))
        for session_id in file_data['session_id'].unique():
            session_data = file_data[file_data['session_id'] == session_id]
            session_data['response_length'] = session_data['output_response'].str.len()
            sns.kdeplot(data=session_data, x='response_length', label=f'Session {session_id}')
        
        plt.title('Response Length Distribution Across Sessions', fontsize=14, pad=20)
        plt.xlabel('Response Length (characters)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(file_dir, 'multi_session_response_length.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Session Comparison Heatmap
        plt.figure(figsize=(15, 8))
        hourly_activity = file_data.groupby([
            file_data['timestamp'].dt.hour,
            file_data['session_id']
        ]).size().unstack(fill_value=0)
        
        sns.heatmap(hourly_activity, cmap='YlOrRd', cbar_kws={'label': 'Number of Interactions'})
        plt.title('Activity Heatmap: Hour vs Session', fontsize=14, pad=20)
        plt.xlabel('Session ID', fontsize=12)
        plt.ylabel('Hour of Day', fontsize=12)
        plt.savefig(os.path.join(file_dir, 'session_activity_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Created visualizations for {source_file} in {file_dir}")

    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "="*50)
        print("📊 CHAT DATA ANALYSIS SUMMARY REPORT")
        print("="*50)
        
        total_sessions = self.processed_df['session_id'].nunique()
        total_interactions = len(self.processed_df)
        total_tokens_used = self.processed_df['total_tokens'].sum()
        avg_tokens_per_interaction = self.processed_df['total_tokens'].mean()
        
        print(f"\n📈 Overall Statistics:")
        print(f"   • Total Sessions Analyzed: {total_sessions}")
        print(f"   • Total Interactions: {total_interactions}")
        print(f"   • Total Tokens Used: {total_tokens_used:,}")
        print(f"   • Average Tokens per Interaction: {avg_tokens_per_interaction:.2f}")
        
        print(f"\n🔍 File Breakdown:")
        for source_file in self.processed_df['source_file'].unique():
            file_data = self.processed_df[self.processed_df['source_file'] == source_file]
            print(f"\n   • File: {source_file}")
            print(f"     - Sessions: {file_data['session_id'].nunique()}")
            print(f"     - Total Interactions: {len(file_data)}")
            print(f"     - Total Tokens: {file_data['total_tokens'].sum():,}")
            print(f"     - Average Response Length: {file_data['output_response'].str.len().mean():.0f} characters")
        
        print(f"\n⏰ Time Analysis:")
        busiest_hour = self.processed_df.groupby(self.processed_df['timestamp'].dt.hour).size().idxmax()
        print(f"   • Busiest Hour: {busiest_hour}:00")
        
        print(f"\n📝 Top Topics (Most Common Words):")
        all_text = ' '.join(self.processed_df['input_prompt'].astype(str))
        words = re.findall(r'\b\w+\b', all_text.lower())
        common_words = Counter(words).most_common(5)
        for word, count in common_words:
            print(f"   • '{word}': {count} times")

def main():
    """
    Main function to run the complete analysis
    """
    # Initialize analyzer
    analyzer = ChatDataAnalyzer()
    
    # Create directory if it doesn't exist
    os.makedirs(analyzer.data_directory, exist_ok=True)
    os.makedirs(analyzer.output_base_dir, exist_ok=True)
    
    print("🤖 Chat Data Visualization Script")
    print("=" * 40)
    print(f"📂 Looking for JSON files in: {analyzer.data_directory}")
    
    # Check if directory has JSON files
    json_files = [f for f in os.listdir(analyzer.data_directory) if f.endswith('.json')]
    if not json_files:
        print(f"\n⚠️  No JSON files found in {analyzer.data_directory}")
        print("Please add your JSON files to this directory and run again.")
        return
    
    # Load and process data
    analyzer.load_json_files()
    if not analyzer.all_data:
        print("❌ No data loaded. Exiting.")
        return
        
    analyzer.process_data()
    
    # Generate visualizations for each file
    print("\n🎨 Generating visualizations...")
    for source_file in analyzer.processed_df['source_file'].unique():
        analyzer.create_visualizations(source_file)
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\n✅ Analysis complete! Check the visualization_outputs directory for results.")

if __name__ == "__main__":
    main() 