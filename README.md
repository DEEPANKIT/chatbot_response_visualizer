# Chat Data Analysis and Visualization

This script analyzes chat data from multiple JSON files and generates comprehensive visualizations and insights.

## Features

- Processes multiple JSON files containing chat data
- Generates various visualizations:
  - Token usage analysis
  - Interaction patterns
  - Content analysis

## Requirements

- Python 3.7 or higher
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Create a directory called `json_files` in the same location as the script
2. Place your JSON files in the `json_files` directory
3. Run the script:

```bash
python chat_analyzer.py
```

## Output Files

The script generates the following files:

- `common_words_analysis.png`: lists the frequency histogram of top 20 most commonly occuring words
- `multi_session_interation_frequency.png`: interaction frequency over time across different sessions
- `token_analysis.png`: Visualizations of token usage patterns
- `interaction_analysis.png`: Analysis of interaction patterns
- `content_analysis.png`: Content analysis visualizations


## JSON File Format

Your JSON files should follow this structure:

```json
{
  "chat_history": [
    {
      "user_id": "string",
      "project_id": "string",
      "session_id": "string",
      "datetime": "YYYY-MM-DD HH:MM:SS",
      "session_total_tokens": number,
      "chat_data": [
        {
          "input_prompt": "string",
          "output_response": "string",
          "timestamp": "YYYY-MM-DD HH:MM:SS",
          "input_tokens": number,
          "output_tokens": number,
          "total_tokens": number
        }
      ]
    }
  ]
}
```

## Notes

- The script will automatically process all JSON files in the `json_files` directory
- Each file should contain chat data in the specified format
- The analysis combines data from all files for comprehensive insights
