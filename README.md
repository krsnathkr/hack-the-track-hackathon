# Toyota GR Cup - Overtake AI

## 🏎️ Project Overview
This tool is an AI-powered race engineer dashboard designed for the Toyota GR Cup. It analyzes telemetry data to provide real-time overtake probability, strategic insights, and driver feedback.

Key features include:
- **Live Overtake Probability**: Calculates the % chance of a successful overtake based on speed, gap, and cornering analysis.
- **Ghost Mode**: Compare your live lap against the session's fastest lap.
- **Sector Analysis**: Breakdown of time deltas across track sectors.
- **G-Force & Speed Traces**: Visualizations for driver inputs and car performance.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hack-the-track-hackathon
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run

1. Ensure your virtual environment is activated.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. The dashboard will open automatically in your default web browser (usually at `http://localhost:8501`).

## 📂 Project Structure
- `app.py`: Main Streamlit application entry point.
- `src/overtake_model.py`: Core logic for overtake probability and race analysis.
- `src/data_processor.py`: Utilities for loading and cleaning telemetry data.
- `data/`: Directory for storing telemetry CSV files.

## 🧪 Running Tests
To run the unit tests for the overtake model:
```bash
python src/overtake_model.py
```
