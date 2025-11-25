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

## 📱 Running on Another Device

### Accessing via Local Network (LAN)
To access the app from another device (e.g., phone, tablet, or laptop) on the same Wi-Fi network:

1. Run the app with the following command to allow external access:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```

2. Find your computer's local IP address:
   - **Mac/Linux**: Open Terminal and run `ifconfig` (look for `en0` or `wlan0` -> `inet`).
   - **Windows**: Open Command Prompt and run `ipconfig` (look for `IPv4 Address`).

3. On your other device, open a browser and visit:
   ```
   http://<YOUR_IP_ADDRESS>:8501
   ```
   (Replace `<YOUR_IP_ADDRESS>` with the actual IP, e.g., `192.168.1.15`)

### Setting up on a New Machine
If you want to run the code entirely on a different computer, follow the [Installation](#-installation) steps above to clone the repo and install dependencies on that machine.

## 💾 Data Setup
To keep the repository light, we only include a sample data file. To run the full simulation with all tracks (Barber, COTA, Indianapolis, Road America), you need to download the full telemetry dataset.

1. **Download the Data**

2. **Unzip the File**:
   Extract the contents of the downloaded zip file.

3. **Place in Data Directory**:
   Move the extracted folders into the `data/` directory of this project. Your folder structure should look like this:
   ```
   hack-the-track-hackathon/
   ├── data/
   │   ├── barber/
   │   ├── COTA/
   │   ├── indianapolis/
   │   ├── road-america/
   │   └── ...
   ```

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
