# Attendance Intelligence Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade HR analytics dashboard built with **Streamlit** to monitor employee attendance, calculate productivity losses, predict lateness risks, and generate AI‑powered recommendations.

---

## 📌 Overview

The **Attendance Intelligence Dashboard** transforms raw attendance logs into actionable insights. It handles complex attendance patterns (multiple In/Out per day), computes lateness, break time, overtime, and financial impact, and provides interactive visualizations for executives, HR, and department heads. Built-in machine learning predicts which employees are likely to be late, and the Gemini API delivers tailored recommendations to improve punctuality and productivity.

---

## ✨ Features

- **Multi‑in/out handling** – correctly aggregates attendance with multiple In/Out pairs per day.
- **Comprehensive metrics** – lateness (minutes), work hours, overtime, break time, missed minutes, and financial loss (PKR).
- **Dynamic filters** – date range, shift start/end, grace minutes.
- **Rich visualizations**:
  - Executive summary (KPIs, department/factory comparisons, trends, top/bottom employees)
  - Employee profile (individual trends, detailed records)
  - Department & factory unit analysis (heatmaps, trends, cost areas)
- **Predictive analytics** – Random Forest classifier trained on past lateness to flag high‑risk employees.
- **AI‑powered recommendations** – connect to Google Gemini API to get personalized suggestions based on your data.
- **Master data editor** – edit employee details (name, department, factory, salary) directly in the app; data persists as CSV.
- **Downloadable reports** – export filtered attendance data as CSV.
- **Persistent API key** – store your Gemini key locally for convenience.

---

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/muhammadumair818/attendance-dashboard.git
   cd attendance-intelligence-dashboard
Create a virtual environment (optional but recommended)

bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install required packages

bash
pip install -r requirements.txt
If you don't have a requirements.txt, install manually:

bash
pip install streamlit pandas plotly scikit-learn requests
Run the app

bash
streamlit run app.py
🚀 Usage
1. Prepare your attendance CSV
Your CSV must contain the following columns (order doesn't matter):

Column	Description	Example
Emp_ID	Employee ID (string)	E001
Date	Date of the record (any parseable format)	1/1/2026
Action	In or Out	In
Time	Time in HH:MM (24‑hour)	9:02
Example:

csv
Emp_ID,Date,Action,Time
E001,1/1/2026,In,9:02
E001,1/1/2026,Out,12:02
E001,1/1/2026,In,16:14
E001,1/1/2026,Out,20:14
E002,1/1/2026,In,8:58
E002,1/1/2026,Out,17:24
2. Upload and process
In the sidebar, upload your CSV.

Adjust shift start/end and grace minutes (default 30 minutes).

The dashboard will automatically process the data and display all tabs.

3. Employee master data
The app creates a sample employee_master.csv on first run. You can edit it using the ✏️ Edit Master Data button in the sidebar.

The master file must contain at least: Emp_ID, Name, Department, Factory_Unit. Optionally per_min_salary (default 3.85 PKR/min).

4. AI recommendations
Enter your Google Gemini API key in the sidebar (you can save it locally).

Navigate to the 💡 Recommendations tab and click Get Recommendations. The app will analyze your data and generate insights.

📂 Data Format Requirements
Attendance file
Must have columns: Emp_ID, Date, Action, Time.

Action must be exactly In or Out.

Time should be in HH:MM format (e.g., 9:02 or 17:30).

Employee master file
The dashboard expects a CSV named employee_master.csv in the same directory.

Required columns: Emp_ID, Name, Department, Factory_Unit.

Optional: per_min_salary (salary per minute in PKR) – defaults to 3.85 if missing.

You can add extra columns; they will be preserved when editing.

⚙️ Configuration
Setting	Location	Description
Shift Start/End	Tracking tab (top)	Set the official working hours.
Grace Minutes	Tracking tab	Allowed total of late minutes + break minutes before it's counted as lost productivity.
Gemini API Key	Sidebar	Your Google AI Studio API key. Saved to api_key.txt if "Save permanently" is checked.
🧠 How It Works
Attendance aggregation

Pairs In and Out events for each employee per day.

Calculates first In, last Out, total work, total break, late minutes, overtime, etc.

Financial loss

Late_Cost = Late_Mins × per_min_salary (PKR).

Predictive model

A Random Forest classifier is trained on historical lateness values (previous day's lateness to predict next day's lateness risk).

Returns probability of being late, labeled as High, Medium, or Low.

Gemini recommendations

A prompt containing key metrics (total employees, average lateness, department summary, worst employees, weekday patterns) is sent to the Gemini API.

The response is displayed as actionable insights.

📦 Dependencies
Python 3.8+

streamlit

pandas

plotly

scikit-learn

requests

All can be installed via pip install -r requirements.txt.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📄 License
MIT

📧 Contact
For questions or support, please open an issue or contact the project maintainer.

Enjoy analyzing your attendance data like never before!
