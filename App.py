import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import time
import requests
from datetime import datetime, time as dt_time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =============================================================================
# PAGE CONFIGURATION & CUSTOM CSS
# =============================================================================
st.set_page_config(page_title="Attendance Intelligence Dashboard", layout="wide")

st.markdown("""
<style>
    .kpi-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-title {
        font-size: 14px;
        color: #B0B0B0;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #B0B0B0;
    }
    .green { color: #2ECC71; }
    .yellow { color: #F1C40F; }
    .red { color: #E74C3C; }
    .blue { color: #4A90E2; }
    .orange { color: #FF9800; }

    /* Larger tab headings */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_PER_MIN_SALARY = 3.85
WORKING_HOURS_PER_DAY = 8

COLOR_PALETTE = {
    'primary': '#4A90E2',
    'success': '#2ECC71',
    'warning': '#F1C40F',
    'danger': '#E74C3C',
    'info': '#00BCD4',
    'neutral': '#9E9E9E'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_kpi_card(col, title, value, subtext, color_class, icon="📊"):
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{icon} {title}</div>
        <div class="kpi-value {color_class}">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

def time_to_minutes(t):
    """Convert HH:MM or H:MM to minutes since midnight."""
    if pd.isna(t) or t == '':
        return None
    try:
        parts = str(t).split(':')
        if len(parts) == 2:
            h, m = int(parts[0]), int(parts[1])
            if 0 <= h <= 23 and 0 <= m <= 59:
                return h * 60 + m
    except:
        pass
    return None

# =============================================================================
# DATA LOADING (WITH PERSISTENCE)
# =============================================================================
def load_employee_master():
    """Load master data from CSV or create sample if not exists."""
    master_file = 'employee_master.csv'
    if os.path.exists(master_file):
        try:
            df = pd.read_csv(master_file)
            df.columns = df.columns.str.strip()
            # Rename columns
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'emp' in col_lower and ('id' in col_lower or 'code' in col_lower):
                    rename_map[col] = 'Emp_ID'
                elif col_lower == 'name':
                    rename_map[col] = 'Name'
                elif col_lower in ['dept', 'department', 'division']:
                    rename_map[col] = 'Department'
                elif 'factory' in col_lower or 'unit' in col_lower:
                    rename_map[col] = 'Factory_Unit'
                elif 'category' in col_lower:
                    rename_map[col] = 'Category'
                elif 'level' in col_lower:
                    rename_map[col] = 'Level'
                elif 'per_min_salary' in col_lower or 'salary_per_min' in col_lower:
                    rename_map[col] = 'per_min_salary'
            df = df.rename(columns=rename_map)
            required = ['Emp_ID', 'Name', 'Department', 'Factory_Unit']
            for col in required:
                if col not in df.columns:
                    st.error(f"Missing required column: {col}")
                    return get_sample_master()
                df[col] = df[col].astype(str).str.strip()
            # Ensure per_min_salary column exists
            if 'per_min_salary' not in df.columns:
                df['per_min_salary'] = DEFAULT_PER_MIN_SALARY
            else:
                df['per_min_salary'] = pd.to_numeric(df['per_min_salary'], errors='coerce').fillna(DEFAULT_PER_MIN_SALARY)
            # Drop any duplicate salary column if it exists
            if 'per min salary' in df.columns:
                df = df.drop(columns=['per min salary'])
            df = df.drop_duplicates(subset=['Emp_ID'], keep='first')
            return df
        except Exception as e:
            st.error(f"Error reading master file: {e}")
            return get_sample_master()
    else:
        st.info("employee_master.csv not found. Creating sample data.")
        df = get_sample_master()
        df.to_csv(master_file, index=False)
        st.success(f"Created {master_file} with sample data.")
        return df

def get_sample_master():
    sample = {
        "Emp_ID": ["E001", "E002", "E003", "E004", "E005", "E006", "E007", "E008", "E009", "E010"],
        "Name": ["Ahmed Khan", "Fatima Ali", "Sara Ahmed", "Bilal Hussain", "Zara Tariq",
                 "Usman Malik", "Ayesha Siddiqui", "Hamza Ali", "Sana Khan", "Omar Farooq"],
        "Department": ["HR", "Finance", "IT", "Production", "Sales", "IT", "HR", "Production", "Finance", "Sales"],
        "Factory_Unit": ["Karachi", "Lahore", "Karachi", "Faisalabad", "Lahore",
                         "Karachi", "Lahore", "Faisalabad", "Karachi", "Lahore"],
        "Category": ["Staff", "Staff", "Staff", "Operator", "Staff", "Staff", "Staff", "Operator", "Staff", "Staff"],
        "Level": ["Junior", "Junior", "Mid", "Junior", "Junior", "Mid", "Senior", "Mid", "Junior", "Junior"],
        "per_min_salary": [3, 2, 6, 3, 3, 7, 11, 6, 2, 4]
    }
    df = pd.DataFrame(sample)
    df['Emp_ID'] = df['Emp_ID'].astype(str).str.strip()
    return df

def save_master(df):
    """Save master DataFrame to CSV."""
    if 'per min salary' in df.columns:
        df = df.drop(columns=['per min salary'])
    df.to_csv('employee_master.csv', index=False)
    st.success("✅ Master data saved to employee_master.csv")

# =============================================================================
# ATTENDANCE AGGREGATION (MULTIPLE IN/OUT PER DAY)
# =============================================================================
def aggregate_attendance(attendance_df, shift_start, shift_end, grace_minutes):
    start_mins = shift_start.hour * 60 + shift_start.minute
    end_mins = shift_end.hour * 60 + shift_end.minute

    daily_records = []
    grouped = attendance_df.groupby(['Emp_ID', 'Date'])

    for (emp_id, date), group in grouped:
        group = group.sort_values('Time_Mins')
        intervals = []
        active_in = None

        for _, row in group.iterrows():
            if row['Action'] == 'In':
                if active_in is None:
                    active_in = row['Time_Mins']
            elif row['Action'] == 'Out':
                if active_in is not None:
                    intervals.append((active_in, row['Time_Mins']))
                    active_in = None

        if not intervals:
            continue

        first_in = intervals[0][0]
        last_out = intervals[-1][1]
        total_work = sum(out - inp for inp, out in intervals)
        total_break = 0
        for i in range(1, len(intervals)):
            gap = intervals[i][0] - intervals[i-1][1]
            if gap > 0:
                total_break += gap

        late = max(0, first_in - start_mins)
        total_non_work = late + total_break
        missed = max(0, total_non_work - grace_minutes)
        overtime = max(0, last_out - end_mins)
        early_exit = max(0, end_mins - last_out)
        work_hours = total_work / 60
        late_flag = 1 if late > 0 else 0
        weekday = date.strftime('%A')

        daily_records.append({
            'Emp_ID': emp_id,
            'Date': date,
            'First_In_Mins': first_in,
            'Last_Out_Mins': last_out,
            'Late_Mins': late,
            'Work_Hours': work_hours,
            'Overtime_Mins': overtime,
            'Early_Exit_Mins': early_exit,
            'Missed_Mins': missed,
            'Break_Mins': total_break,
            'Late_Flag': late_flag,
            'Weekday': weekday,
            'Weekday_Num': date.weekday()
        })

    return pd.DataFrame(daily_records)

def process_attendance(master_df, attendance_df, shift_start, shift_end, grace_minutes):
    required_cols = ['Emp_ID', 'Date', 'Action', 'Time']
    missing = [c for c in required_cols if c not in attendance_df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None

    attendance_df['Emp_ID'] = attendance_df['Emp_ID'].astype(str).str.strip()
    attendance_df['Date'] = pd.to_datetime(attendance_df['Date'], errors='coerce')
    if attendance_df['Date'].isna().all():
        st.error("Invalid date format. Please check your CSV.")
        return None

    attendance_df['Time_Mins'] = attendance_df['Time'].apply(time_to_minutes)
    attendance_df = attendance_df[attendance_df['Action'].isin(['In', 'Out'])]

    agg_df = aggregate_attendance(attendance_df, shift_start, shift_end, grace_minutes)
    if agg_df.empty:
        st.warning("No complete In/Out pairs found.")
        return None

    result = agg_df.merge(master_df, on='Emp_ID', how='left')
    result['Name'] = result['Name'].fillna(result['Emp_ID'])
    result['Department'] = result['Department'].fillna('Not Assigned')
    result['Factory_Unit'] = result['Factory_Unit'].fillna('Not Assigned')
    result['per_min_salary'] = result['per_min_salary'].fillna(DEFAULT_PER_MIN_SALARY)
    result['Late_Cost'] = result['Late_Mins'] * result['per_min_salary']
    return result

# =============================================================================
# PLOTTING FUNCTIONS (unchanged)
# =============================================================================
def plot_daily_trend(df):
    daily = df.groupby('Date')['Late_Mins'].mean().reset_index()
    daily['MA7'] = daily['Late_Mins'].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Late_Mins'],
                              mode='lines+markers', name='Daily Avg',
                              line=dict(color=COLOR_PALETTE['primary'])))
    fig.add_trace(go.Scatter(x=daily['Date'], y=daily['MA7'],
                              mode='lines', name='7-Day MA',
                              line=dict(color=COLOR_PALETTE['success'], dash='dash')))
    fig.update_layout(title='Daily Lateness Trend with Moving Average',
                      xaxis_title='Date', yaxis_title='Avg Late Minutes',
                      template='plotly_dark')
    return fig

def plot_monthly_loss(df):
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month')['Late_Cost'].sum().reset_index()
    fig = px.line(monthly, x='Month', y='Late_Cost', markers=True,
                  title='Monthly Productivity Loss (PKR)',
                  color_discrete_sequence=[COLOR_PALETTE['danger']])
    fig.update_layout(template='plotly_dark')
    return fig

def plot_employee_ranking(df, metric, title, ascending=False, color_scale='Reds'):
    top = df.groupby(['Emp_ID', 'Name'])[metric].mean().reset_index()
    top = top.nlargest(10, metric) if not ascending else top.nsmallest(10, metric)
    fig = px.bar(top, y='Name', x=metric, orientation='h',
                 title=title, color=metric, color_continuous_scale=color_scale)
    fig.update_layout(template='plotly_dark', yaxis_title='')
    return fig

def plot_dept_loss(dept_stats):
    fig = px.bar(dept_stats, x='Department', y='Total_Loss',
                 title='Productivity Loss by Department (PKR)',
                 color='Total_Loss', color_continuous_scale='Reds')
    fig.update_layout(template='plotly_dark', xaxis_title='', yaxis_title='Loss (PKR)')
    return fig

def plot_factory_loss(factory_stats):
    fig = px.bar(factory_stats, x='Factory_Unit', y='Total_Loss',
                 title='Productivity Loss by Factory Unit (PKR)',
                 color='Total_Loss', color_continuous_scale='Reds')
    fig.update_layout(template='plotly_dark', xaxis_title='', yaxis_title='Loss (PKR)')
    return fig

def plot_dept_trend(df, department):
    dept_data = df[df['Department'] == department]
    daily = dept_data.groupby('Date')['Late_Mins'].mean().reset_index()
    fig = px.line(daily, x='Date', y='Late_Mins', markers=True,
                  title=f'Lateness Trend - {department}',
                  color_discrete_sequence=[COLOR_PALETTE['primary']])
    fig.update_layout(template='plotly_dark')
    return fig

def plot_dept_heatmap(df, department):
    dept_data = df[df['Department'] == department]
    pivot = dept_data.pivot_table(index='Emp_ID', columns='Weekday', values='Late_Mins', aggfunc='mean')
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(columns=order, fill_value=0)
    fig = px.imshow(pivot, text_auto='.1f', aspect='auto',
                    color_continuous_scale='Reds',
                    title=f'Late Minutes Heatmap - {department}')
    fig.update_layout(template='plotly_dark')
    return fig

def plot_factory_dept_comparison(df, factory):
    factory_data = df[df['Factory_Unit'] == factory]
    dept_avg = factory_data.groupby('Department')['Late_Mins'].mean().reset_index()
    fig = px.bar(dept_avg, x='Department', y='Late_Mins',
                 title=f'Average Lateness by Department in {factory}',
                 color='Late_Mins', color_continuous_scale='Reds')
    fig.update_layout(template='plotly_dark')
    return fig

def plot_factory_trend(df, factory):
    factory_data = df[df['Factory_Unit'] == factory]
    daily = factory_data.groupby('Date')['Late_Mins'].mean().reset_index()
    fig = px.line(daily, x='Date', y='Late_Mins', markers=True,
                  title=f'Lateness Trend - {factory}',
                  color_discrete_sequence=[COLOR_PALETTE['primary']])
    fig.update_layout(template='plotly_dark')
    return fig

def plot_factory_cost(df, factory):
    factory_data = df[df['Factory_Unit'] == factory]
    daily = factory_data.groupby('Date')['Late_Cost'].sum().reset_index()
    fig = px.area(daily, x='Date', y='Late_Cost',
                  title=f'Daily Productivity Loss - {factory} (PKR)',
                  color_discrete_sequence=[COLOR_PALETTE['danger']])
    fig.update_layout(template='plotly_dark')
    return fig

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================
def train_model(df):
    if df is None or df.empty or len(df) < 10:
        return None, None
    try:
        df = df.sort_values(['Emp_ID', 'Date'])
        df['Prev_Late'] = df.groupby('Emp_ID')['Late_Mins'].shift(1).fillna(0)
        df['Target'] = (df['Late_Mins'] > 0).astype(int)
        train = df.dropna(subset=['Prev_Late', 'Target'])
        if len(train) < 10:
            return None, None
        X = train[['Prev_Late']]
        y = train['Target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        return model, scaler
    except Exception:
        return None, None

def predict_next(df, model, scaler):
    if df is None or df.empty or model is None or scaler is None:
        return None
    try:
        latest = df.groupby('Emp_ID').agg({
            'Late_Mins': 'last',
            'Name': 'first',
            'Department': 'first'
        }).reset_index()
        latest.columns = ['Emp_ID', 'Prev_Late', 'Name', 'Department']
        X = latest[['Prev_Late']]
        X_scaled = scaler.transform(X)
        latest['Late_Prob'] = model.predict_proba(X_scaled)[:, 1]
        latest['Risk'] = latest['Late_Prob'].apply(
            lambda x: "High" if x > 0.6 else "Medium" if x > 0.3 else "Low"
        )
        return latest
    except Exception:
        return None

# =============================================================================
# RECOMMENDATION HELPER (Gemini API with retry)
# =============================================================================
def get_gemini_recommendations_with_retry(api_key, metrics, max_retries=3, initial_delay=2):
    """
    Call Gemini API with exponential backoff on rate limits.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    prompt = f"""You are an HR analytics expert. Based on the following attendance data, provide actionable recommendations to improve punctuality and productivity.

Data:
- Total employees: {metrics['total_employees']}
- Average lateness: {metrics['avg_late']:.1f} minutes
- On-time percentage: {metrics['ontime_pct']:.1f}%
- Total financial loss: PKR {metrics['total_loss']:,.0f}
- Average overtime: {metrics['avg_overtime']:.1f} minutes

Department performance:
{metrics['dept_summary']}

Top 5 worst employees (highest avg late):
{metrics['worst_employees']}

Weekday patterns:
{metrics['weekday_pattern']}

Please provide:
1. Key observations about overall attendance behavior.
2. Department‑specific recommendations.
3. Individual employee suggestions (for the worst offenders).
4. Policy recommendations (e.g., shift adjustments, grace period).
5. Any other insights that could help reduce lateness and improve productivity.
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {"Content-Type": "application/json"}
    
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 429:
                # Rate limit – wait and retry
                time.sleep(delay)
                delay *= 2  # exponential backoff
                continue
            response.raise_for_status()
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "No response from API. Please try again."
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return f"Error calling Gemini API after {max_retries} attempts: {str(e)}"
            time.sleep(delay)
            delay *= 2
    return "Failed to get recommendations after multiple retries."

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'master_df' not in st.session_state:
        st.session_state.master_df = load_employee_master()

    st.markdown("""
    <div style="text-align:center; padding:20px 0;">
        <h1 style="font-size:42px;">📊 Attendance Intelligence Dashboard</h1>
        <p style="color:#8892A0;">Enterprise-grade HR Analytics & Predictive Insights</p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # Sidebar: file upload, master edit button, and persistent API key
    with st.sidebar:
        st.header("📂 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Attendance CSV",
            type=["csv"],
            help="Columns: Emp_ID, Date, Action, Time"
        )
        with st.expander("📋 CSV Format Example"):
            st.code("""Emp_ID,Date,Action,Time
E001,1/1/2026,In,9:02
E001,1/1/2026,Out,12:02
E001,1/1/2026,In,16:14
E001,1/1/2026,Out,20:14
E002,1/1/2026,In,8:58
E002,1/1/2026,Out,17:24""")
        
        st.markdown("---")
        if st.button("✏️ Edit Master Data", use_container_width=True):
            st.session_state.show_editor = True

        # Persistent API key storage
        st.markdown("---")
        st.header("🔑 API Key (Recommendations)")
        API_KEY_FILE = "api_key.txt"
        saved_key = None
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, "r") as f:
                saved_key = f.read().strip()
                if saved_key == "":
                    saved_key = None
        if "api_key" not in st.session_state:
            st.session_state.api_key = saved_key if saved_key else ""

        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Gemini API key. It will be saved locally if you check the box below."
        )
        save_key = st.checkbox("Save this key permanently (stored in api_key.txt)", value=(saved_key is not None))

        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input

        if save_key:
            if st.session_state.api_key and st.session_state.api_key != saved_key:
                with open(API_KEY_FILE, "w") as f:
                    f.write(st.session_state.api_key)
                st.success("API key saved.")
            elif not st.session_state.api_key and saved_key:
                if os.path.exists(API_KEY_FILE):
                    os.remove(API_KEY_FILE)
                    st.success("Saved key removed.")
        else:
            if os.path.exists(API_KEY_FILE):
                os.remove(API_KEY_FILE)
                st.info("Saved key removed (unchecked).")

    # Master editor popup (appears in main area when button clicked)
    if st.session_state.get('show_editor', False):
        with st.expander("📝 Employee Master Data Editor", expanded=True):
            edit_df = st.session_state.master_df.copy()
            if 'per min salary' in edit_df.columns:
                edit_df = edit_df.drop(columns=['per min salary'])
            edited = st.data_editor(
                edit_df,
                num_rows="dynamic",
                use_container_width=True,
                key="master_editor_popup"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Changes"):
                    st.session_state.master_df = edited
                    save_master(edited)
                    if st.session_state.processed_df is not None:
                        st.session_state.processed_df = None
                        st.rerun()
                    st.success("Master data saved.")
            with col2:
                if st.button("Close Editor"):
                    st.session_state.show_editor = False
                    st.rerun()

    # Process attendance if file uploaded
    if uploaded_file is not None:
        try:
            attendance_df = pd.read_csv(uploaded_file)
            st.session_state.raw_attendance = attendance_df
            st.success(f"✅ Loaded {len(attendance_df)} records")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    # Main content
    if 'raw_attendance' not in st.session_state:
        st.info("👈 Please upload an attendance CSV file from the sidebar.")
        return

    # TABS: Tracking, Predictive, Recommendations
    main_tab1, main_tab2, main_tab3 = st.tabs(["📈 Tracking", "🔮 Predictive Analytics", "💡 Recommendations"])

    with main_tab1:
        st.subheader("Shift & Grace Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            shift_start = st.time_input("Shift Start", dt_time(9, 0), key="shift_start")
        with col2:
            shift_end = st.time_input("Shift End", dt_time(18, 0), key="shift_end")
        with col3:
            grace_minutes = st.number_input("Grace Minutes (Late + Break)", value=30, min_value=0, step=5, key="grace_minutes")
        
        st.divider()
        
        st.markdown("### 📅 Date Range Filter")
        # Process attendance if not already done or if settings changed
        if (st.session_state.processed_df is None or
            st.session_state.get('last_shift_start') != shift_start or
            st.session_state.get('last_shift_end') != shift_end or
            st.session_state.get('last_grace') != grace_minutes):
            with st.spinner("Processing attendance data..."):
                processed = process_attendance(
                    st.session_state.master_df,
                    st.session_state.raw_attendance,
                    shift_start,
                    shift_end,
                    grace_minutes
                )
                if processed is not None:
                    st.session_state.processed_df = processed
                    st.session_state.last_shift_start = shift_start
                    st.session_state.last_shift_end = shift_end
                    st.session_state.last_grace = grace_minutes
                    st.success(f"✅ Processed {len(processed)} attendance records")
                else:
                    st.session_state.processed_df = None
                    st.error("Processing failed.")
                    return
        
        df = st.session_state.processed_df
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        date_range = st.date_input(
            "Select date range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key="tracking_date_range"
        )
        if len(date_range) == 2:
            start, end = date_range
            filtered_df = df[(df['Date'].dt.date >= start) & (df['Date'].dt.date <= end)]
            st.info(f"Showing data from {start} to {end} – {len(filtered_df)} records")
        else:
            filtered_df = df

        # Sub‑tabs inside Tracking
        sub_tabs = st.tabs(["🏢 Executive Summary", "👤 Employee Profile", "🏭 Department Analysis", "🏭 Factory Unit Analysis"])

        # Department and factory summaries (using filtered data)
        dept_summary = filtered_df.groupby('Department').agg(
            Employees=('Emp_ID', 'nunique'),
            Avg_Late=('Late_Mins', 'mean'),
            Total_Loss=('Late_Cost', 'sum')
        ).reset_index()
        factory_summary = filtered_df.groupby('Factory_Unit').agg(
            Employees=('Emp_ID', 'nunique'),
            Avg_Late=('Late_Mins', 'mean'),
            Total_Loss=('Late_Cost', 'sum')
        ).reset_index()

        # ----- Sub‑tab 1: Executive Summary -----
        with sub_tabs[0]:
            st.subheader("Organization‑wide Performance")

            dept_valid = dept_summary[dept_summary['Department'] != 'Not Assigned']
            factory_valid = factory_summary[factory_summary['Factory_Unit'] != 'Not Assigned']
            col1, col2, col3 = st.columns(3)
            if not dept_valid.empty:
                best_dept = dept_valid.loc[dept_valid['Avg_Late'].idxmin(), 'Department']
                col1.metric("Best Department", best_dept, delta="Lowest lateness")
            else:
                col1.metric("Best Department", "N/A")
            if not factory_valid.empty:
                best_factory = factory_valid.loc[factory_valid['Avg_Late'].idxmin(), 'Factory_Unit']
                col2.metric("Best Factory Unit", best_factory, delta="Lowest lateness")
            else:
                col2.metric("Best Factory Unit", "N/A")
            emp_late = filtered_df.groupby(['Emp_ID', 'Name'])['Late_Mins'].mean().reset_index()
            if not emp_late.empty:
                best_emp = emp_late.loc[emp_late['Late_Mins'].idxmin(), 'Name']
                col3.metric("Best Employee", best_emp, delta=f"{emp_late['Late_Mins'].min():.1f} min avg")
            else:
                col3.metric("Best Employee", "N/A")

            col1, col2 = st.columns(2)
            with col1:
                if not dept_valid.empty:
                    st.plotly_chart(plot_dept_loss(dept_valid), use_container_width=True)
                else:
                    st.info("No department data")
            with col2:
                if not factory_valid.empty:
                    st.plotly_chart(plot_factory_loss(factory_valid), use_container_width=True)
                else:
                    st.info("No factory data")

            col1, col2 = st.columns(2)
            with col1:
                if not filtered_df.empty:
                    st.plotly_chart(plot_daily_trend(filtered_df), use_container_width=True)
                else:
                    st.info("No data for trend")
            with col2:
                if not filtered_df.empty:
                    st.plotly_chart(plot_monthly_loss(filtered_df), use_container_width=True)
                else:
                    st.info("No data for loss trend")

            col1, col2 = st.columns(2)
            with col1:
                if not filtered_df.empty:
                    st.plotly_chart(plot_employee_ranking(filtered_df, 'Late_Mins', '🔴 Top 10 Worst Employees (Avg Late)',
                                                          ascending=True, color_scale='Reds'), use_container_width=True)
                else:
                    st.info("No data")
            with col2:
                if not filtered_df.empty:
                    st.plotly_chart(plot_employee_ranking(filtered_df, 'Late_Mins', '🟢 Top 10 Best Employees (Avg Late)',
                                                          ascending=False, color_scale='Greens'), use_container_width=True)
                else:
                    st.info("No data")

        # ----- Sub‑tab 2: Employee Profile -----
        with sub_tabs[1]:
            st.subheader("Employee Profile")
            employees = filtered_df['Name'].unique()
            if len(employees) == 0:
                st.warning("No employee data available")
            else:
                selected_emp = st.selectbox("Select Employee", employees, key="emp_select")
                emp_data = filtered_df[filtered_df['Name'] == selected_emp].copy()
                if not emp_data.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Avg Late (min)", f"{emp_data['Late_Mins'].mean():.1f}")
                    col2.metric("Avg Work Hours", f"{emp_data['Work_Hours'].mean():.1f}")
                    col3.metric("Total Missed (min)", f"{emp_data['Missed_Mins'].sum():.0f}")
                    col4.metric("Total Late Cost", f"{emp_data['Late_Cost'].sum():,.0f} PKR")

                    st.markdown("#### 📈 Lateness Trend")
                    if len(emp_data) > 1:
                        fig = px.line(emp_data.sort_values('Date'), x='Date', y='Late_Mins',
                                      title=f"Late Minutes Trend for {selected_emp}",
                                      markers=True, color_discrete_sequence=[COLOR_PALETTE['danger']])
                        fig.update_layout(template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data for trend chart")

                    st.markdown("#### 📋 Attendance Records")
                    emp_data['In_Time'] = emp_data['First_In_Mins'].apply(
                        lambda x: f"{x//60:02d}:{x%60:02d}" if pd.notna(x) else ""
                    )
                    emp_data['Out_Time'] = emp_data['Last_Out_Mins'].apply(
                        lambda x: f"{x//60:02d}:{x%60:02d}" if pd.notna(x) else ""
                    )
                    display_cols = ['Date', 'In_Time', 'Out_Time', 'Late_Mins', 'Work_Hours', 'Break_Mins', 'Late_Cost']
                    st.dataframe(emp_data[display_cols].sort_values('Date', ascending=False), use_container_width=True)

        # ----- Sub‑tab 3: Department Analysis -----
        with sub_tabs[2]:
            st.subheader("Department Analysis")
            departments = sorted(filtered_df['Department'].unique())
            if len(departments) == 0:
                st.warning("No department data available")
            else:
                selected_dept = st.selectbox("Select Department", departments, key="dept_filter")
                dept_data = filtered_df[filtered_df['Department'] == selected_dept]
                kpi_cols = st.columns(3)
                kpi_cols[0].metric("Avg Lateness", f"{dept_data['Late_Mins'].mean():.1f} min")
                kpi_cols[1].metric("Total Loss", f"{dept_data['Late_Cost'].sum():,.0f} PKR")
                kpi_cols[2].metric("Employees", dept_data['Emp_ID'].nunique())
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_employee_ranking(dept_data, 'Late_Mins',
                                                          f'Employee Ranking in {selected_dept} (Avg Late)',
                                                          ascending=True, color_scale='Reds'), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_dept_trend(filtered_df, selected_dept), use_container_width=True)
                st.plotly_chart(plot_dept_heatmap(filtered_df, selected_dept), use_container_width=True)
                emp_table = dept_data.groupby(['Emp_ID', 'Name']).agg(
                    Total_Days=('Date', 'nunique'),
                    Avg_Late=('Late_Mins', 'mean'),
                    Total_Loss=('Late_Cost', 'sum')
                ).reset_index().sort_values('Avg_Late', ascending=False)
                st.markdown(f"#### Employees in {selected_dept}")
                st.dataframe(emp_table, use_container_width=True)

        # ----- Sub‑tab 4: Factory Unit Analysis -----
        with sub_tabs[3]:
            st.subheader("Factory Unit Analysis")
            factories = sorted(filtered_df['Factory_Unit'].unique())
            if len(factories) == 0:
                st.warning("No factory data available")
            else:
                selected_factory = st.selectbox("Select Factory Unit", factories, key="factory_filter")
                factory_data = filtered_df[filtered_df['Factory_Unit'] == selected_factory]
                kpi_cols = st.columns(3)
                kpi_cols[0].metric("Avg Lateness", f"{factory_data['Late_Mins'].mean():.1f} min")
                kpi_cols[1].metric("Total Loss", f"{factory_data['Late_Cost'].sum():,.0f} PKR")
                kpi_cols[2].metric("Employees", factory_data['Emp_ID'].nunique())
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_factory_dept_comparison(filtered_df, selected_factory), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_factory_trend(filtered_df, selected_factory), use_container_width=True)
                st.plotly_chart(plot_factory_cost(filtered_df, selected_factory), use_container_width=True)
                emp_table = factory_data.groupby(['Emp_ID', 'Name', 'Department']).agg(
                    Total_Days=('Date', 'nunique'),
                    Avg_Late=('Late_Mins', 'mean'),
                    Total_Loss=('Late_Cost', 'sum')
                ).reset_index().sort_values('Avg_Late', ascending=False)
                st.markdown(f"#### Employees in {selected_factory}")
                st.dataframe(emp_table, use_container_width=True)

    # ========================== PREDICTIVE ANALYTICS TAB ==========================
    with main_tab2:
        st.subheader("Lateness Prediction & Risk Analysis")
        model, scaler = train_model(filtered_df)
        if model is not None:
            pred_df = predict_next(filtered_df, model, scaler)
            if pred_df is not None:
                st.markdown("### 🚨 High Risk Employees")
                high = pred_df[pred_df['Risk'] == "High"].sort_values('Late_Prob', ascending=False)
                if len(high) > 0:
                    st.dataframe(high[['Name', 'Department', 'Late_Prob', 'Risk']])
                    fig_high = px.bar(high, x='Name', y='Late_Prob', title='High Risk Employees', color='Risk',
                                      color_discrete_map={'High': COLOR_PALETTE['danger']})
                    fig_high.update_layout(template='plotly_dark')
                    st.plotly_chart(fig_high, use_container_width=True)
                else:
                    st.success("✅ No high-risk employees detected.")

                st.markdown("### ⚠️ Medium Risk Employees")
                medium = pred_df[pred_df['Risk'] == "Medium"].sort_values('Late_Prob', ascending=False)
                if len(medium) > 0:
                    st.dataframe(medium[['Name', 'Department', 'Late_Prob', 'Risk']])

                st.markdown("### 📊 All Employees - Risk Distribution")
                risk_counts = pred_df['Risk'].value_counts()
                fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, title='Risk Distribution',
                                 color_discrete_map={'Low': COLOR_PALETTE['success'],
                                                     'Medium': COLOR_PALETTE['warning'],
                                                     'High': COLOR_PALETTE['danger']})
                fig_pie.update_layout(template='plotly_dark')
                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("### 📋 All Employee Predictions")
                st.dataframe(pred_df[['Name', 'Department', 'Late_Prob', 'Risk']].sort_values('Late_Prob', ascending=False))
            else:
                st.warning("Unable to generate predictions. Please ensure enough historical data.")
        else:
            st.warning("Not enough data for prediction. Minimum 10 attendance records required.")
        
        st.markdown("### 🚨 Extreme Lateness Outliers")
        Q1 = filtered_df['Late_Mins'].quantile(0.25)
        Q3 = filtered_df['Late_Mins'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR
        outliers = filtered_df[filtered_df['Late_Mins'] > outlier_threshold][['Name', 'Department', 'Date', 'Late_Mins']]
        if not outliers.empty:
            st.dataframe(outliers.sort_values('Late_Mins', ascending=False).head(10))
        else:
            st.info("No outliers detected.")

    # ========================== RECOMMENDATIONS TAB (with button) ==========================
    with main_tab3:
        st.subheader("💡 AI-Driven Recommendations")
        if not st.session_state.get("api_key"):
            st.warning("Please enter your Gemini API key in the sidebar to enable recommendations.")
        else:
            if st.button("🔍 Get Recommendations", use_container_width=True):
                with st.spinner("Analyzing attendance data and generating insights..."):
                    # Department summary (exclude "Not Assigned")
                    dept_summary_filtered = dept_summary[dept_summary['Department'] != 'Not Assigned']
                    dept_summary_str = dept_summary_filtered.to_string(index=False) if not dept_summary_filtered.empty else "No departments with valid data."
                    
                    # Top 5 worst employees (by avg late)
                    worst_emp = filtered_df.groupby(['Emp_ID', 'Name'])['Late_Mins'].mean().reset_index()
                    worst_emp = worst_emp.nlargest(5, 'Late_Mins')
                    worst_emp_str = worst_emp[['Name', 'Late_Mins']].to_string(index=False) if not worst_emp.empty else "No data."
                    
                    # Weekday pattern (sorted by day order)
                    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekday_pattern = filtered_df.groupby('Weekday')['Late_Mins'].mean().reset_index()
                    weekday_pattern = weekday_pattern.set_index('Weekday').reindex(weekday_order, fill_value=0).reset_index()
                    weekday_pattern.columns = ['Weekday', 'Late_Mins']
                    weekday_str = weekday_pattern.to_string(index=False)
                    
                    metrics = {
                        'total_employees': filtered_df['Emp_ID'].nunique(),
                        'avg_late': filtered_df['Late_Mins'].mean(),
                        'ontime_pct': (filtered_df['Late_Flag'] == 0).mean() * 100,
                        'total_loss': filtered_df['Late_Cost'].sum(),
                        'avg_overtime': filtered_df['Overtime_Mins'].mean(),
                        'dept_summary': dept_summary_str,
                        'worst_employees': worst_emp_str,
                        'weekday_pattern': weekday_str
                    }
                    
                    recommendations = get_gemini_recommendations_with_retry(st.session_state.api_key, metrics)
                    
                st.markdown("### 📝 Recommendations")
                st.write(recommendations)

    # Download button (uses filtered data)
    st.divider()
    csv_data = filtered_df[['Emp_ID', 'Name', 'Department', 'Factory_Unit', 'Date',
                            'Late_Mins', 'Work_Hours', 'Break_Mins', 'Overtime_Mins', 'Late_Cost']].copy()
    csv_data['In_Time'] = filtered_df['First_In_Mins'].apply(lambda x: f"{x//60:02d}:{x%60:02d}" if pd.notna(x) else "")
    csv_data['Out_Time'] = filtered_df['Last_Out_Mins'].apply(lambda x: f"{x//60:02d}:{x%60:02d}" if pd.notna(x) else "")
    csv_data = csv_data.drop(columns=['First_In_Mins', 'Last_Out_Mins'], errors='ignore')
    csv_data = csv_data.to_csv(index=False)
    st.download_button("📥 Download Filtered Data as CSV", data=csv_data,
                       file_name=f"attendance_report_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime='text/csv')

if __name__ == "__main__":
    main()