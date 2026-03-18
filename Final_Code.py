import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import time, timedelta, datetime
import numpy as np

# =============================================================================
# PAGE CONFIGURATION & CUSTOM CSS
# =============================================================================
st.set_page_config(page_title="Enterprise Attendance Analytics", layout="wide")

# Custom CSS for professional KPI cards
st.markdown("""
<style>
    .kpi-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: #FFFFFF;
        margin: 5px;
    }
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
    .green { color: #4CAF50; }
    .yellow { color: #FFC107; }
    .red { color: #F44336; }
    .blue { color: #2196F3; }
    .orange { color: #FF9800; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
SALARY_PER_MONTH = 60000
WORKING_DAYS_PER_MONTH = 26
WORKING_HOURS_PER_DAY = 8
WORKING_MINUTES_PER_MONTH = WORKING_DAYS_PER_MONTH * WORKING_HOURS_PER_DAY * 60
SALARY_PER_MINUTE = SALARY_PER_MONTH / WORKING_MINUTES_PER_MONTH

COLOR_PALETTE = {
    'primary': '#2196F3',      # Blue
    'success': '#4CAF50',       # Green
    'warning': '#FFC107',       # Yellow
    'danger': '#F44336',        # Red
    'info': '#00BCD4',          # Cyan
    'neutral': '#9E9E9E'        # Gray
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def time_to_minutes(t_val):
    """Convert time string or datetime to minutes since midnight."""
    try:
        t = pd.to_datetime(str(t_val))
        return t.hour * 60 + t.minute
    except:
        return 0

def create_kpi_card(col, title, value, subtext, color_class):
    """Render a styled KPI card in a column."""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value {color_class}">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess uploaded file."""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Basic validation
    required_cols = ['Emp_ID', 'Name', 'Dept', 'Date', 'Action', 'Time']
    if not all(col in df.columns for col in required_cols):
        st.error(f"File must contain columns: {required_cols}")
        return None
    
    return df

def process_data(df, shift_start, shift_end, grace_minutes):
    """Transform raw data into analytics-ready dataframe."""
    # Pivot to get In/Out per row
    summary = df.pivot_table(
        index=['Emp_ID', 'Name', 'Dept', 'Date'],
        columns='Action',
        values='Time',
        aggfunc='first'
    ).reset_index()
    
    if 'In' not in summary.columns or 'Out' not in summary.columns:
        st.error("File must contain both 'In' and 'Out' actions.")
        return None
    
    summary['Date'] = pd.to_datetime(summary['Date'])
    
    # Convert times to minutes
    summary['In_Mins'] = summary['In'].apply(time_to_minutes)
    summary['Out_Mins'] = summary['Out'].apply(time_to_minutes)
    
    # Shift boundaries
    start_mins = shift_start.hour * 60 + shift_start.minute
    end_mins = shift_end.hour * 60 + shift_end.minute
    
    # Late minutes (after grace)
    summary['Late_Mins'] = (summary['In_Mins'] - (start_mins + grace_minutes)).clip(lower=0)
    
    # Early exit minutes
    summary['Early_Exit_Mins'] = (end_mins - summary['Out_Mins']).clip(lower=0)
    
    # Overtime minutes
    summary['Overtime_Mins'] = (summary['Out_Mins'] - end_mins).clip(lower=0)
    
    # Work hours (assume 1 hour break)
    summary['Work_Hours'] = ((summary['Out_Mins'] - summary['In_Mins'] - 60) / 60).round(2)
    summary['Work_Hours'] = summary['Work_Hours'].clip(lower=0)
    
    # Late flag
    summary['Late_Flag'] = (summary['Late_Mins'] > 0).astype(int)
    
    # Financial impact
    summary['Late_Cost'] = summary['Late_Mins'] * SALARY_PER_MINUTE
    
    # Performance score (composite KPI)
    summary['Score'] = (
        100
        - summary['Late_Mins'] * 0.5
        - summary['Early_Exit_Mins'] * 0.3
        + summary['Overtime_Mins'] * 0.1
    ).clip(0, 100)
    
    # Day of week
    summary['Weekday'] = summary['Date'].dt.day_name()
    summary['Weekday_Num'] = summary['Date'].dt.weekday  # Monday=0
    
    # Month for aggregation
    summary['Month'] = summary['Date'].dt.to_period('M').astype(str)
    
    return summary

def calculate_advanced_metrics(df):
    """Compute additional analytics and risk metrics."""
    # Employee-level aggregations
    emp_stats = df.groupby(['Emp_ID', 'Name', 'Dept']).agg(
        Total_Days=('Date', 'nunique'),
        Late_Days=('Late_Flag', 'sum'),
        Avg_Late_Mins=('Late_Mins', 'mean'),
        Median_Late_Mins=('Late_Mins', 'median'),
        Max_Late_Mins=('Late_Mins', 'max'),
        Total_Late_Cost=('Late_Cost', 'sum'),
        Avg_Work_Hours=('Work_Hours', 'mean'),
        Avg_Score=('Score', 'mean')
    ).reset_index()
    
    emp_stats['Late_Frequency'] = (emp_stats['Late_Days'] / emp_stats['Total_Days'] * 100).round(1)
    
    # Risk categorization based on frequency and severity
    conditions = [
        (emp_stats['Late_Frequency'] <= 10) & (emp_stats['Avg_Late_Mins'] <= 5),
        (emp_stats['Late_Frequency'] <= 20) | (emp_stats['Avg_Late_Mins'] <= 15),
        (emp_stats['Late_Frequency'] > 20) | (emp_stats['Avg_Late_Mins'] > 15)
    ]
    choices = ['Low', 'Medium', 'High']
    emp_stats['Risk_Category'] = np.select(conditions, choices, default='High')
    
    # Department-level aggregations
    dept_stats = df.groupby('Dept').agg(
        Employees=('Emp_ID', 'nunique'),
        Total_Late_Mins=('Late_Mins', 'sum'),
        Avg_Late_Mins=('Late_Mins', 'mean'),
        Late_Days=('Late_Flag', 'sum'),
        Total_Days=('Late_Flag', 'count'),
        Total_Late_Cost=('Late_Cost', 'sum'),
        Avg_Work_Hours=('Work_Hours', 'mean'),
        Avg_Score=('Score', 'mean')
    ).reset_index()
    dept_stats['Lateness_Rate'] = (dept_stats['Late_Days'] / dept_stats['Total_Days'] * 100).round(1)
    
    # Weekday patterns
    weekday_pattern = df.groupby('Weekday').agg(
        Avg_Late_Mins=('Late_Mins', 'mean'),
        Late_Count=('Late_Flag', 'sum'),
        Total_Days=('Late_Flag', 'count')
    ).reset_index()
    # Ensure correct order
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_pattern['Weekday'] = pd.Categorical(weekday_pattern['Weekday'], categories=order, ordered=True)
    weekday_pattern = weekday_pattern.sort_values('Weekday')
    
    # Outlier detection (IQR method)
    Q1 = df['Late_Mins'].quantile(0.25)
    Q3 = df['Late_Mins'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    outliers = df[df['Late_Mins'] > outlier_threshold][['Name', 'Dept', 'Date', 'Late_Mins']]
    
    # Correlation
    corr = df[['Late_Mins', 'Work_Hours', 'Overtime_Mins', 'Early_Exit_Mins']].corr()
    
    return emp_stats, dept_stats, weekday_pattern, outliers, corr

def generate_insights(df, dept_stats, emp_stats, outliers):
    """Generate automated text insights."""
    insights = []
    
    # Worst department
    worst_dept = dept_stats.loc[dept_stats['Avg_Late_Mins'].idxmax()]
    insights.append(f"🔴 **{worst_dept['Dept']}** department has the highest average lateness ({worst_dept['Avg_Late_Mins']:.1f} min).")
    
    # Best department
    best_dept = dept_stats.loc[dept_stats['Avg_Late_Mins'].idxmin()]
    insights.append(f"🟢 **{best_dept['Dept']}** department is most punctual ({best_dept['Avg_Late_Mins']:.1f} min avg).")
    
    # Worst weekday
    weekday_avg = df.groupby('Weekday')['Late_Mins'].mean()
    if not weekday_avg.empty:
        worst_day = weekday_avg.idxmax()
        insights.append(f"⚠️ **{worst_day}** is the worst day for lateness (avg {weekday_avg.max():.1f} min).")
    
    # Outliers
    if not outliers.empty:
        insights.append(f"🚨 **{len(outliers)} outlier records** detected with extremely high lateness (> {outliers['Late_Mins'].quantile(0.75):.0f} min).")
    
    # Top risk employees
    high_risk = emp_stats[emp_stats['Risk_Category'] == 'High'].nlargest(3, 'Avg_Late_Mins')
    if not high_risk.empty:
        names = ', '.join(high_risk['Name'].tolist())
        insights.append(f"⚠️ **High risk employees**: {names}.")
    
    return insights

# =============================================================================
# PLOTTING FUNCTIONS (with unique keys assigned later)
# =============================================================================
def plot_daily_trend(df):
    """Daily average late minutes with 7-day moving average."""
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

def plot_dept_heatmap(df):
    """Heatmap of average late minutes by department and weekday."""
    pivot = df.pivot_table(index='Dept', columns='Weekday', values='Late_Mins', aggfunc='mean')
    # Reorder weekdays
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(columns=order, fill_value=0)
    
    fig = px.imshow(pivot, text_auto='.1f', aspect='auto',
                    color_continuous_scale='Reds',
                    title='Average Late Minutes: Department vs Weekday')
    fig.update_layout(template='plotly_dark')
    return fig

def plot_distribution(df):
    """Histogram of late minutes."""
    fig = px.histogram(df, x='Late_Mins', nbins=30,
                       title='Distribution of Late Minutes',
                       color_discrete_sequence=[COLOR_PALETTE['primary']])
    fig.update_layout(template='plotly_dark',
                      xaxis_title='Late Minutes', yaxis_title='Frequency')
    return fig

def plot_correlation_scatter(df):
    """Scatter plot of late minutes vs work hours."""
    fig = px.scatter(df, x='Late_Mins', y='Work_Hours',
                     trendline='ols',
                     title='Correlation: Late Minutes vs Work Hours',
                     color_discrete_sequence=[COLOR_PALETTE['info']])
    fig.update_layout(template='plotly_dark')
    return fig

def plot_dept_loss(dept_stats):
    """Department-wise financial loss."""
    fig = px.bar(dept_stats, x='Dept', y='Total_Late_Cost',
                 title='Productivity Loss by Department (PKR)',
                 color='Total_Late_Cost',
                 color_continuous_scale='Reds')
    fig.update_layout(template='plotly_dark', xaxis_title='', yaxis_title='Loss (PKR)')
    return fig

def plot_leaderboard(emp_stats, metric='Avg_Score', ascending=False, title='Top Employees'):
    """Horizontal bar chart for leaderboard."""
    top = emp_stats.nlargest(10, metric) if not ascending else emp_stats.nsmallest(10, metric)
    fig = px.bar(top, y='Name', x=metric, orientation='h',
                 title=title,
                 color=metric, color_continuous_scale='Blues' if not ascending else 'Reds_r')
    fig.update_layout(template='plotly_dark', yaxis_title='')
    return fig

# =============================================================================
# SIDEBAR & SESSION STATE
# =============================================================================
if 'view_sample' not in st.session_state:
    st.session_state.view_sample = False

with st.sidebar:
    st.header("📂 Data Source")
    if st.button("📄 View Standard Format"):
        st.session_state.view_sample = True
        st.rerun()
    
    uploaded_file = st.file_uploader("Upload Attendance File", type=['xlsx', 'csv'])
    
    st.header("⚙️ Settings")
    shift_start = st.time_input("Shift Start", time(9, 0))
    shift_end = st.time_input("Shift End", time(18, 0))
    grace_minutes = st.number_input("Grace Minutes", value=10, min_value=0)
    
    st.markdown("---")
    st.info("ℹ️ Use filters after uploading data.")

# =============================================================================
# SAMPLE FORMAT VIEW
# =============================================================================
if st.session_state.view_sample:
    st.title("✅ Required Data Format (Sample)")
    sample = pd.DataFrame([
        [101, 'Ali Khan', 'IT', '2023-10-01', 'In', '09:05:00'],
        [101, 'Ali Khan', 'IT', '2023-10-01', 'Out', '18:10:00'],
        [102, 'Sara Ahmed', 'HR', '2023-10-01', 'In', '09:15:00'],
        [102, 'Sara Ahmed', 'HR', '2023-10-01', 'Out', '17:55:00'],
    ], columns=['Emp_ID', 'Name', 'Dept', 'Date', 'Action', 'Time'])
    st.table(sample)
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.view_sample = False
        st.rerun()
    st.stop()

# =============================================================================
# MAIN APP
# =============================================================================
st.title("🏢 Enterprise Attendance Analytics Dashboard")

if not uploaded_file:
    st.info("👈 Please upload an attendance file to begin.")
    st.stop()

# Load and process data
raw_df = load_data(uploaded_file)
if raw_df is None:
    st.stop()

df = process_data(raw_df, shift_start, shift_end, grace_minutes)
if df is None:
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

all_depts = sorted(df['Dept'].unique())
selected_depts = []
with st.sidebar.expander("🏢 Departments", expanded=True):
    for dept in all_depts:
        if st.checkbox(dept, value=True, key=f"dept_{dept}"):
            selected_depts.append(dept)

if len(date_range) != 2:
    st.warning("Please select both start and end dates.")
    st.stop()

start_date, end_date = date_range
mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date) & (df['Dept'].isin(selected_depts))
filtered_df = df.loc[mask].copy()

if filtered_df.empty:
    st.warning("No data for selected filters.")
    st.stop()

# Calculate advanced metrics
emp_stats, dept_stats, weekday_pattern, outliers, corr = calculate_advanced_metrics(filtered_df)

# =============================================================================
# KPI ROW (PROFESSIONAL CARDS)
# =============================================================================
st.subheader("📊 Key Performance Indicators")
kpi_cols = st.columns(6)

total_employees = filtered_df['Emp_ID'].nunique()
avg_late = filtered_df['Late_Mins'].mean()
avg_work_hours = filtered_df['Work_Hours'].mean()
ontime_pct = (filtered_df['Late_Flag'] == 0).mean() * 100
total_loss = filtered_df['Late_Cost'].sum()
avg_overtime = filtered_df['Overtime_Mins'].mean()

# Color classes based on thresholds
late_color = 'green' if avg_late < 5 else 'yellow' if avg_late < 15 else 'red'
ontime_color = 'green' if ontime_pct > 95 else 'yellow' if ontime_pct > 85 else 'red'
loss_color = 'green' if total_loss < 10000 else 'yellow' if total_loss < 50000 else 'red'

create_kpi_card(kpi_cols[0], "👥 Total Employees", f"{total_employees}", "Active staff", "blue")
create_kpi_card(kpi_cols[1], "⏰ Avg Late", f"{avg_late:.1f} min", "Per employee per day", late_color)
create_kpi_card(kpi_cols[2], "📅 Avg Work Hours", f"{avg_work_hours:.1f} hrs", "Per day", "blue")
create_kpi_card(kpi_cols[3], "✅ On-Time %", f"{ontime_pct:.1f}%", "Compliance rate", ontime_color)
create_kpi_card(kpi_cols[4], "💰 Total Loss", f"{total_loss:,.0f} PKR", "Productivity cost", loss_color)
create_kpi_card(kpi_cols[5], "⌛ Avg Overtime", f"{avg_overtime:.1f} min", "Per day", "orange")

st.divider()

# =============================================================================
# INTELLIGENCE LAYER: AUTOMATED INSIGHTS
# =============================================================================
insights = generate_insights(filtered_df, dept_stats, emp_stats, outliers)
with st.expander("💡 Automated Insights", expanded=True):
    for insight in insights:
        st.markdown(insight)

st.divider()

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", "🏢 Department Deep Dive", "👥 Employee Analytics", 
    "📅 Trends & Patterns", "🔬 Advanced Analytics"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_daily_trend(filtered_df), use_container_width=True, key="daily_trend_tab1")
    
    with col2:
        st.plotly_chart(plot_dept_heatmap(filtered_df), use_container_width=True, key="dept_heatmap_tab1")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.plotly_chart(plot_leaderboard(emp_stats, 'Avg_Score', ascending=False, 
                                         title='🏆 Top 10 Punctual Employees'), 
                                         use_container_width=True, key="leaderboard_top_tab1")
    
    with col4:
        st.plotly_chart(plot_leaderboard(emp_stats, 'Avg_Late_Mins', ascending=True,
                                         title='⚠️ Bottom 10 (Highest Lateness)'), 
                                         use_container_width=True, key="leaderboard_bottom_tab1")

# --- TAB 2: DEPARTMENT DEEP DIVE ---
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(dept_stats.sort_values('Avg_Late_Mins'), x='Avg_Late_Mins', y='Dept',
                     orientation='h', title='Average Lateness by Department',
                     color='Avg_Late_Mins', color_continuous_scale='Reds')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True, key="dept_avg_late_tab2")
    
    with col2:
        fig = px.bar(dept_stats, x='Dept', y='Lateness_Rate',
                     title='Lateness Rate (%) by Department',
                     color='Lateness_Rate', color_continuous_scale='Reds')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True, key="dept_lateness_rate_tab2")
    
    st.plotly_chart(plot_dept_loss(dept_stats), use_container_width=True, key="dept_loss_tab2")
    
    st.subheader("Department Summary Table")
    st.dataframe(dept_stats.style.format({
        'Avg_Late_Mins': '{:.1f}',
        'Lateness_Rate': '{:.1f}%',
        'Total_Late_Cost': '{:,.0f} PKR',
        'Avg_Work_Hours': '{:.1f}',
        'Avg_Score': '{:.1f}'
    }), use_container_width=True)

# --- TAB 3: EMPLOYEE ANALYTICS ---
with tab3:
    st.subheader("Employee Risk Categorization")
    risk_counts = emp_stats['Risk_Category'].value_counts().reset_index()
    risk_counts.columns = ['Risk', 'Count']
    fig = px.pie(risk_counts, values='Count', names='Risk', 
                 title='Employee Risk Distribution',
                 color='Risk', color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
    st.plotly_chart(fig, use_container_width=True, key="risk_pie_tab3")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_leaderboard(emp_stats, 'Avg_Score', ascending=False,
                                         title='Top 10 by Performance Score'), 
                                         use_container_width=True, key="leaderboard_score_tab3")
    with col2:
        st.plotly_chart(plot_leaderboard(emp_stats, 'Late_Frequency', ascending=True,
                                         title='Top 10 by Late Frequency (%)'), 
                                         use_container_width=True, key="leaderboard_freq_tab3")
    
    st.subheader("Employee Drill-Down")
    selected_emp = st.selectbox("Select Employee", filtered_df['Name'].unique())
    emp_data = filtered_df[filtered_df['Name'] == selected_emp]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Late", f"{emp_data['Late_Mins'].mean():.1f} min")
    col2.metric("Late Frequency", f"{(emp_data['Late_Flag'].mean()*100):.1f}%")
    col3.metric("Avg Score", f"{emp_data['Score'].mean():.1f}")
    col4.metric("Risk", emp_stats[emp_stats['Name'] == selected_emp]['Risk_Category'].iloc[0])
    
    fig = px.line(emp_data.sort_values('Date'), x='Date', y='Late_Mins',
                  title=f'Daily Lateness for {selected_emp}',
                  markers=True)
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True, key=f"employee_line_{selected_emp}")

# --- TAB 4: TRENDS & PATTERNS ---
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_daily_trend(filtered_df), use_container_width=True, key="daily_trend_tab4")
        
        # Weekday pattern
        fig = px.bar(weekday_pattern, x='Weekday', y='Avg_Late_Mins',
                     title='Average Lateness by Weekday',
                     color='Avg_Late_Mins', color_continuous_scale='Reds')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True, key="weekday_pattern_tab4")
    
    with col2:
        # Monthly loss trend
        monthly = filtered_df.groupby('Month')['Late_Cost'].sum().reset_index()
        fig = px.line(monthly, x='Month', y='Late_Cost',
                      title='Monthly Productivity Loss (PKR)',
                      markers=True, line_shape='spline')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True, key="monthly_loss_tab4")
        
        # Overtime trend
        ot_trend = filtered_df.groupby('Date')['Overtime_Mins'].mean().reset_index()
        fig = px.line(ot_trend, x='Date', y='Overtime_Mins',
                      title='Average Overtime Trend',
                      markers=True)
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True, key="overtime_trend_tab4")

# --- TAB 5: ADVANCED ANALYTICS ---
with tab5:
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_distribution(filtered_df), use_container_width=True, key="dist_tab5")
        st.plotly_chart(plot_correlation_scatter(filtered_df), use_container_width=True, key="corr_scatter_tab5")
    
    with col2:
        # Box plot by department
        fig = px.box(filtered_df, x='Dept', y='Late_Mins',
                     title='Late Minutes Distribution by Department',
                     color='Dept', color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="box_dept_tab5")
        
        # Outlier table
        st.subheader("🚨 Extreme Lateness Outliers")
        if not outliers.empty:
            st.dataframe(outliers.sort_values('Late_Mins', ascending=False).head(10))
        else:
            st.info("No outliers detected.")
    
    st.subheader("Correlation Matrix")
    fig = px.imshow(corr, text_auto=True, aspect='auto',
                    color_continuous_scale='RdBu', title='Correlation Between Metrics')
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True, key="corr_matrix_tab5")

# =============================================================================
# DOWNLOAD BUTTON
# =============================================================================
st.divider()
csv_data = filtered_df[['Name', 'Dept', 'Date', 'In', 'Out', 'Late_Mins', 
                        'Early_Exit_Mins', 'Overtime_Mins', 'Work_Hours', 
                        'Late_Cost', 'Score']].to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv_data,
    file_name=f"attendance_report_{start_date}_to_{end_date}.csv",
    mime='text/csv'
)