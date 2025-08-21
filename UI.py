# A fully customized prop trading account app with a secondary admin dashboard.
# This app includes dynamic parameter linking, a gamified simulation, and a
# new admin page with data visualization and filtering for risk analysis.
# The forecasting model has been updated to conceptually show a total profit
# forecast for the firm using an LSTM-style approach.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Prop Trading Account Configurator",
    page_icon="📈",
    layout="wide"
)

# --- USER AUTHENTICATION ---
def check_login():
    """Returns `True` if the user is logged in."""
    if not st.session_state.get("logged_in"):
        show_login_form()
        return False
    return True

def show_login_form():
    """Displays a login form."""
    with st.form("login_form"):
        st.title("Admin Login")
        username = st.text_input("Username").lower()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

        if submitted:
            # Securely check if credentials exist and match
            if "credentials" in st.secrets and "usernames" in st.secrets["credentials"] and \
               username in st.secrets["credentials"]["usernames"] and \
               password == st.secrets["credentials"]["usernames"][username]["password"]:
                
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["name"] = st.secrets["credentials"]["usernames"][username]["name"]
                st.rerun() 
            else:
                st.error("Invalid username or password")

# --- YOUR ORIGINAL CODE STARTS HERE (UNCHANGED) ---

# A function to generate a sample database for the admin dashboard
def generate_sample_data(num_users=1500, max_accounts_per_user=15):
    """Generates a sample DataFrame for the admin dashboard with multiple accounts per user."""
    data = []
    tiers = ["Beginner", "Intermediate", "Pro"]
    
    user_ids = [f"user-{str(i).zfill(4)}" for i in range(num_users)]
    
    account_counter = 0
    for user_id in user_ids:
        num_accounts = np.random.randint(1, max_accounts_per_user + 1)
        for i in range(num_accounts):
            tier = np.random.choice(tiers)
            if tier == "Beginner":
                max_drawdown = 2500
                profit_target = 4000
                payout_cap = 5000
                consistency_req = np.random.randint(20, 40)
                trading_days_required = 30
            elif tier == "Intermediate":
                max_drawdown = 5000
                profit_target = 8000
                payout_cap = 15000
                consistency_req = np.random.randint(30, 50)
                trading_days_required = 60
            else: # Pro
                max_drawdown = 10000
                profit_target = 15000
                payout_cap = 25000
                consistency_req = np.random.randint(40, 60)
                trading_days_required = 90
            
            account_type = np.random.choice(["Evaluation", "Funded"])
            
            # Simulate current performance
            if account_type == "Funded":
                current_pnl = np.random.randint(0, profit_target + 500)
                status = "Active"
                recent_pnl_trajectory = np.random.normal(loc=profit_target/trading_days_required, scale=50, size=30)
            else: # Evaluation
                status = np.random.choice(["Active", "Failed"])
                if status == "Active":
                    current_pnl = np.random.randint(-1000, profit_target)
                else:
                    current_pnl = np.random.randint(-max_drawdown, 0)
                recent_pnl_trajectory = np.random.normal(loc=0, scale=30, size=30)
                
            account_id = f"{user_id}-account-{str(i).zfill(2)}"
            
            data.append([
                user_id, account_id, tier, account_type, status, max_drawdown,
                profit_target, consistency_req, payout_cap, trading_days_required,
                current_pnl, list(recent_pnl_trajectory)
            ])
            account_counter += 1
    
    df = pd.DataFrame(data, columns=[
        "User ID", "Account ID", "Tier", "Account Type", "Status", "Max Drawdown",
        "Profit Target", "Consistency Req", "Payout Cap", "Trading Days Required",
        "Current P&L", "Recent P&L Trajectory"
    ])
    return df

# A function to set the default values based on the selected tier
def set_defaults(tier):
    if tier == "Beginner":
        st.session_state.max_drawdown = 2500
        st.session_state.profit_target = 4000
        st.session_state.consistency_req = 20
        st.session_state.payout_cap = 5000
        st.session_state.trading_days = 30
        st.session_state.daily_loss_percentage = 20
    elif tier == "Intermediate":
        st.session_state.max_drawdown = 5000
        st.session_state.profit_target = 8000
        st.session_state.consistency_req = 30
        st.session_state.payout_cap = 15000
        st.session_state.trading_days = 60
        st.session_state.daily_loss_percentage = 25
    elif tier == "Pro":
        st.session_state.max_drawdown = 10000
        st.session_state.profit_target = 15000
        st.session_state.consistency_req = 40
        st.session_state.payout_cap = 25000
        st.session_state.trading_days = 90
        st.session_state.daily_loss_percentage = 30

# Callback functions to sync sliders and text inputs
def update_from_input(key):
    st.session_state[key] = st.session_state[f"{key}_input"]

def update_from_slider(key):
    st.session_state[key] = st.session_state[f"{key}_slider"]

# Callback function to handle the tier change correctly
def handle_tier_change():
    set_defaults(st.session_state.tier)
    if 'sim_results' in st.session_state:
        del st.session_state.sim_results
        
# Callback function for the risk threshold slider
def update_risk_threshold():
    st.session_state.risk_threshold = st.session_state.risk_threshold_slider

# --- Configuration Page Functions ---
def free_account_config():
    """Renders the configuration page for a free account with fixed rules."""
    st.subheader("Free Account Rules (Non-adjustable)")
    st.markdown("These are the fixed rules for all free accounts. You cannot change these settings.")

    # Fixed values for a free account
    max_drawdown = 1000
    daily_loss_limit = 200
    profit_target = 1500
    consistency_req = 50
    payout_cap = 500
    trading_days = 10

    st.write("---")

    # Display settings using disabled sliders and text inputs
    st.subheader("Account Maximum Drawdown")
    col_s_dd, col_i_dd = st.columns([4, 1])
    with col_s_dd:
        st.slider(
            "Set the maximum allowable drawdown:",
            min_value=0,
            max_value=2000,
            value=max_drawdown,
            disabled=True
        )
    with col_i_dd:
        st.number_input(
            "Value:",
            value=max_drawdown,
            disabled=True,
            key="free_dd_input"
        )
    st.markdown(f"**Current Value:** `${max_drawdown:,}`")
    st.write("---")

    st.subheader("Daily Loss Limit")
    col_s_dll, col_i_dll = st.columns([4, 1])
    with col_s_dll:
        st.slider(
            "Daily loss limit:",
            min_value=0,
            max_value=1000,
            value=daily_loss_limit,
            disabled=True
        )
    with col_i_dll:
        st.number_input(
            "Value:",
            value=daily_loss_limit,
            disabled=True,
            key="free_dll_input"
        )
    st.markdown(f"**Current Value:** `${daily_loss_limit:,}`")
    st.write("---")

    st.subheader("Profit Target")
    col_s_pt, col_i_pt = st.columns([4, 1])
    with col_s_pt:
        st.slider(
            "Set your profit target:",
            min_value=0,
            max_value=3000,
            value=profit_target,
            disabled=True
        )
    with col_i_pt:
        st.number_input(
            "Value:",
            value=profit_target,
            disabled=True,
            key="free_pt_input"
        )
    st.markdown(f"**Current Value:** `${profit_target:,}`")
    st.write("---")

    st.subheader("Consistency Requirement")
    col_s_cr, col_i_cr = st.columns([4, 1])
    with col_s_cr:
        st.slider(
            "Set the consistency requirement for your trading:",
            min_value=0,
            max_value=100,
            value=consistency_req,
            step=5,
            disabled=True
        )
    with col_i_cr:
        st.number_input(
            "Value:",
            value=consistency_req,
            disabled=True,
            key="free_cr_input"
        )
    st.markdown(f"**Current Value:** `{consistency_req}%`")
    st.write("---")

    st.subheader("Payout Cap")
    col_s_pc, col_i_pc = st.columns([4, 1])
    with col_s_pc:
        st.slider(
            "Set the maximum amount you can withdraw:",
            min_value=0,
            max_value=1000,
            value=payout_cap,
            disabled=True
        )
    with col_i_pc:
        st.number_input(
            "Value:",
            value=payout_cap,
            disabled=True,
            key="free_pc_input"
        )
    st.markdown(f"**Current Value:** `${payout_cap:,}`")
    st.write("---")

    st.subheader("Trading Days Required")
    col_s_td, col_i_td = st.columns([4, 1])
    with col_s_td:
        st.slider(
            "Enter the minimum number of trading days required:",
            min_value=1,
            max_value=30,
            value=trading_days,
            disabled=True
        )
    with col_i_td:
        st.number_input(
            "Value:",
            value=trading_days,
            disabled=True,
            key="free_td_input"
        )
    st.markdown(f"**Current Value:** `{trading_days}` days")
    st.write("---")
    
    st.button("Start Free Account")

def paid_account_config():
    """Renders the main configuration page for paid accounts."""
    # --- Header and Title ---
    st.title("Fully Customizable Prop Account")
    st.markdown("Configure your account with dynamic settings and simulate your performance.")
    st.write("---")

    # --- Account Tier and Price ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Select Account Tier")
        new_tier = st.selectbox(
            "Choose your account level:",
            ("Beginner", "Intermediate", "Pro"),
            on_change=handle_tier_change,
            key="tier"
        )
    with col2:
        st.subheader("Price")
        price = {"Beginner": "$350", "Intermediate": "$500", "Pro": "$1000"}
        st.metric(label=" ", value=price[st.session_state.tier])

    st.write("---")

    # --- Dynamic Parameters (Drawdown and Daily Loss) ---
    st.subheader("Account Maximum Drawdown")
    col_s_dd, col_i_dd = st.columns([4, 1])
    with col_s_dd:
        st.slider(
            "Set the maximum allowable drawdown:",
            min_value=0,
            max_value=10000,
            value=st.session_state.max_drawdown,
            step=100,
            key="max_drawdown_slider",
            on_change=lambda: update_from_slider("max_drawdown")
        )
    with col_i_dd:
        st.number_input(
            "Enter value:",
            min_value=0,
            max_value=10000,
            value=st.session_state.max_drawdown,
            step=100,
            key="max_drawdown_input",
            on_change=lambda: update_from_input("max_drawdown")
        )
    st.markdown(f"**Current Value:** `${st.session_state.max_drawdown:,}`")

    # The Daily Loss Limit is now a percentage of the Max Drawdown
    st.subheader("Daily Loss Limit")
    col_s_dll, col_i_dll = st.columns([4, 1])
    with col_s_dll:
        st.slider(
            "Set daily loss limit as a percentage of max drawdown:",
            min_value=1,
            max_value=50,
            value=st.session_state.daily_loss_percentage,
            step=1,
            help="This limit will automatically update as you change the Max Drawdown.",
            key="daily_loss_percentage_slider",
            on_change=lambda: update_from_slider("daily_loss_percentage")
        )
    with col_i_dll:
        st.number_input(
            "Enter value:",
            min_value=1,
            max_value=50,
            value=st.session_state.daily_loss_percentage,
            step=1,
            key="daily_loss_percentage_input",
            on_change=lambda: update_from_input("daily_loss_percentage")
        )
    daily_loss_limit = (st.session_state.max_drawdown * st.session_state.daily_loss_percentage) / 100
    st.markdown(f"**Current Daily Loss Limit:** `${daily_loss_limit:,}`")
    st.write("---")

    # --- Remaining Parameters ---
    st.subheader("Drawdown Type")
    drawdown_type = st.radio(
        "Choose the method for calculating your drawdown:",
        ("End of Day", "Trailing", "Static"),
        horizontal=True
    )
    st.write("---")

    st.subheader("Profit Target")
    col_s_pt, col_i_pt = st.columns([4, 1])
    with col_s_pt:
        st.slider(
            "Set your profit target:",
            min_value=0,
            max_value=20000,
            value=st.session_state.profit_target,
            step=500,
            key="profit_target_slider",
            on_change=lambda: update_from_slider("profit_target")
        )
    with col_i_pt:
        st.number_input(
            "Enter value:",
            min_value=0,
            max_value=20000,
            value=st.session_state.profit_target,
            step=500,
            key="profit_target_input",
            on_change=lambda: update_from_input("profit_target")
        )
    st.markdown(f"**Current Value:** `${st.session_state.profit_target:,}`")
    st.write("---")

    st.subheader("Consistency Requirement")
    col_s_cr, col_i_cr = st.columns([4, 1])
    with col_s_cr:
        st.slider(
            "Set the consistency requirement for your trading:",
            min_value=0,
            max_value=40,
            value=st.session_state.consistency_req,
            step=5,
            key="consistency_req_slider",
            on_change=lambda: update_from_slider("consistency_req")
        )
    with col_i_cr:
        st.number_input(
            "Enter value:",
            min_value=0,
            max_value=40,
            value=st.session_state.consistency_req,
            step=5,
            key="consistency_req_input",
            on_change=lambda: update_from_input("consistency_req")
        )
    st.markdown(f"**Current Value:** `{st.session_state.consistency_req}%`")
    st.write("---")

    st.subheader("Payout Cap")
    col_s_pc, col_i_pc = st.columns([4, 1])
    with col_s_pc:
        st.slider(
            "Set the maximum amount you can withdraw:",
            min_value=0,
            max_value=25000,
            value=st.session_state.payout_cap,
            step=100,
            key="payout_cap_slider",
            on_change=lambda: update_from_slider("payout_cap")
        )
    with col_i_pc:
        st.number_input(
            "Enter value:",
            min_value=0,
            max_value=25000,
            value=st.session_state.payout_cap,
            step=100,
            key="payout_cap_input",
            on_change=lambda: update_from_input("payout_cap")
        )
    st.markdown(f"**Current Value:** `${st.session_state.payout_cap:,}`")
    st.write("---")

    st.subheader("Trading Days Required")
    col_s_td, col_i_td = st.columns([4, 1])
    with col_s_td:
        st.slider(
            "Enter the minimum number of trading days required:",
            min_value=1,
            max_value=90,
            value=st.session_state.trading_days,
            step=1,
            key="trading_days_slider",
            on_change=lambda: update_from_slider("trading_days")
        )
    with col_i_td:
        st.number_input(
            "Enter value:",
            min_value=1,
            max_value=90,
            value=st.session_state.trading_days,
            step=1,
            key="trading_days_input",
            on_change=lambda: update_from_input("trading_days")
        )
    st.markdown(f"**Current Value:** `{st.session_state.trading_days}` days")
    st.write("---")

    # --- Gamified Simulation and Visualization ---
    st.subheader("Simulation")
    if st.button("Run Trading Challenge Simulation"):
        # Simple simulation logic
        days_to_simulate = st.session_state.trading_days
        daily_pnl = np.random.normal(loc=st.session_state.profit_target / days_to_simulate, scale=100, size=days_to_simulate)
        
        # Account for consistency requirement
        consistent_days = np.random.binomial(days_to_simulate, st.session_state.consistency_req / 100)
        daily_pnl[days_to_simulate - consistent_days:] *= 1.5

        cumulative_pnl = np.cumsum(daily_pnl)
        
        # Check if the challenge was passed
        passed = (cumulative_pnl[-1] > st.session_state.profit_target)
        
        # Store results in session state
        st.session_state.sim_results = {
            "passed": passed,
            "final_profit": cumulative_pnl[-1],
            "chart_data": pd.DataFrame({
                "Day": list(range(1, days_to_simulate + 1)),
                "Profit/Loss": cumulative_pnl
            }).set_index("Day"),
            "profit_target": st.session_state.profit_target
        }
        
    # Display the results and chart if a simulation has been run
    if st.session_state.sim_results:
        st.write("---")
        st.subheader("Challenge Results")
        results = st.session_state.sim_results
        
        if results["passed"]:
            st.success("🎉 You passed the trading challenge! 🎉")
            st.markdown(f"**Final Profit:** `${results['final_profit']:,.2f}` (Target: `${results['profit_target']:,}`)")
            st.balloons()
        else:
            st.error("📉 You did not pass the challenge. Try adjusting your settings! 📉")
            st.markdown(f"**Final Profit:** `${results['final_profit']:,.2f}` (Target: `${results['profit_target']:,}`)")

        # Display visualization of the P&L curve
        st.subheader("Profit and Loss Curve")
        st.line_chart(results["chart_data"])

def main_config_page():
    """Renders the main configuration page with a sub-page menu."""
    st.sidebar.subheader("Account Type")
    config_page_type = st.sidebar.radio(
        "Choose an account type:",
        ["Paid (Customizable)", "Free (Fixed Rules)"]
    )
    
    if config_page_type == "Paid (Customizable)":
        paid_account_config()
    elif config_page_type == "Free (Fixed Rules)":
        free_account_config()

# --- Admin Page Function ---
def admin_page():
    """Renders the admin dashboard page with added filters and search."""
    st.title("Admin Dashboard")
    st.markdown("Analyze account data and perform risk analysis.")
    st.write("---")

    # Generate or retrieve sample data
    if "df_admin" not in st.session_state or "User ID" not in st.session_state.df_admin.columns:
        st.session_state.df_admin = generate_sample_data()
    df = st.session_state.df_admin

    # Initialize risk threshold in session state if not present
    if "risk_threshold" not in st.session_state:
        st.session_state.risk_threshold = 50
    
    # Key Performance Indicators
    total_accounts = len(df)
    funded_accounts = len(df[df["Account Type"] == "Funded"])
    total_profit = df["Current P&L"].sum()
    
    # New metric: Funded accounts above the adjustable profit threshold
    above_threshold_accounts = len(df[
        (df["Account Type"] == "Funded") &
        (df["Current P&L"] > (st.session_state.risk_threshold / 100 * df["Profit Target"]))
    ])

    # New metric for profitable multi-account users
    multi_account_users = df['User ID'].value_counts()
    multi_account_users = multi_account_users[multi_account_users > 1].index
    
    funded_multi_account_df = df[
        df['User ID'].isin(multi_account_users) & 
        (df['Account Type'] == 'Funded') & 
        (df['Current P&L'] > 0)
    ]
    profitable_multi_account_users = funded_multi_account_df['User ID'].nunique()

    st.subheader("Key Metrics")
    st.metric("Total Accounts", f"{total_accounts:,}")
    st.metric("Total Funded", f"{funded_accounts:,}")
    st.metric("Total P&L", f"${total_profit:,.2f}")
    st.metric(
        f"Funded > {st.session_state.risk_threshold}% Target", 
        f"{above_threshold_accounts:,}"
    )
    st.metric("Users with Profitable Funded Accounts", f"{profitable_multi_account_users:,}")
    st.write("---")
    
    # --- New Filtering and Search Section ---
    st.subheader("Account Database Filters")
    
    search_query = st.text_input("Search by Account ID:")
    
    col_filters_1, col_filters_2 = st.columns(2)
    with col_filters_1:
        account_type_filter = st.multiselect(
            "Account Type:",
            options=df["Account Type"].unique(),
            default=df["Account Type"].unique()
        )
        status_filter = st.multiselect(
            "Account Status:",
            options=df["Status"].unique(),
            default=df["Status"].unique()
        )

    with col_filters_2:
        min_dd, max_dd = int(df["Max Drawdown"].min()), int(df["Max Drawdown"].max())
        drawdown_range = st.slider(
            "Max Drawdown Range:",
            min_value=min_dd,
            max_value=max_dd,
            value=(min_dd, max_dd),
            step=100
        )
        
        min_pc, max_pc = int(df["Payout Cap"].min()), int(df["Payout Cap"].max())
        payout_cap_range = st.slider(
            "Payout Cap Range:",
            min_value=min_pc,
            max_value=max_pc,
            value=(min_pc, max_pc),
            step=100
        )

    # Apply filters to the DataFrame
    filtered_df = df[
        df["Account ID"].str.contains(search_query, case=False, na=False) &
        df["Account Type"].isin(account_type_filter) &
        df["Status"].isin(status_filter) &
        (df["Max Drawdown"] >= drawdown_range[0]) &
        (df["Max Drawdown"] <= drawdown_range[1]) &
        (df["Payout Cap"] >= payout_cap_range[0]) &
        (df["Payout Cap"] <= payout_cap_range[1])
    ]

    st.markdown(f"**Accounts found:** {len(filtered_df):,}")
    st.dataframe(filtered_df, use_container_width=True)
    st.write("---")

    # Risk Analysis section
    st.subheader("Risk Analysis")
    st.markdown("Adjust the profit progress threshold to filter funded accounts.")
    
    col_s_rt, col_i_rt = st.columns([4, 1])
    with col_s_rt:
        st.slider(
            "Profit Progress Threshold (%):",
            min_value=0,
            max_value=100,
            value=st.session_state.risk_threshold,
            key="risk_threshold_slider",
            on_change=update_risk_threshold
        )
    with col_i_rt:
        st.number_input(
            "Enter %:",
            min_value=0,
            max_value=100,
            value=st.session_state.risk_threshold,
            key="risk_threshold_input",
            on_change=lambda: st.session_state.update(
                risk_threshold=st.session_state.risk_threshold_input
            )
        )
    
    risk_filtered_df = df[
        (df["Account Type"] == "Funded") &
        (df["Current P&L"] > (st.session_state.risk_threshold / 100 * df["Profit Target"]))
    ]
    st.dataframe(risk_filtered_df, use_container_width=True)
    st.write("---")
    
    # New section for multi-account users
    st.subheader("Multi-Account User Analysis")
    st.markdown(
        "Filtered list of users who have more than one account and at least one "
        "funded account with a positive P&L."
    )
    
    multi_account_users = df['User ID'].value_counts()
    multi_account_users = multi_account_users[multi_account_users > 1].index
    
    funded_multi_account_df = df[
        df['User ID'].isin(multi_account_users) & 
        (df['Account Type'] == 'Funded') & 
        (df['Current P&L'] > 0)
    ]
    
    if not funded_multi_account_df.empty:
        multi_account_user_ids = funded_multi_account_df['User ID'].unique()
        multi_account_user_details = df[df['User ID'].isin(multi_account_user_ids)]
        st.dataframe(multi_account_user_details.sort_values(by="User ID"), use_container_width=True)
    else:
        st.info("No users found matching the criteria.")
    
    st.write("---")

    # The new conceptual LSTM-based forecast
    st.subheader("Firm-wide Profit Forecast (Conceptual LSTM)")
    st.markdown(
        "This chart shows a conceptual forecast for the total firm-wide P&L. "
        "The **Historical** line shows a simulated history, and the **Forecast** "
        "line simulates a prediction from a conceptual LSTM model. "
    )
    
    days_in_history = 90
    historical_dates = pd.date_range(end=date.today(), periods=days_in_history)
    random_walk = np.random.normal(loc=total_profit / days_in_history, scale=5000, size=days_in_history)
    historical_cumulative_pnl = np.cumsum(random_walk)
    
    scaling_factor = total_profit / historical_cumulative_pnl[-1] if historical_cumulative_pnl[-1] != 0 else 1
    historical_cumulative_pnl_scaled = historical_cumulative_pnl * scaling_factor
    
    historical_df = pd.DataFrame({
        "Date": historical_dates,
        "Historical P&L": historical_cumulative_pnl_scaled,
    }).set_index("Date")

    days_in_forecast = 60
    forecast_dates = pd.date_range(start=historical_dates[-1] + timedelta(days=1), periods=days_in_forecast)
    
    last_10_days_trend = np.mean(np.diff(historical_cumulative_pnl_scaled[-10:]))
    forecast_pnl = np.random.normal(loc=last_10_days_trend, scale=1000, size=days_in_forecast)
    
    forecast_cumulative_pnl = np.cumsum(forecast_pnl) + historical_cumulative_pnl_scaled[-1]
    
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast P&L": forecast_cumulative_pnl,
    }).set_index("Date")

    combined_df = pd.merge(historical_df, forecast_df, how="outer", left_index=True, right_index=True)
    
    st.line_chart(combined_df)
    st.write("---")

def main():
    """Main application logic for page routing."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Configuration", "Admin Dashboard"])

    # --- Session State Initialization ---
    if "tier" not in st.session_state: st.session_state.tier = "Beginner"
    if "max_drawdown" not in st.session_state: st.session_state.max_drawdown = 2500
    if "profit_target" not in st.session_state: st.session_state.profit_target = 4000
    if "consistency_req" not in st.session_state: st.session_state.consistency_req = 20
    if "payout_cap" not in st.session_state: st.session_state.payout_cap = 5000
    if "trading_days" not in st.session_state: st.session_state.trading_days = 30
    if "daily_loss_percentage" not in st.session_state: st.session_state.daily_loss_percentage = 20
    if 'sim_results' not in st.session_state: st.session_state.sim_results = None
    if 'risk_threshold' not in st.session_state: st.session_state.risk_threshold = 50
    if "risk_threshold_slider" not in st.session_state: st.session_state.risk_threshold_slider = 50
    if "risk_threshold_input" not in st.session_state: st.session_state.risk_threshold_input = 50

    if page == "Configuration":
        main_config_page()
    elif page == "Admin Dashboard":
        admin_page()

# --- APP ROUTING ---
if __name__ == "__main__":
    if check_login():
        main()
