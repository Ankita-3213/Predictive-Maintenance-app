import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Predictive Maintenance - NASA Turbofan", layout="wide")

# Load data
@st.cache_data
def load_data():
    col_names = ['unit_number', 'time_in_cycles'] + \
                [f'op_setting_{i+1}' for i in range(3)] + \
                [f'sensor_measurement_{i+1}' for i in range(21)]
    data = pd.read_csv("train_FD001.txt", sep=' ', header=None)
    data.drop([26, 27], axis=1, inplace=True)  # drop extra empty columns
    data.columns = col_names

    # Calculate RUL
    max_cycles = data.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max']
    merged = data.merge(max_cycles, on='unit_number')
    data['RUL'] = merged['max'] - data['time_in_cycles']
    return data

data = load_data()

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Overview", "Visualizations", "Model Evaluation"],
        icons=["house", "table", "bar-chart", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    st.title("🏭 Predictive Maintenance Dashboard")
    st.markdown("""
        Welcome! This dashboard uses the **NASA C-MAPSS** dataset to predict the *Remaining Useful Life (RUL)* of turbofan engines using sensor data.
    """)

elif selected == "Data Overview":
    st.title("📂 Engine Data Sample")
    st.dataframe(data.head(30))
    st.markdown(f"**Total Records:** {data.shape[0]}")
    st.markdown(f"**Unique Engines:** {data['unit_number'].nunique()}")

elif selected == "Visualizations":
    st.title("📊 Exploratory Visualizations")

    st.subheader("Engine Life Distribution")
    engine_life = data.groupby('unit_number')['time_in_cycles'].max()
    fig1, ax1 = plt.subplots()
    sns.histplot(engine_life, kde=True, ax=ax1)
    ax1.set_title("Engine Life (Cycles) Distribution")
    ax1.set_xlabel("Cycles")
    st.pyplot(fig1)

    st.subheader("Sensor Trend for Selected Engine")
    engine_id = st.slider("Select Engine ID", 1, int(data['unit_number'].max()), 1)
    selected_engine = data[data['unit_number'] == engine_id]

    # Plotting sensors 2, 3, 4, 7
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for i in [ 2, 3, 4, 7]:
        ax2.plot(
            selected_engine['time_in_cycles'],
            selected_engine[f'sensor_measurement_{i}'],
            label=f'Sensor {i}'
        )

    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Sensor Reading")
    ax2.set_title(f"Sensor Measurements for Engine {engine_id}")
    ax2.legend()
    st.pyplot(fig2)

elif selected == "Model Evaluation":
    st.title("📈 Random Forest Model Evaluation for RUL")

    features = [col for col in data.columns if 'sensor' in col or 'op_setting' in col]
    X = data[features]
    y = data['RUL']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.success("✅ Model training complete")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    st.subheader("Actual vs Predicted RUL")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred, alpha=0.6, edgecolors='w')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax3.set_xlabel("Actual RUL")
    ax3.set_ylabel("Predicted RUL")
    ax3.set_title("Actual vs Predicted RUL")
    st.pyplot(fig3)

    # Feature Importance
    st.subheader("📌 Feature Importance")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by="Importance", ascending=False)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(10), ax=ax4)
    ax4.set_title("Top 10 Important Features (Sensors)")
    st.pyplot(fig4)
