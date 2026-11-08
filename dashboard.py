import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ── Load & Prepare Data ───────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("Housing.csv")
    le = LabelEncoder()
    yes_no_cols = ["mainroad", "guestroom", "basement",
                   "hotwaterheating", "airconditioning",
                   "prefarea", "furnishingstatus"]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return df, model, X, X_test, y_test, y_pred

df, model, X, X_test, y_test, y_pred = load_and_train()

# ── Title ─────────────────────────────────────────
st.title("🏠 House Price Prediction Dashboard")
st.markdown("Interactive ML dashboard — explore data and predict house prices using Linear Regression.")
st.markdown("---")

# ── KPI Metrics ───────────────────────────────────
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏠 Total Houses", f"{len(df):,}")
col2.metric("💰 Average Price", f"₹{df['price'].mean():,.0f}")
col3.metric("🤖 Model Accuracy", f"{r2*100:.2f}%")
col4.metric("📉 Avg Error (MAE)", f"₹{mae:,.0f}")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Data Analysis",
                              "🤖 ML Model",
                              "🔮 Price Predictor"])

# ── Tab 1: Data Analysis ──────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price Distribution")
        fig1 = px.histogram(df, x="price", nbins=30,
                            color_discrete_sequence=["#2196F3"],
                            title="Distribution of House Prices")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Price vs Area")
        fig2 = px.scatter(df, x="area", y="price",
                          color_discrete_sequence=["#1F3864"],
                          title="House Price vs Area",
                          labels={"area": "Area (sq ft)",
                                  "price": "Price"})
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Price by Bedrooms")
        avg_bed = df.groupby("bedrooms")["price"].mean().reset_index()
        fig3 = px.bar(avg_bed, x="bedrooms", y="price",
                      color_discrete_sequence=["#2E75B6"],
                      title="Average Price by Bedrooms")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig4 = px.imshow(corr, text_auto=".2f",
                         color_continuous_scale="Blues",
                         title="Feature Correlation Heatmap")
        st.plotly_chart(fig4, use_container_width=True)

# ── Tab 2: ML Model ───────────────────────────────
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs Predicted Prices")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color="#2E75B6", opacity=0.6),
            name="Predictions"))
        fig5.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect Prediction"))
        fig5.update_layout(
            title="Actual vs Predicted House Prices",
            xaxis_title="Actual Price",
            yaxis_title="Predicted Price")
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        }).sort_values("Coefficient", ascending=False)
        colors = ["#4CAF50" if c > 0 else "#FF7043"
                  for c in feat_imp["Coefficient"]]
        fig6 = px.bar(feat_imp, x="Coefficient", y="Feature",
                      orientation="h",
                      color="Feature",
                      color_discrete_sequence=colors,
                      title="Feature Importance")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("### 📊 Model Performance Metrics")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("R² Score", f"{r2:.4f}")
    mc2.metric("Accuracy", f"{r2*100:.2f}%")
    mc3.metric("MAE", f"₹{mae:,.0f}")

# ── Tab 3: Price Predictor ────────────────────────
with tab3:
    st.subheader("🔮 Predict Your House Price")
    st.markdown("Fill in the details below and get an instant price prediction!")

    col1, col2, col3 = st.columns(3)
    with col1:
        area        = st.number_input("Area (sq ft)", 1000, 20000, 5000)
        bedrooms    = st.slider("Bedrooms", 1, 6, 3)
        bathrooms   = st.slider("Bathrooms", 1, 4, 2)
        stories     = st.slider("Stories", 1, 4, 2)
    with col2:
        mainroad    = st.selectbox("Main Road", ["yes", "no"])
        guestroom   = st.selectbox("Guest Room", ["yes", "no"])
        basement    = st.selectbox("Basement", ["yes", "no"])
        parking     = st.slider("Parking Spots", 0, 3, 1)
    with col3:
        hotwaterheating  = st.selectbox("Hot Water Heating", ["yes", "no"])
        airconditioning  = st.selectbox("Air Conditioning", ["yes", "no"])
        prefarea         = st.selectbox("Preferred Area", ["yes", "no"])
        furnishingstatus = st.selectbox("Furnishing",
                                        ["furnished", "semi-furnished",
                                         "unfurnished"])

    def encode(val):
        return 1 if val == "yes" else 0

    furnish_map = {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}

    input_data = pd.DataFrame([[
        area, bedrooms, bathrooms, stories,
        encode(mainroad), encode(guestroom),
        encode(basement), encode(hotwaterheating),
        encode(airconditioning), parking,
        encode(prefarea), furnish_map[furnishingstatus]
    ]], columns=X.columns)

    if st.button("🔮 Predict Price", type="primary"):
        prediction = model.predict(input_data)[0]
        st.success(f"### 🏠 Estimated House Price: ₹{prediction:,.0f}")
        st.info(f"This prediction is based on a Linear Regression model with {r2*100:.2f}% accuracy.")

# ── Footer ────────────────────────────────────────
st.markdown("---")
st.markdown("**Built by Rebecca Raj A** | Dataset: Housing Prices (Kaggle) | Tools: Python, Streamlit, Scikit-learn, Plotly")