"""Streamlit interface for the preferred customer churn model."""

from __future__ import annotations

import streamlit as st

from churn_predictor import predict_customer, train_model


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_model():
    return train_model()


st.title("📉 Customer Churn Predictor")
st.caption("A LightGBM classifier trained on the IBM Telco Customer Churn dataset")

with st.sidebar:
    st.header("Model details")
    st.markdown(
        """
        **Preferred model:** LightGBM  
        **Notebook ROC-AUC:** 0.853  
        **Notebook churn recall:** 75%  
        **Notebook F1 score:** 0.63

        LightGBM was selected because it produced the strongest ROC-AUC and
        identified substantially more churners than the other evaluated models.
        """
    )
    st.info(
        "This portfolio application supports retention planning. Its prediction "
        "should be combined with business context before contacting a customer."
    )

st.write(
    "Enter a customer's account and service details. The model will estimate the "
    "likelihood that the customer will discontinue service."
)

with st.form("customer_form"):
    st.subheader("Customer and account")
    customer_col, account_col, billing_col = st.columns(3)

    with customer_col:
        gender = st.selectbox("Gender", ("Female", "Male"))
        senior_citizen = st.selectbox("Senior citizen", ("No", "Yes"))
        partner = st.selectbox("Has a partner", ("No", "Yes"))
        dependents = st.selectbox("Has dependents", ("No", "Yes"))

    with account_col:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
        paperless_billing = st.selectbox("Paperless billing", ("No", "Yes"))
        payment_method = st.selectbox(
            "Payment method",
            (
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ),
        )

    with billing_col:
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.05
        )
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, max_value=20000.0, value=840.0, step=0.05
        )
        phone_service = st.selectbox("Phone service", ("Yes", "No"))
        multiple_lines = st.selectbox(
            "Multiple lines",
            ("No", "Yes") if phone_service == "Yes" else ("No phone service",),
        )

    st.subheader("Internet services")
    internet_service = st.selectbox("Internet service", ("Fiber optic", "DSL", "No"))
    internet_options = ("No", "Yes") if internet_service != "No" else ("No internet service",)
    service_columns = st.columns(3)
    with service_columns[0]:
        online_security = st.selectbox("Online security", internet_options)
        online_backup = st.selectbox("Online backup", internet_options)
    with service_columns[1]:
        device_protection = st.selectbox("Device protection", internet_options)
        tech_support = st.selectbox("Tech support", internet_options)
    with service_columns[2]:
        streaming_tv = st.selectbox("Streaming TV", internet_options)
        streaming_movies = st.selectbox("Streaming movies", internet_options)

    submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen == "Yes"),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        with st.spinner("Training the model and calculating churn risk..."):
            model_bundle = get_model()
            result = predict_customer(model_bundle.pipeline, customer)

        st.divider()
        if result.will_churn:
            st.error("⚠️ High churn risk")
            st.write(
                "Consider prioritizing this customer for a retention review, while "
                "checking their recent service and support history."
            )
        else:
            st.success("✅ Lower churn risk")
            st.write("The model does not currently flag this customer as likely to churn.")

        st.metric("Estimated churn probability", f"{result.churn_probability:.1%}")
        st.progress(result.churn_probability)
        st.caption(
            f"Model trained on {model_bundle.training_rows:,} labeled customers in this "
            "repository. Probabilities are estimates, not guarantees."
        )
    except Exception:
        st.error("The model could not complete the prediction. Please review the inputs and try again.")

with st.expander("How the prediction works"):
    st.write(
        "Numeric values are median-imputed and service categories are one-hot encoded. "
        "Those features are passed to the class-balanced LightGBM model selected in the "
        "project analysis. Training and prediction share one pipeline, preventing the "
        "feature mismatches that can occur when preprocessing is performed separately."
    )
