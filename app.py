import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Review Sentiment Classifier & Cloud Computing Project",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Self-Contained Trained Sentiment Model (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    """
    Trains a lightweight TF-IDF + Logistic Regression sentiment classifier
    on a representative set of movie and product reviews.
    Runs instantaneously and eliminates dependency on external .pkl files.
    """
    training_data = [
        # Positive reviews
        ("This movie was absolutely fantastic, great acting and thrilling plot!", "Positive"),
        ("Excellent product, high build quality and fast shipping. Highly recommend!", "Positive"),
        ("Loved every minute of it, truly a masterpiece and wonderful cast.", "Positive"),
        ("Outstanding battery life and great performance, worth every penny.", "Positive"),
        ("Best purchase I have made this year, superb customer service.", "Positive"),
        ("Brilliant direction and amazing soundtrack, thoroughly enjoyed it.", "Positive"),
        ("Very satisfied with the quality, works as expected and easy to use.", "Positive"),
        ("Incredible experience, very impressive and exceeded my expectations.", "Positive"),
        ("Five stars! Will definitely buy again, great value for money.", "Positive"),
        ("Clean interface, fast delivery, and very reliable. Love it!", "Positive"),
        ("A heartwarming and genuinely delightful experience from start to finish.", "Positive"),
        ("Remarkable quality, sleek design, and very comfortable to use.", "Positive"),
        ("The staff was polite and the food was delicious and fresh.", "Positive"),
        ("Super happy with the result, works flawlessly without any glitches.", "Positive"),
        ("Top notch service and exceptional support, very pleased.", "Positive"),

        # Negative reviews
        ("Terrible movie, waste of time and money, completely boring.", "Negative"),
        ("Horrible product, broke on the first day of use. Do not buy!", "Negative"),
        ("Worst experience ever, customer support was rude and unhelpful.", "Negative"),
        ("Poor quality, cheap plastic material and arrived damaged.", "Negative"),
        ("Extremely disappointed with the performance, lagging and full of bugs.", "Negative"),
        ("Awful direction and pathetic storyline, I regret watching this.", "Negative"),
        ("Defective item, does not work as advertised, total scam.", "Negative"),
        ("Very bad battery life, died within an hour, complete waste of cash.", "Negative"),
        ("Slow delivery, package arrived torn, and item was missing parts.", "Negative"),
        ("Unhappy with this purchase, feels very flimsy and overpriced.", "Negative"),
        ("Utterly frustrating and chaotic, nothing worked as expected.", "Negative"),
        ("Terrible customer service, refused refund and ignored complaints.", "Negative"),
        ("Disastrous product, overheated immediately and stopped working.", "Negative"),
        ("Boring, dull, and uninspired. Save your money and skip it.", "Negative"),
        ("Never buying from this brand again, dreadful quality and support.", "Negative"),
    ]

    df = pd.DataFrame(training_data, columns=["text", "sentiment"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True)),
        ("clf", LogisticRegression(solver="liblinear", random_state=42))
    ])

    pipeline.fit(df["text"], df["sentiment"])
    return pipeline

model = load_sentiment_model()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("☁️ Cloud Computing")
    st.caption("Mini Project 3: Publish a Sentiment App Online")
    st.markdown("---")
    st.markdown("""
    **Project Highlights:**
    - ⚡ **Framework:** Streamlit (Python)
    - 🌐 **Cloud Deployment:** Streamlit Community Cloud
    - 📁 **Service Model:** Platform as a Service (PaaS)
    - 🔗 **Version Control:** GitHub
    """)
    st.markdown("---")
    st.markdown("### 💡 Quick Test Reviews")
    st.markdown("Click any button below to test sample reviews:")
    
    sample_1 = "The build quality is incredible and battery life easily lasts two full days. Super happy!"
    sample_2 = "Horrible customer support, defective item, and refused refund. Total waste of money."
    sample_3 = "The device is decent for basic daily tasks, though the camera quality is average."
    
    if st.button("Sample 1: Positive Review"):
        st.session_state["review_input"] = sample_1
    if st.button("Sample 2: Negative Review"):
        st.session_state["review_input"] = sample_2
    if st.button("Sample 3: Mixed Review"):
        st.session_state["review_input"] = sample_3

# ---------------------------------------------------------
# Header & Navigation Tabs
# ---------------------------------------------------------
st.title("🚀 Review Sentiment Checker & Cloud Computing")
st.markdown("A cloud-hosted sentiment analysis web application with cloud computing architecture documentation.")

tab1, tab2, tab3 = st.tabs([
    "🔍 Mini Project: Sentiment Checker", 
    "☁️ IaaS vs PaaS vs SaaS & Streamlit", 
    "📊 AWS vs Azure vs GCP Free Tier"
])

# =========================================================
# TAB 1: Sentiment Checker
# =========================================================
with tab1:
    st.header("Movie & Product Review Sentiment Classifier")
    st.markdown("Enter any product, movie, or service review below to predict whether it is **Positive** or **Negative**.")

    default_text = st.session_state.get(
        "review_input", 
        "This product exceeded all my expectations! Outstanding build quality and fast shipping."
    )

    review_text = st.text_area(
        "Enter your review here:", 
        value=default_text, 
        height=130,
        placeholder="Type or paste a review..."
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("Clear Input"):
            st.session_state["review_input"] = ""
            st.rerun()

    if analyze_clicked or review_text:
        if review_text.strip():
            prediction = model.predict([review_text])[0]
            probabilities = model.predict_proba([review_text])[0]
            classes = list(model.classes_)
            
            neg_idx = classes.index("Negative")
            pos_idx = classes.index("Positive")
            pos_prob = probabilities[pos_idx] * 100
            neg_prob = probabilities[neg_idx] * 100

            st.markdown("---")
            st.subheader("Analysis Results")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if prediction == "Positive":
                    st.success(f"### 🎉 Predicted Sentiment: **{prediction}**")
                    st.metric(label="Positive Confidence", value=f"{pos_prob:.1f}%")
                else:
                    st.error(f"### ⚠️ Predicted Sentiment: **{prediction}**")
                    st.metric(label="Negative Confidence", value=f"{neg_prob:.1f}%")

            with res_col2:
                st.markdown("**Confidence Breakdown:**")
                st.write(f"🟢 **Positive:** {pos_prob:.1f}%")
                st.progress(float(pos_prob / 100))
                st.write(f"🔴 **Negative:** {neg_prob:.1f}%")
                st.progress(float(neg_prob / 100))
        else:
            st.warning("Please type or select a review to analyze.")

# =========================================================
# TAB 2: IaaS vs PaaS vs SaaS & Why Streamlit Cloud is PaaS
# =========================================================
with tab2:
    st.header("Cloud Service Models & Platform Classification")
    
    st.subheader("1. Three Lines on IaaS vs PaaS vs SaaS:")
    st.info("""
    1. **IaaS (Infrastructure as a Service):** Delivers fundamental on-demand computing infrastructure—such as virtual servers, physical machines, networking, and block storage—where users manage the operating system, middleware, and application runtime (e.g., AWS EC2, Google Compute Engine, Azure VMs).
    2. **PaaS (Platform as a Service):** Delivers a managed cloud development and deployment environment where the cloud provider manages servers, networking, operating system, and language runtime, allowing developers to focus solely on writing and deploying application code (e.g., Streamlit Community Cloud, Heroku, AWS Elastic Beanstalk).
    3. **SaaS (Software as a Service):** Delivers complete, fully functional software applications accessible over the web, where the vendor manages everything end-to-end, requiring zero code or infrastructure setup from end users (e.g., Google Workspace, Microsoft 365, Netflix).
    """)

    st.subheader("2. Why Streamlit Community Cloud is PaaS:")
    st.success("""
    **Why Streamlit Cloud is PaaS:**  
    Streamlit Community Cloud is classified as **Platform as a Service (PaaS)** because it abstracts away all infrastructure management. As developers, we do not configure virtual machines, install Linux distributions, configure Nginx/Apache reverse proxies, or manage SSL certificates. Instead, Streamlit Cloud pulls our Python repository from GitHub, provisions the required containerized Python runtime environment, installs the dependencies specified in `requirements.txt`, and serves the live web application on a public URL automatically.
    """)

# =========================================================
# TAB 3: AWS vs Azure vs GCP Free Tier Comparison Table
# =========================================================
with tab3:
    st.header("Cloud Service Providers: Free Tier Comparison")
    st.markdown("Comparison of what the top 3 public cloud providers offer in their free tiers:")

    cloud_data = [
        {
            "Cloud Provider": "Amazon Web Services (AWS)",
            "Always Free Services": "AWS Lambda (1 Million requests/month), Amazon DynamoDB (25 GB NoSQL storage), Amazon SNS (1 Million publishes/month), Amazon SQS (1 Million requests/month).",
            "12 Months Free / Trial Credits": "750 hours/month of Amazon EC2 (t2.micro / t3.micro Linux/Windows), 5 GB Amazon S3 Standard Cloud Storage, 750 hours/month of Amazon RDS database."
        },
        {
            "Cloud Provider": "Microsoft Azure",
            "Always Free Services": "Azure Cosmos DB (1,000 RU/s throughput + 25 GB storage), Azure App Service (up to 10 web/API apps), Azure Functions (1 Million executions/month), Azure Event Grid.",
            "12 Months Free / Trial Credits": "$200 free credit for the first 30 days + 12 months free access to 750 hours of B1s Burstable VM, 5 GB Blob Storage, and 250 GB Azure SQL Database."
        },
        {
            "Cloud Provider": "Google Cloud Platform (GCP)",
            "Always Free Services": "1 e2-micro VM instance per month (US regions), Cloud Functions (2 Million invocations/month), Cloud Run (2 Million requests/month), 5 GB Google Cloud Storage.",
            "12 Months Free / Trial Credits": "$300 free trial credits valid for 90 days across all Google Cloud services for new accounts."
        }
    ]

    df_cloud = pd.DataFrame(cloud_data)
    
    st.dataframe(
        df_cloud, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Cloud Provider": st.column_config.TextColumn("Cloud Provider", width="medium"),
            "Always Free Services": st.column_config.TextColumn("Always Free Services", width="large"),
            "12 Months Free / Trial Credits": st.column_config.TextColumn("12-Month Free / Trial Offerings", width="large")
        }
    )

    st.markdown("---")
    st.caption("Submitted for Cloud Computing Mini Project 3 — Streamlit Community Cloud Deployment.")
