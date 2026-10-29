# ☁️ Cloud Computing Mini Project: Publish a Sentiment App Online

This project is a complete submission for the **Cloud Computing Mini Project**, demonstrating the development and cloud deployment of a machine learning web application using **Platform as a Service (PaaS)**.

---

## 📌 Deliverable 1: Live Web App (Streamlit Community Cloud)

- **Application File:** `app.py`
- **Deployment Platform:** Streamlit Community Cloud (PaaS)
- **Features:**
  - Real-time sentiment classification (Positive / Negative) with confidence scores.
  - Interactive test samples for quick evaluation.
  - Dedicated sections for Cloud Computing Theory and Free Tier Comparisons.

---

## 📌 Deliverable 2: Cloud Service Models (IaaS vs PaaS vs SaaS)

1. **IaaS (Infrastructure as a Service):** Delivers fundamental on-demand computing infrastructure—such as virtual servers, physical machines, networking, and block storage—where users manage the operating system, middleware, and application runtime (e.g., AWS EC2, Google Compute Engine, Azure VMs).
2. **PaaS (Platform as a Service):** Delivers a managed cloud development and deployment environment where the cloud provider manages servers, networking, operating system, and language runtime, allowing developers to focus solely on writing and deploying application code (e.g., Streamlit Community Cloud, Heroku, AWS Elastic Beanstalk).
3. **SaaS (Software as a Service):** Delivers complete, fully functional software applications accessible over the web, where the vendor manages everything end-to-end, requiring zero code or infrastructure setup from end users (e.g., Google Workspace, Microsoft 365, Netflix).

### Why Streamlit Community Cloud is PaaS:
Streamlit Community Cloud is classified as **Platform as a Service (PaaS)** because it completely abstracts server provisioning and infrastructure maintenance. Developers simply push Python code (`app.py` and `requirements.txt`) to a GitHub repository. Streamlit Cloud automatically handles container provisioning, operating system updates, Python environment setup, package installation, domain routing, and HTTPS/SSL certification.

---

## 📌 Deliverable 3: Cloud Free Tier Comparison (AWS vs. Azure vs. Google Cloud)

| Cloud Provider | Always Free Offerings | 12 Months Free / Trial Credits |
|---|---|---|
| **Amazon Web Services (AWS)** | AWS Lambda (1M requests/mo), Amazon DynamoDB (25 GB NoSQL storage), Amazon SNS (1M publishes/mo), Amazon SQS (1M requests/mo). | 750 hours/month of Amazon EC2 (t2.micro / t3.micro), 5 GB Amazon S3 Standard Storage, 750 hours/month of Amazon RDS database. |
| **Microsoft Azure** | Azure Cosmos DB (1,000 RU/s + 25 GB storage), Azure App Service (up to 10 web/API apps), Azure Functions (1M executions/mo). | $200 free credit (first 30 days) + 12 months free access to 750 hrs B1s Burstable VM, 5 GB Blob Storage, and 250 GB Azure SQL Database. |
| **Google Cloud Platform (GCP)** | 1 e2-micro VM instance per month (US regions), Cloud Functions (2M invocations/mo), Cloud Run (2M requests/mo), 5 GB Google Cloud Storage. | $300 free trial credits valid for 90 days across all Google Cloud services for new accounts. |

---

## 🚀 How to Run Locally

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open your browser at `http://localhost:8501`.
