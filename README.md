# VeriNews AI — Hybrid Fake News Detection

An academic BCA project prototype combining **TF-IDF + Logistic Regression** with an optional **Google Gemini LLM review**, a dark Streamlit dashboard, and MySQL analysis history.

> **Important:** `data/sample_news.csv` is a tiny synthetic smoke-test dataset. It is included only so the app runs immediately. It is NOT suitable for research conclusions, accuracy claims, or real-world news detection. Train with a large, properly labeled and independently evaluated dataset before your final demonstration.

## 1. Features

- Dark dashboard and navigation
- ML prediction with class-score visualization
- Gemini LLM explanation, textual warning signs, and verification checklist
- Hybrid result: agreement, conflicting signals, or needs verification
- MySQL persistence and history table
- CSV export of stored history
- Works without Gemini (ML-only mode); works without MySQL (history is disabled)

## 2. Requirements

- Python 3.10–3.12 recommended
- MySQL 8.x (local or hosted)
- Optional Gemini API key from [Google AI Studio](https://aistudio.google.com/)
- GitHub account for free Streamlit Community Cloud deployment

## 3. Run locally (Windows)

Open Command Prompt or PowerShell inside this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 4. Configure Gemini (optional)

1. Create an API key in [Google AI Studio](https://aistudio.google.com/).
2. Put it in `.env`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Do not commit `.env` or expose the key in frontend code or screenshots. Gemini availability and free-tier limits can change; check the current pricing and rate limits before relying on it.

## 5. Configure MySQL

1. Start MySQL Server and open MySQL Workbench or the MySQL CLI.
2. Run `schema.sql`.
3. Add the database credentials to `.env`:

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=verinews
```

4. Restart Streamlit. On the first successful analysis, the app also creates the `analyses` table if needed.

For a remote MySQL provider, use the hostname, port, username, password, and database name supplied by that provider. Never commit credentials to GitHub.

## 6. Train with a real dataset

Prepare a CSV with two columns:

```csv
text,label
"Full news article or headline text here",real
"Another article text here",fake
```

Use consistent labels. The default project expects `real` and `fake`. You can also use another pair of labels, but keep the labels consistent throughout the dataset.

Run:

```bash
python train_model.py --data data/your_dataset.csv
```

The trained model is saved to `models/fake_news_model.joblib`. Restart the app after training. The `models/` folder is intentionally excluded from Git; either train during setup/deployment or remove `models/` from `.gitignore` if you choose to commit the trained artifact and it is reasonably small.

### Dataset suggestions

- LIAR dataset: short political claims with truthfulness labels. It has multiple label classes, so decide and document a defensible mapping if converting it to binary labels.
- FakeNewsNet: research dataset with article and social-context information; check its license, terms, and data availability.
- Use a held-out test split and avoid duplicates/leakage between training and testing.

Do not merge labels carelessly. “False”, “pants-fire”, “half-true”, “mostly-true”, and “true” are not interchangeable categories. Explain any mapping in your report.

## 7. Deploy free with Streamlit Community Cloud

1. Create a GitHub repository and upload this project (do not upload `.env`, passwords, or private keys).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
4. In app settings, open **Advanced settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your_gemini_key"
GEMINI_MODEL = "gemini-2.5-flash"

MYSQL_HOST = "your_mysql_host"
MYSQL_PORT = "3306"
MYSQL_USER = "your_mysql_user"
MYSQL_PASSWORD = "your_mysql_password"
MYSQL_DATABASE = "verinews"
```

5. Click Deploy. Check the app logs if startup fails.

**Free-hosting caveats:** Streamlit Community Cloud offers free app hosting, but usage/resource limits can apply. A remote MySQL free tier is a separate service and its availability, storage, uptime, and connection policies may change. The application will still run in ML-only mode if MySQL or Gemini is not configured.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## 8. Project architecture

```text
User article
    |
    +--> TF-IDF vectorizer --> Logistic Regression --> ML label + scores
    |
    +--> Gemini LLM review --------------------------> LLM label + explanation
                                  |
                           Hybrid decision logic
                                  |
                         Streamlit results UI
                                  |
                           MySQL analyses table
```

## 9. Evaluation for your final-year report

Use a genuinely labeled dataset and report:

- Accuracy
- Precision, recall, and F1-score per class
- Confusion matrix
- Dataset size, source, class distribution, and preprocessing
- Train/test split method and random seed
- ML-only vs hybrid comparison
- LLM failure cases, API latency, and API cost/limit considerations

The LLM output is not a ground-truth label. For an honest comparison, evaluate it against a held-out labeled test set and disclose prompt/model version and evaluation date.

## 10. Viva explanation

**Why hybrid?** The ML model is fast, repeatable, and learns patterns from labeled examples. The LLM adds readable explanations and suggests verification steps. Comparing them exposes disagreements instead of hiding uncertainty.

**Why TF-IDF?** It converts words and word combinations into numerical features based on their importance across the training corpus.

**Why Logistic Regression?** It is a strong, interpretable baseline for sparse text-classification features and produces class scores.

**Why MySQL?** It stores analysis history in structured rows so records can be queried, filtered, and exported.

**Main limitation?** Neither model checks the claim against live authoritative sources in this version. The tool provides preliminary signals, not a definitive fact-check.
