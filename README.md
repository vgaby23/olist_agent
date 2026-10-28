# 🛍️ Olist Virtual Assistant

An AI chat assistant for an online marketplace. You ask a question in plain language, for example *"Which cheap toys get the best reviews?"*, and it answers using real e-commerce data: about 100,000 orders from [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), a Brazilian marketplace.

The assistant serves three kinds of users, and each one sees different data:

| | Who | What they can ask about |
|---|---|---|
| 🛒 | **Customer** (anonymous shopper) | Good products, good sellers, what other shoppers say, delivery times |
| 🏪 | **Seller** (shop owner) | Their own sales, orders, deliveries and reviews, compared with their category |
| 🏢 | **Company** (Olist staff) | Everything on the platform: revenue, top sellers, top customers, reviews |

---

## Contents

1. [How it works](#1-how-it-works)
2. [What you can ask](#2-what-you-can-ask)
3. [Who can see what](#3-who-can-see-what)
4. [Setup guide](#4-setup-guide)
5. [Using the app](#5-using-the-app)
6. [Running the tests](#6-running-the-tests)
7. [Project structure](#7-project-structure)
8. [Troubleshooting](#8-troubleshooting)
9. [Limitations](#9-limitations)
10. [Team and dataset](#10-team-and-dataset)

---

## 1. How it works

The app uses three AI **agents**. An agent is a language model (here OpenAI's `gpt-6-luna`) that can use tools, such as running a database query, to answer a question.

```
                 You ask a question
                         │
                         ▼
          ┌──────────────────────────────┐
          │       Supervisor agent       │
          │ 1. Is this question allowed  │
          │    for your role?            │
          │ 2. Which specialist(s) can   │
          │    answer it?                │
          └───────┬──────────────┬───────┘
                  │              │
                  ▼              ▼
     ┌──────────────────┐  ┌──────────────────┐
     │    SQL agent     │  │   Review agent   │
     │  "the numbers"   │  │  "the opinions"  │
     │                  │  │      (RAG)       │
     └────────┬─────────┘  └────────┬─────────┘
              ▼                     ▼
       SQLite database        Qdrant vector DB
       orders, prices,        ~35,000 customer
       ratings, delivery      review comments
```

- **SQL agent:** writes and runs database queries for anything you can count or rank, such as revenue per month, average rating, or the top 10 sellers.
- **Review agent:** searches customer review comments **by meaning** to find complaints, praise and reasons for low ratings. This technique is called **RAG** (Retrieval-Augmented Generation): first *retrieve* the relevant reviews, then let the model *write* an answer from them.
- **Supervisor:** routes your question to one or both specialists and combines their answers. A question like *"Why did my August orders get low ratings?"* uses both: SQL finds the orders, and the review agent reads their reviews.

### Tech stack

| Part | Tool |
|---|---|
| Web interface | [Streamlit](https://streamlit.io) |
| Agents | [LangChain](https://www.langchain.com) / LangGraph |
| Language model and embeddings | OpenAI (`gpt-6-luna`, `text-embedding-3-small`) |
| Numbers database | SQLite (opened read-only) |
| Review search database | [Qdrant](https://qdrant.tech) (a vector database) |

---

## 2. What you can ask

Each role starts with suggestion buttons on an empty chat. Here are typical questions.

### 🛒 Customer

| I want to… | Example question |
|---|---|
| Find good sellers | "What are the 5 best-rated sellers for health & beauty?" |
| Get product ideas | "Which cheap toys get the best reviews?" |
| Read what shoppers think | "What do reviewers complain about most in bed_bath_table?" |
| Check a specific seller | "What do reviewers say about the top-ranked seller for watches?" |
| Know delivery times | "What's the average delivery time to RJ?" |

You can also **upload a product photo** (the assistant guesses the category) or **send a voice message** in English, Indonesian or Portuguese (it is turned into text). Every role can use photo and voice.

### 🏪 Seller

| I want to… | Example question |
|---|---|
| Track my sales | "How has my monthly revenue changed over the last 12 months?" |
| Understand bad reviews | "What are the main complaints in my 1-2 star reviews?" |
| Compare with my market | "How does my average rating compare to the benchmark for my category?" |
| Know my customers | "Which states do most of my customers come from?" |
| Check my deliveries | "What's my average delivery time and late-delivery rate?" |

### 🏢 Company

| I want to… | Example question |
|---|---|
| Follow revenue | "What was the total GMV per month in 2018?" |
| Find weak categories | "Which categories have the lowest average review scores?" |
| Reward top sellers | "Top 10 sellers by revenue in health_beauty" |
| Fix delivery problems | "Which states have the slowest deliveries and the most late orders?" |
| Understand payments | "What payment methods do customers use, and how many installments on average?" |

---

## 3. Who can see what

Each role only sees the data it is allowed to see. If you ask for something outside your role, the assistant says no and tells you which rule applies.

| Data | 🛒 Customer | 🏪 Seller | 🏢 Company |
|---|---|---|---|
| Category statistics | ranks and ranges | exact numbers | exact numbers |
| Product and seller statistics | ranks and ranges | ranks and ranges | exact numbers |
| Delivery times per state | ✅ | ✅ | ✅ |
| Review comments | all (per product or seller, not per order) | own orders only | all |
| Orders, payments, customers | ❌ | own only (buyers get nicknames) | all |

**"Ranks and ranges"** means you see, for example, *sales rank #3* and *price R$50–99* instead of exact sales numbers or revenue.

The rules are protected in **two layers**, so a clever question can't get around them:

1. **The supervisor checks the question first** and refuses anything outside the role's rules. The rules have ids C1–C7 for customers and S1–S7 for sellers, and they are all in [`agent/policy.py`](agent/policy.py).
2. **The code enforces the rules** even if the AI makes a mistake. Database queries only run against views made for that role ([`sql_tools.py`](dataset/preprocessing/numeric_data/sql_tools.py)). Review searches for a seller are always locked to their own seller ID ([`agent/rag_agent.py`](agent/rag_agent.py)).

---

## 4. Setup guide

Follow these steps in order. Steps 4 and 5 only need to run **once**.

### What you need

- **Python 3.13** ([download](https://www.python.org/downloads/))
- **Git** ([download](https://git-scm.com/downloads))
- An **OpenAI API key** ([get one](https://platform.openai.com/api-keys)). Using the app costs a small amount per question.
- A **Qdrant** database. The free tier of [Qdrant Cloud](https://cloud.qdrant.io) is enough. You need its **URL** and **API key**.

### Step 1: Download the code

```bash
git clone https://github.com/vgaby23/olist_agent.git
```

```bash
cd olist_agent
```

All the commands below are run from this `olist_agent` folder.

### Step 2: Create a virtual environment and install packages

A virtual environment keeps this project's packages separate from the rest of your computer.

```bash
python -m venv .venv
```

Turn it on. On macOS / Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

You should now see `(.venv)` at the start of your terminal line. Install the packages:

```bash
pip install -r requirements.txt
```

### Step 3: Add your API keys

Copy the example settings file:

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in these three lines:

```ini
OPENAI_API_KEY=sk-...your key...
QDRANT_URL=https://your-cluster-id.qdrant.tech:6333
QDRANT_API_KEY=...your Qdrant key...
```

> ⚠️ Never upload `.env` to GitHub. It is already listed in `.gitignore`.

<details>
<summary>Optional settings</summary>

| Setting | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `gpt-6-luna` | The OpenAI model that writes the answers. |
| `VISION_MODEL` | `gpt-6-luna` | The OpenAI model that recognises products in photos. |
| `STT_MODEL` | `whisper-1` | The OpenAI model that turns voice messages into text |
| `STT_LANGUAGE` | `auto` | The spoken language. `auto` detects it. Set a code such as `id` or `en` to force one language. |
| `STT_FALLBACK_LANGUAGE` | `id` | If auto-detection hears a language other than English, Indonesian or Portuguese (for example, Indonesian mistaken for Thai), the audio is transcribed again in this language. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | The model that turns reviews into vectors. If you change it, redo step 5. |
| `COLLECTION_NAME` | `olist_reviews` | The name of the review collection in Qdrant |
| `OLIST_DB_PATH` | `dataset/olist.db` | Where the SQLite database is |

</details>

### Step 4: Build the database

This reads the raw CSV files in `dataset/raw_data/`, cleans them, and saves `dataset/olist.db` (about 115 MB). It only takes a few seconds.

```bash
python dataset/preprocessing/numeric_data/clean_olist.py
```

<details>
<summary>What does the cleaning do?</summary>

- Revenue only counts **delivered** orders. Other orders are kept but flagged.
- The analysis period is **January 2017 – August 2018**. Data outside it is incomplete.
- When an order has several reviews, the first one is used.
- Products without a category are labelled `uncategorized`.
- Invalid payment records are fixed or removed, and seller city names are cleaned.

</details>

### Step 5: Upload the reviews to Qdrant

First do a **dry run**. It prepares the reviews and prints a few examples, but uploads nothing and costs nothing:

```bash
python -m dataset.preprocessing.vector_data.text_ingest --data-dir dataset/raw_data --dry-run
```

You should see `Dry run finished. 34,732 documents, 73 categories.` Now do the real upload. It turns each review into a vector with OpenAI, so it takes a while and costs a little:

```bash
python -m dataset.preprocessing.vector_data.text_ingest --data-dir dataset/raw_data
```

> 💡 To test with a small sample first, add `--limit 500` to the command above.

Check that the upload worked:

```bash
python -m dataset.preprocessing.vector_data.check_upload
```

### Step 6: Start the app

The REST API and Streamlit interface run as separate processes. The API owns the agents
and database; Streamlit sends each chat turn to it over HTTP. Set `API_URL` in Streamlit
secrets to the API base URL (`http://localhost:8000` locally).

```bash
uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
```

In another terminal:

```bash
streamlit run streamlit_app.py
```

The interface opens at http://localhost:8501. The API health check is at
http://localhost:8000/health, and interactive API docs are at http://localhost:8000/docs.
For Streamlit Community Cloud, use `streamlit_app.py` as the app entry point and configure `API_URL`
in the app's Secrets settings. The API does not require a bearer token; protect a deployed
API with platform-level access controls if it should not be publicly accessible.
Press `Ctrl + C` in each terminal to stop the local processes.

The current role/password login is a demo login, not production authentication. For a
production launch, replace it with an identity provider and authorization linked to the
authenticated user.

---

## 5. Using the app

### Logging in

Pick a role in **"I am a"** and enter its password. The password is the role name plus `123`.

| Role | Password | Extra |
|---|---|---|
| 🛒 Customer | `customer123` | none, shoppers are anonymous |
| 🏪 Seller | `seller123` | a **Seller ID** (required, so a seller only ever sees their own shop). Click **"Use sample seller ID"** to fill in the seller with the most orders. |
| 🏢 Company | `company123` | none |

> These are demo passwords for testing only.

### Chat features

| Feature | Where |
|---|---|
| 💬 **Suggestion buttons** | On an empty chat, click one to ask it |
| 🖼️ **Photo** / 🎤 **voice** questions | The attach and microphone buttons in the message box. Speak in English, Indonesian or Portuguese: the language is detected automatically. |
| 📈 **Charts and tables** | Under answers that come from the database, with a CSV download |
| 🔢 **Token usage** | Under each answer: how many tokens (and so how much cost) it used |
| ➕ **New chat** | Sidebar. Your previous chat is saved in **Conversation history** (up to 20 per login, until you close the browser tab). |
| ⬇️ **Download chat** | Sidebar. Saves the conversation as a Markdown file. |
| 🔄 **Retry** | Appears if a question fails |

An answer usually takes **20–40 seconds**, because the agents may run several queries and searches.

---

## 6. Running the tests

These tests run **offline**: they don't call OpenAI or Qdrant, so they're free. They check that the role rules hold in code.

| Test | Command |
|---|---|
| Database security (role rules, blocked queries) | `python -m dataset.preprocessing.numeric_data.test_sql_security` |
| Review search security | `python -m dataset.preprocessing.vector_data.test_rag_security` |
| Photo and voice input | `python -m pytest test_folder/test_input_preprocessing.py` |
| API auth, role scope and request validation | `python -m pytest test_folder/test_api.py` |

The security tests need `dataset/olist.db` from step 4.

There's also an evaluation that sends allowed and forbidden questions to the real supervisor. It **does call OpenAI**, so it costs a little:

```bash
python -m test_folder.eval_scope_policy
```

---

## 7. Project structure

```
olist_agent/
├── streamlit_app.py               ← Streamlit interface entry point
├── main.py                        FastAPI REST API entry point
├── interface.py                   Streamlit UI and API request flow
├── services/
│   └── chat_service.py            agent orchestration called by the API
├── schemas/
│   └── chat.py                    validated chat request and response models
│
├── agent/                         the AI agents
│   ├── supervisor_agent.py        routes questions, checks the rules
│   ├── sql_agent.py               answers with database queries
│   ├── rag_agent.py               answers from review comments
│   ├── policy.py                  who may see what (rules C1–C7, S1–S7)
│   ├── category_resolver.py       "make up" → health_beauty
│   ├── language.py, llm.py        answer language, model settings
│
├── utils/                         shared helpers and API client
│   ├── auth.py                    login page
│   ├── api_client.py              requests the assistant API
│   ├── chat_config.py             suggestion questions per role
│   ├── chat_display.py            tables, charts, token usage
│   ├── chat_history.py            new chat, history, download
│   ├── chat_runtime.py            "agent is thinking…" labels
│   ├── input_preprocessing.py     photo → category, voice → text
│   └── categories.json            list of product categories
│
├── dataset/
│   ├── raw_data/                  original Olist CSV files
│   └── preprocessing/
│       ├── numeric_data/          build olist.db + safe query tools
│       └── vector_data/           upload reviews to Qdrant + search
│
├── conn/connection.py             connection settings
├── test_folder/                   evaluation and tuning scripts
├── requirements.txt               Python packages
└── .env.example                   template for your API keys
```

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `dataset/olist.db not found` | Run step 4 to build the database. |
| `ModuleNotFoundError: No module named 'dataset'` (or `agent`) | Run commands from the `olist_agent` folder, and use `python -m …` where this guide does. |
| `ModuleNotFoundError` for another package | The virtual environment is off. Activate it (step 2) and run `pip install -r requirements.txt`. |
| OpenAI authentication error | Check `OPENAI_API_KEY` in `.env`, then restart the app. |
| Review questions fail or find nothing | Check `QDRANT_URL` and `QDRANT_API_KEY`, then run `check_upload` (step 5). |
| `Unknown seller id` at login | The Seller ID must exist in the data. Use **"Use sample seller ID"**. |
| `Enter your seller ID to continue` | Sellers can't log in without a Seller ID. Fill it in, or click **"Use sample seller ID"**. |
| A voice message is transcribed in the wrong language | Set `STT_LANGUAGE` in `.env` to your language code, for example `en` or `id`, then restart the app. |
| An answer takes a long time | Normal. Complex questions can take up to a minute. |

---

## 9. Limitations

- **Products have no names** in the dataset, only an id and a category, so answers talk about categories and product ids.
- **Data covers 2016–2018.** The analysis uses Jan 2017 – Aug 2018, so "2018" means January–August only.
- **Customers are anonymous**, so there's no "my order history" for shoppers.
- Chat history is kept only while the browser tab is open.

**Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) on Kaggle, licensed CC BY-NC-SA 4.0.
