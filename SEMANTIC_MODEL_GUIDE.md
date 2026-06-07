# The Semantic Model: How AI Understands Your Data
## Building a Business Dictionary for Your AI Data Analyst

---

## TL;DR

The **semantic model** is a YAML file that teaches your AI agent your business language. Instead of the LLM guessing which tables to use, it learns:
- What each table and column means in business terms
- How to find the right data (synonyms)
- What operations make sense (sum revenue? no, sum REGION?)
- How tables connect (which JOIN conditions are valid?)

**Generate it once:**
```bash
python tools/generate_semantic_model.py --database DB --schema SCHEMA --output semantic_model.yml
```

**Set it and forget it:**
```env
SEMANTIC_MODEL_PATH=semantic_model.yml
```

Now every AI query uses your business context automatically.

---

## The Problem: Garbage In, Garbage Out

Imagine your AI chatbot gets asked: **"What was our ARR by region last quarter?"**

### Without a semantic model, the LLM has to guess:

1. **What is ARR?**
   - Annual Recurring Revenue? 
   - Account Revenue Review?
   - Something else?
   - (Guess wrong → wrong column queried)

2. **Where does it live?**
   - SALES table? REVENUE table? CUSTOMER_METRICS table?
   - (Guess wrong → table not found)

3. **How do I get ARR by region?**
   - Do I SUM it? AVG it? COUNT it?
   - Which table has REGION? ORDERS? CUSTOMERS? ACCOUNTS?
   - How does REGION connect to revenue?
   - (Guess wrong → NULLs, missing rows, cartesian products)

The LLM will hallucinate reasonable-sounding SQL that **looks correct but runs against the wrong tables**. Your results will be confidently, completely wrong.

### With a semantic model, you answer those questions once:

```yaml
# ARR is defined here
- name: CUSTOMER_ANNUAL_VALUE
  description: "Annual recurring revenue per customer"
  synonyms: [ARR, annual_revenue, yearly_customer_value]
  columns:
    - name: ANNUAL_REVENUE
      description: "Total annual recurring revenue"
      synonyms: [ARR, recurring_revenue]
      is_measure: true  # ✅ Yes, sum this

# Region is defined here
- name: REGION
  description: "Geographic region (US, EMEA, APAC, LATAM)"
  synonyms: [geography, territory, area]
  is_measure: false  # ❌ Don't sum this
  is_dimension: true # ✅ Use this for grouping

# The connection is defined here
relationships:
  - name: customer_to_region
    left_table: CUSTOMER_ANNUAL_VALUE
    left_column: CUSTOMER_ID
    right_table: DIM_CUSTOMER
    right_column: CUSTOMER_ID
```

Now every query uses this pre-verified context. No guessing. Deterministic results.

---

## How the Semantic Model Flows Through Your Pipeline

When a user asks **"Top 3 regions by ARR"**, here's what happens:

### Node 1: Intent Extraction
```
User input: "Top 3 regions by ARR"
           ↓ (semantic model checks synonyms)
Normalized: intent=rank_by_metric, 
            metric=ANNUAL_REVENUE (found via synonym),
            dimension=REGION,
            top_n=3
```

The LLM recognizes "ARR" because you taught it: `synonyms: [ARR, annual_revenue, recurring_revenue]`

### Node 4: Dataset Selection
```
Available tables:
  - CUSTOMER_ANNUAL_VALUE ← Semantic model says: "Annual revenue per customer"
  - SALES_BY_REGION      ← Semantic model says: "Pre-aggregated by region"
  - ORDER_LINEITEM       ← (raw transactional data)

LLM picks: SALES_BY_REGION
(because description matches "revenue by region")
```

Without semantic model, LLM would pick ORDERS and try to self-join REGION somehow.

### Node 5: SQL Generation
```
Selected table: SALES_BY_REGION
Semantic relationships tell LLM:
  - REVENUE is a measure → use SUM()
  - REGION is a dimension → use GROUP BY

Generated SQL:
SELECT REGION, SUM(REVENUE) as total_arr
FROM SALES_BY_REGION
GROUP BY REGION
ORDER BY total_arr DESC
LIMIT 3
```

The LLM didn't have to guess the aggregation or GROUP BY. It read it from the semantic model.

---

## What Goes Into the Semantic Model

### Auto-Generated (by the tool)

Run this once, passing your Snowflake credentials:

```bash
python tools/generate_semantic_model.py \
    --database TPCH_DATA_PRODUCT \
    --schema GOLD \
    --output semantic_model.yml
```

The tool connects to Snowflake and extracts:

```yaml
version: "1.0"
name: "TPCH_DATA_PRODUCT.GOLD Semantic Model"
tables:
  - name: CUSTOMER_LIFETIME_VALUE
    database: TPCH_DATA_PRODUCT
    schema: GOLD
    description: "Customer profitability and lifetime metrics"
    synonyms: [customer_value, customer_ltv, customer_metrics]
    columns:
      - name: CUSTOMER_ID
        data_type: VARCHAR
        is_primary_key: true
        description: "Unique customer identifier"
        is_measure: false
        is_dimension: false
      - name: LIFETIME_NET_REVENUE
        data_type: NUMBER
        is_primary_key: false
        description: "Total net revenue across all orders"
        synonyms: [customer_revenue, lifetime_revenue, total_revenue]
        is_measure: true
        is_dimension: false
```

- **Table names and column names** ← Read from INFORMATION_SCHEMA
- **Data types** ← Read from schema metadata
- **Primary keys** ← Detected by naming convention (ends in _KEY, _ID)

### Human-Customized (by you, in the editor)

The tool does its best guess, but you should add business context:

```yaml
- name: SALES_BY_REGION_MONTH
  database: TPCH_DATA_PRODUCT
  schema: GOLD
  description: |  # ← You write this (tool generates "TODO: describe")
    Monthly revenue aggregated by geographic region.
    Used for regional KPI tracking and territory planning.
    Updated daily via ETL process.
  synonyms:       # ← You add business aliases
    - revenue_by_region
    - regional_sales
    - geography_sales
    - regional_performance
  columns:
    - name: SALES_MONTH
      data_type: DATE
      description: "First day of the month (YYYY-01-01 format)"
      synonyms: [month, period, sales_month]
      is_measure: false
      is_dimension: true
    - name: REVENUE
      data_type: NUMBER
      description: "Total net revenue (excluding discounts and returns)"
      synonyms: [sales, net_sales, total_sales]  # ← Key: your synonyms
      is_measure: true                            # ← Tells LLM to SUM()
      is_dimension: false
    - name: REGION
      data_type: VARCHAR
      description: "Geographic region: US, EMEA, APAC, LATAM, OTHER"
      synonyms: [geography, territory, area, location]
      is_measure: false
      is_dimension: true  # ← Tells LLM to GROUP BY
```

---

## Relationships: Teaching the LLM How Tables Connect

The tool auto-detects relationships by matching column names:

```yaml
relationships:
  - name: sales_to_customer
    description: "Each sale belongs to one customer"
    left_table: SALES_BY_REGION_MONTH  # The table with the foreign key
    left_column: REGION_ID
    right_table: DIM_REGION             # The dimension table
    right_column: REGION_ID
    join_type: LEFT
    cardinality: many_to_one
```

This tells the LLM:
- "If I need REGION details, LEFT JOIN DIM_REGION on REGION_ID"
- "Multiple sales can belong to one region" (cardinality)
- The join is safe and verified (not a hallucination)

---

## The Four-Step Setup

### Step 1: Generate the Base File

Prerequisites:
- `.env` file with Snowflake credentials
- `OPENAI_API_KEY` (optional; used to enrich descriptions with GPT-4o)

```bash
python tools/generate_semantic_model.py \
    --database TPCH_DATA_PRODUCT \
    --schema GOLD \
    --output semantic_model.yml \
    --include-tables "CUSTOMER_LIFETIME_VALUE,SALES_BY_REGION_MONTH,PRODUCT_SALES_SUMMARY"
```

Output:
```
Connecting to Snowflake (TPCH_DATA_PRODUCT.GOLD)...
Fetching tables...
  Found 5 table(s): CUSTOMER_LIFETIME_VALUE, SALES_BY_REGION_MONTH, ...
Fetching columns...
Detecting relationships...
  Detected 4 relationship(s)
Generating descriptions with GPT-4o (5 table(s))...
  Describing CUSTOMER_LIFETIME_VALUE... done
  Describing SALES_BY_REGION_MONTH... done
  ...

Done! Written to: semantic_model.yml
  Tables:        5
  Relationships: 4

Next: edit descriptions/synonyms, then set SEMANTIC_MODEL_PATH=semantic_model.yml in .env
```

### Step 2: Edit the YAML File

Open `semantic_model.yml` in your editor and improve:

**Add descriptions:**
```yaml
# Before (auto-generated):
description: "TODO: describe what this table contains"

# After (you wrote):
description: "Monthly revenue aggregated by geographic region and product line. Used for territory planning and revenue forecasting."
```

**Add synonyms your team uses:**
```yaml
# These let users ask questions in their own language
synonyms:
  - revenue_by_region
  - regional_sales
  - sales_by_territory
  - geographic_breakdown
```

**Mark measures vs. dimensions** (the tool often guesses right, but verify):
```yaml
- name: ORDER_COUNT
  is_measure: true   # ✅ Yes, SUM this
  is_dimension: false

- name: CUSTOMER_REGION
  is_measure: false  # ❌ Don't sum regions
  is_dimension: true # ✅ GROUP BY region
```

### Step 3: Enable in `.env`

```env
SEMANTIC_MODEL_PATH=semantic_model.yml
```

### Step 4: Restart and Test

```bash
# Kill the old server (Ctrl+C)
uvicorn app:app --reload
```

Test that the model loads:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What was our ARR by region?"}'
```

The semantic model is now active. Check the server logs — you should see:
```
semantic_model.loaded  name='TPCH_DATA_PRODUCT.GOLD Semantic Model'  tables=5  relationships=4
```

---

## Advanced: Multi-Database Semantic Models

If you have data across multiple databases, generate separate semantic models for each:

```bash
# Gold layer
python tools/generate_semantic_model.py \
    --database TPCH_DATA_PRODUCT \
    --schema GOLD \
    --output semantic_model__GOLD.yml

# Analytics layer
python tools/generate_semantic_model.py \
    --database ANALYTICS \
    --schema PUBLIC \
    --output semantic_model__ANALYTICS.yml

# Staging layer
python tools/generate_semantic_model.py \
    --database STAGING \
    --schema RAW \
    --output semantic_model__STAGING.yml
```

**Don't** set `SEMANTIC_MODEL_PATH`. Instead, the system auto-discovers all `*_semantic_model.yml` files and merges them:

```
semantic_model__GOLD.yml
semantic_model__ANALYTICS.yml
semantic_model__STAGING.yml
        ↓ (auto-merged on startup)
One merged model with all tables visible
```

Now the LLM can intelligently cross-reference tables across databases. Ask "Compare GOLD revenue to ANALYTICS projections" and it will JOIN across both databases using the pre-verified relationships.

---

## FAQ

### Q: Does the semantic model get regenerated every time I start the server?

**A:** No. It's loaded once from disk, then cached in memory. Changes to the YAML require a server restart.

### Q: Can I regenerate the semantic model without overwriting my edits?

**A:** The tool is smart: it generates a **new** file (e.g., `semantic_model__TPCH_DATA_PRODUCT__GOLD.yml`) so your edits to the old one are never lost. You can manually merge if needed.

To regenerate, use `--no-llm` to skip the slow GPT-4o calls:

```bash
python tools/generate_semantic_model.py \
    --database TPCH_DATA_PRODUCT \
    --schema GOLD \
    --output semantic_model.yml \
    --no-llm
```

### Q: What if the semantic model is wrong or outdated?

**A:** Edit the YAML by hand or regenerate. Either way:
1. Modify the YAML file
2. Restart the server
3. Test with a sample query

### Q: How does the semantic model affect query cost?

**A:** It **doesn't increase cost**. The LLM still makes the same API call to GPT-4o. But better semantic context → better table selection → fewer retries → lower cost overall.

### Q: Can I use the semantic model without Snowflake?

**A:** The **generation tool** requires Snowflake (it reads INFORMATION_SCHEMA). But you can write the YAML manually for any warehouse. Just follow the schema in [config/semantic_model_schema.py](./config/semantic_model_schema.py).

### Q: What if I don't set SEMANTIC_MODEL_PATH?

**A:** The system falls back to **dynamic discovery**: the `metadata_discovery_node` queries INFORMATION_SCHEMA on every request. This is:
- ✅ No pre-work required
- ❌ Slower (extra Snowflake roundtrip per query)
- ❌ No synonyms, relationships, or measure/dimension hints
- ❌ Less accurate table selection

For production, always set `SEMANTIC_MODEL_PATH`.

---

## What a Good Semantic Model Looks Like

Here's an example from a financial services company:

```yaml
version: "1.0"
name: "Finance Semantic Model"

tables:
  - name: TRANSACTION_FACT
    description: "All customer transactions (deposits, withdrawals, transfers)"
    synonyms: [transactions, movements, activity]
    columns:
      - name: TRANSACTION_AMOUNT
        description: "Amount in USD"
        synonyms: [amount, txn_amount, size]
        is_measure: true
        is_dimension: false
      - name: TRANSACTION_TYPE
        description: "DEPOSIT, WITHDRAWAL, TRANSFER, FEE"
        synonyms: [type, action, category]
        is_measure: false
        is_dimension: true
      - name: TRANSACTION_DATE
        description: "Date transaction posted to account"
        synonyms: [date, posted_date, effective_date]
        is_measure: false
        is_dimension: true

  - name: CUSTOMER_SEGMENT
    description: "Customer segment assignment (Premium, Standard, Budget)"
    synonyms: [segment, tier, customer_tier]
    columns:
      - name: SEGMENT_NAME
        description: "Customer segment: Premium, Standard, Budget"
        synonyms: [segment, tier, customer_type]
        is_measure: false
        is_dimension: true

relationships:
  - name: transaction_to_customer
    left_table: TRANSACTION_FACT
    left_column: CUSTOMER_ID
    right_table: CUSTOMER_SEGMENT
    right_column: CUSTOMER_ID
    cardinality: many_to_one
```

Now users can ask:
- "How much did Premium customers deposit last month?" ← Recognized "Premium" via synonym
- "What's our withdrawal rate by segment?" ← Recognized "withdrawal" → WITHDRAWAL type
- "Which customers transferred the most?" ← Correct JOIN: TRANSACTION_FACT → CUSTOMER_SEGMENT

---

## Summary

The semantic model is your **data dictionary for AI**. It:

1. **Encodes business logic once** — save time explaining the same table relationships over and over
2. **Teaches synonyms** — "ARR" = "Annual Revenue" = "recurring_revenue"
3. **Prevents hallucinations** — LLM doesn't guess; it reads verified relationships
4. **Scales across teams** — share one YAML, everyone gets better results
5. **Remains transparent** — all in plain YAML, version control friendly

Generate it, customize it, set it in `.env`, and forget about it. Your AI will thank you.

---

## Next Steps

1. **Generate your first model:**
   ```bash
   python tools/generate_semantic_model.py --database DB --schema SCHEMA --output semantic_model.yml
   ```

2. **Edit the descriptions and synonyms** to match your business language

3. **Set it in `.env`:**
   ```env
   SEMANTIC_MODEL_PATH=semantic_model.yml
   ```

4. **Restart the server and test**

5. **Share with your team** — commit the YAML to git, everyone benefits

Happy querying! 🚀
