# Multi-Warehouse Extension Guide: Beyond Snowflake

**Add support for BigQuery, Databricks, Redshift, PostgreSQL, and more to your AI Data Chatbot**

This guide shows how to extend the AI chatbot to work with any SQL-compatible data warehouse, enabling true multi-warehouse queries.

---

## Quick Comparison: Supported Warehouses

| Warehouse | Status | Setup Time | Cost | When to Use |
|---|---|---|---|---|
| **Snowflake** | ✅ Built-in | — | $5-50/mo | Enterprise data warehouse |
| **BigQuery** | 📋 Ready | 30 min | Free tier | Google Cloud, high-volume analytics |
| **Databricks** | 📋 Ready | 30 min | Free tier | ML + analytics unified platform |
| **AWS Redshift** | 📋 Ready | 30 min | ~$1/hr | AWS-native data warehouse |
| **PostgreSQL** | 📋 Ready | 15 min | Free | On-prem or self-hosted |
| **MySQL** | 📋 Ready | 15 min | Free | Self-hosted analytics |
| **DuckDB** | 📋 Ready | 10 min | Free | In-process, no server needed |
| **Presto/Trino** | 📋 Planned | 30 min | — | Federated queries across warehouses |

---

## Architecture: The Connector Pattern

The magic is in the **abstraction layer**:

```
┌─────────────────────────────────┐
│ LangGraph Pipeline (Generic)    │
│ - Intent extraction             │
│ - SQL generation                │
│ - Validation                    │
│ - Analysis & Reporting          │
└──────────────┬──────────────────┘
               │
        ┌──────▼──────┐
        │ Connector   │
        │ Interface   │
        └──────┬──────┘
         ┌─────┴─────────────────────┬──────────┬──────────┐
         │                           │          │          │
    ┌────▼─────┐          ┌──────────▼──┐  ┌────▼────┐ ┌──▼──────┐
    │Snowflake │          │  BigQuery   │  │Redshift │ │Postgres │
    └──────────┘          └─────────────┘  └─────────┘ └─────────┘
```

### The Connector Interface

Every connector implements this interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class DataWarehouseConnector(ABC):
    """Abstract base class for all data warehouse connectors"""
    
    @abstractmethod
    async def execute_query(
        self, 
        sql: str, 
        max_rows: int = 1000
    ) -> Dict[str, Any]:
        """Execute a SELECT query and return results"""
        pass
    
    @abstractmethod
    async def discover_metadata(
        self, 
        database: str = None,
        schema: str = None
    ) -> Dict[str, Any]:
        """Discover available tables and columns"""
        pass
    
    @abstractmethod
    async def list_databases(self) -> List[str]:
        """List all accessible databases"""
        pass
    
    @abstractmethod
    async def list_schemas(self, database: str) -> List[str]:
        """List schemas in a database"""
        pass
    
    @abstractmethod
    async def list_tables(self, database: str, schema: str) -> List[str]:
        """List tables in a schema"""
        pass
    
    @abstractmethod
    async def describe_table(
        self, 
        database: str, 
        schema: str, 
        table: str
    ) -> Dict[str, Any]:
        """Get column information for a table"""
        pass
```

---

## Adding BigQuery Support

### Step 1: Install Dependencies

```bash
pip install google-cloud-bigquery
```

### Step 2: Create the Connector

Create `tools/bigquery_connector.py`:

```python
import os
import json
from typing import Dict, List, Any
from google.cloud import bigquery
from google.oauth2 import service_account
from tenacity import retry, stop_after_attempt, wait_exponential

class BigQueryConnector:
    """Google BigQuery connector for the AI data agent"""
    
    def __init__(self):
        credentials_path = os.getenv("GCP_CREDENTIALS_PATH")
        project_id = os.getenv("GCP_PROJECT_ID")
        
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = bigquery.Client(
                project=project_id,
                credentials=credentials
            )
        else:
            self.client = bigquery.Client(project=project_id)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def execute_query(self, sql: str, max_rows: int = 1000) -> Dict[str, Any]:
        """Execute SQL query against BigQuery"""
        job_config = bigquery.QueryJobConfig(max_results=max_rows)
        query_job = self.client.query(sql, job_config=job_config)
        
        results = query_job.result()
        rows = [dict(row) for row in results]
        columns = [field.name for field in results.schema]
        
        return {
            "rows": rows,
            "columns": columns,
            "row_count": len(rows),
            "execution_time_ms": query_job.ended - query_job.started
        }
    
    async def discover_metadata(
        self, 
        database: str = None,
        schema: str = None
    ) -> Dict[str, Any]:
        """Discover metadata across project and datasets"""
        metadata = {}
        
        # If database (project) specified, limit to that
        if database:
            projects = [database]
        else:
            projects = [self.client.project]
        
        for project_id in projects:
            metadata[project_id] = {}
            
            # List all datasets (schemas) in project
            if schema:
                datasets = [schema]
            else:
                datasets = [d.dataset_id for d in self.client.list_datasets()]
            
            for dataset_id in datasets:
                dataset_ref = self.client.dataset(dataset_id, project=project_id)
                tables = self.client.list_tables(dataset_ref)
                
                metadata[project_id][dataset_id] = {}
                
                for table in tables:
                    table_obj = self.client.get_table(
                        f"{project_id}.{dataset_id}.{table.table_id}"
                    )
                    
                    columns = [
                        {
                            "name": field.name,
                            "type": field.field_type,
                            "nullable": field.mode != "REQUIRED"
                        }
                        for field in table_obj.schema
                    ]
                    
                    metadata[project_id][dataset_id][table.table_id] = columns
        
        return metadata
    
    async def list_databases(self) -> List[str]:
        """List projects (BigQuery projects act as databases)"""
        return [self.client.project]
    
    async def list_schemas(self, database: str) -> List[str]:
        """List datasets in a project"""
        return [d.dataset_id for d in self.client.list_datasets(project=database)]
    
    async def list_tables(self, database: str, schema: str) -> List[str]:
        """List tables in a dataset"""
        dataset_ref = self.client.dataset(schema, project=database)
        tables = self.client.list_tables(dataset_ref)
        return [t.table_id for t in tables]
    
    async def describe_table(
        self, 
        database: str, 
        schema: str, 
        table: str
    ) -> Dict[str, Any]:
        """Get column details for a table"""
        table_obj = self.client.get_table(f"{database}.{schema}.{table}")
        
        return {
            "table_name": table,
            "schema": schema,
            "database": database,
            "columns": [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "nullable": field.mode != "REQUIRED",
                    "description": field.description
                }
                for field in table_obj.schema
            ],
            "row_count": table_obj.num_rows
        }
```

### Step 3: Update Environment Configuration

Add to your `.env`:

```env
# BigQuery Configuration
WAREHOUSE_TYPE=bigquery
GCP_PROJECT_ID=your-gcp-project-id
GCP_CREDENTIALS_PATH=/path/to/service-account-key.json
```

To create service account credentials:
1. Go to GCP Console → Service Accounts
2. Create new service account
3. Grant "BigQuery Admin" role
4. Create JSON key
5. Download and save to your project

### Step 4: Update the MCP Server

Update `tools/mcp_server.py`:

```python
async def get_connector():
    warehouse_type = os.getenv("WAREHOUSE_TYPE", "snowflake").lower()
    
    if warehouse_type == "bigquery":
        from tools.bigquery_connector import BigQueryConnector
        return BigQueryConnector()
    elif warehouse_type == "snowflake":
        from tools.snowflake_connector import SnowflakeConnector
        return SnowflakeConnector()
    else:
        raise ValueError(f"Unsupported warehouse: {warehouse_type}")
```

### Step 5: Test It

```bash
# Update .env to use BigQuery
# WAREHOUSE_TYPE=bigquery

# Restart server
uvicorn app:app --reload

# Test
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 10 most visited pages in my Google Analytics data?"}'
```

---

## Adding AWS Redshift Support

### Step 1: Install Dependencies

```bash
pip install redshift-connector
```

### Step 2: Create the Connector

Create `tools/redshift_connector.py`:

```python
import os
import redshift_connector
from typing import Dict, List, Any
from tenacity import retry, stop_after_attempt, wait_exponential

class RedshiftConnector:
    """AWS Redshift connector for the AI data agent"""
    
    def __init__(self):
        self.host = os.getenv("REDSHIFT_HOST")
        self.port = int(os.getenv("REDSHIFT_PORT", 5439))
        self.database = os.getenv("REDSHIFT_DATABASE")
        self.user = os.getenv("REDSHIFT_USER")
        self.password = os.getenv("REDSHIFT_PASSWORD")
        self.region = os.getenv("REDSHIFT_REGION", "us-east-1")
    
    def _get_connection(self):
        """Create a new connection"""
        return redshift_connector.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            region=self.region
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def execute_query(self, sql: str, max_rows: int = 1000) -> Dict[str, Any]:
        """Execute SQL query against Redshift"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"{sql} LIMIT {max_rows}")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            return {
                "rows": [dict(zip(columns, row)) for row in rows],
                "columns": columns,
                "row_count": len(rows)
            }
        finally:
            cursor.close()
            conn.close()
    
    async def discover_metadata(
        self, 
        database: str = None,
        schema: str = None
    ) -> Dict[str, Any]:
        """Discover metadata"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        metadata = {self.database: {}}
        
        try:
            # Get schemas
            cursor.execute("""
                SELECT schema_name FROM information_schema.schemata
                WHERE catalog_name = %s
                ORDER BY schema_name
            """, (self.database,))
            
            schemas = [row[0] for row in cursor.fetchall()]
            if schema:
                schemas = [s for s in schemas if s == schema]
            
            for schema_name in schemas:
                metadata[self.database][schema_name] = {}
                
                # Get tables in schema
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = %s AND table_catalog = %s
                    ORDER BY table_name
                """, (schema_name, self.database))
                
                tables = [row[0] for row in cursor.fetchall()]
                
                for table_name in tables:
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position
                    """, (schema_name, table_name))
                    
                    columns = [
                        {
                            "name": col[0],
                            "type": col[1],
                            "nullable": col[2] == "YES"
                        }
                        for col in cursor.fetchall()
                    ]
                    
                    metadata[self.database][schema_name][table_name] = columns
        
        finally:
            cursor.close()
            conn.close()
        
        return metadata
    
    async def list_databases(self) -> List[str]:
        """List databases"""
        return [self.database]
    
    async def list_schemas(self, database: str) -> List[str]:
        """List schemas"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT schema_name FROM information_schema.schemata
                WHERE catalog_name = %s
                ORDER BY schema_name
            """, (database,))
            
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()
    
    async def list_tables(self, database: str, schema: str) -> List[str]:
        """List tables in schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = %s AND table_catalog = %s
                ORDER BY table_name
            """, (schema, database))
            
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()
    
    async def describe_table(
        self, 
        database: str, 
        schema: str, 
        table: str
    ) -> Dict[str, Any]:
        """Get column details"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """, (schema, table))
            
            columns = [
                {
                    "name": col[0],
                    "type": col[1],
                    "nullable": col[2] == "YES"
                }
                for col in cursor.fetchall()
            ]
            
            return {
                "table_name": table,
                "schema": schema,
                "database": database,
                "columns": columns
            }
        finally:
            cursor.close()
            conn.close()
```

### Step 3: Environment Configuration

```env
WAREHOUSE_TYPE=redshift
REDSHIFT_HOST=my-cluster.123456.us-east-1.redshift.amazonaws.com
REDSHIFT_PORT=5439
REDSHIFT_DATABASE=analytics
REDSHIFT_USER=analyst
REDSHIFT_PASSWORD=YourSecurePassword
REDSHIFT_REGION=us-east-1
```

---

## Adding PostgreSQL/MySQL Support

For self-hosted PostgreSQL or MySQL, use the simple SQL connector:

### Step 1: Install Dependencies

```bash
# For PostgreSQL
pip install psycopg2-binary

# For MySQL
pip install mysql-connector-python
```

### Step 2: Create Generic SQL Connector

Create `tools/generic_sql_connector.py`:

```python
import os
import psycopg2
from typing import Dict, List, Any

class PostgreSQLConnector:
    """PostgreSQL connector"""
    
    def __init__(self):
        self.conn_params = {
            "host": os.getenv("PG_HOST"),
            "port": int(os.getenv("PG_PORT", 5432)),
            "database": os.getenv("PG_DATABASE"),
            "user": os.getenv("PG_USER"),
            "password": os.getenv("PG_PASSWORD")
        }
    
    async def execute_query(self, sql: str, max_rows: int = 1000) -> Dict[str, Any]:
        """Execute query"""
        conn = psycopg2.connect(**self.conn_params)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"{sql} LIMIT {max_rows}")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            return {
                "rows": [dict(zip(columns, row)) for row in rows],
                "columns": columns,
                "row_count": len(rows)
            }
        finally:
            cursor.close()
            conn.close()
    
    # ... implement other methods similarly
```

### Step 3: Environment

```env
WAREHOUSE_TYPE=postgresql
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=analytics
PG_USER=analyst
PG_PASSWORD=password
```

---

## Switching Between Warehouses

Once you have multiple connectors set up, switching is one `.env` change:

```bash
# Currently using Snowflake
WAREHOUSE_TYPE=snowflake

# Switch to BigQuery
WAREHOUSE_TYPE=bigquery

# Switch to Redshift
WAREHOUSE_TYPE=redshift

# Switch to PostgreSQL
WAREHOUSE_TYPE=postgresql
```

Restart the server and it automatically uses the new connector. **No code changes required.**

---

## Multi-Warehouse Queries (Advanced)

Future enhancement: query across multiple warehouses in one request:

```python
# Not yet implemented, but the architecture supports it

async def multi_warehouse_query(
    question: str,
    warehouses: List[str]  # ["snowflake", "bigquery"]
) -> Dict[str, Any]:
    """Query across multiple warehouses"""
    
    results = {}
    for warehouse in warehouses:
        connector = get_connector(warehouse)
        results[warehouse] = await connector.execute_query(sql)
    
    # Join results
    return merge_results(results)
```

---

## Troubleshooting

### "Could not connect to [warehouse]"
- Verify credentials in `.env`
- Test connection directly with warehouse CLI tool
- Check network access/firewall rules

### "Table not found"
- Verify table name and schema
- Check connector's schema discovery works
- Run `list_tables()` to see available tables

### "Query too slow"
- Add indexes on filtered/joined columns
- Reduce data with WHERE clauses
- Check warehouse scaling settings

---

## Contributing a New Connector

Want to add support for a new warehouse? Follow the pattern:

1. Create `tools/[warehouse]_connector.py`
2. Implement `DataWarehouseConnector` interface
3. Add to warehouse type selector in `mcp_server.py`
4. Add `.env` example to `.env.example`
5. Submit a PR!

---

## Next Steps

- Start with Snowflake (included)
- Add one additional warehouse (BigQuery recommended)
- Test queries against both
- Consider federated queries (coming soon)
