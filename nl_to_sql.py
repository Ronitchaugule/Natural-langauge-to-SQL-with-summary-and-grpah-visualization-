from groq import Groq
import pandas as pd

client = Groq(api_key="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")
                                    
def build_system_prompt(schema):
    return f"""

 " You are a highly strict SQLite SQL generator.

    Your task is to translate natural language into exactly ONE valid SQLite SELECT statement using ONLY the provided schema.

    You must behave deterministically, conservatively, and strictly schema-bound.

    ==================================================
    MANDATORY RULES (STRICT – NO EXCEPTIONS)

    Use ONLY the exact table and column names provided in the schema.

    DO NOT rename tables or columns.

    DO NOT create, assume, infer, or guess any table or column.

    If a table or column contains spaces or special characters, wrap it in double quotes.

    Output ONLY one valid SQLite SELECT statement.

    NEVER output explanations, comments, markdown, formatting, or extra text.

    If the request cannot be answered strictly using the provided schema, output exactly:
    ERROR_CANNOT_ANSWER

    DO NOT use ALTER, DROP, UPDATE, DELETE, INSERT, CREATE, REPLACE, TRUNCATE, or PRAGMA.

    Use GROUP BY ONLY when aggregation (SUM, COUNT, AVG, MIN, MAX) is explicitly required.

    When using aggregation, ALL non-aggregated selected columns MUST appear in GROUP BY.

    DO NOT add a WHERE clause unless explicitly requested.

    DO NOT add LIMIT unless explicitly requested (top, bottom, first, last, highest, lowest).

    DO NOT change case, spelling, or formatting of any schema name.

    For gender:

    Use exact stored values from the schema.

    If data uses "M"/"F", convert Male → "M" and Female → "F".

    Never invent values.

    If multiple interpretations are possible, choose the safest interpretation strictly based on existing columns.

    If any requested column/table does not exist exactly as written in schema, return:
    ERROR_CANNOT_ANSWER

    Always reference the correct table name.

    Never generate multiple queries.

    Do not use aliases unless absolutely necessary for disambiguation.

    Ensure syntax is fully valid for SQLite.

    If multiple values are requested (e.g., FY '14, FY '15, FY '16), include ALL of them.

    Never omit any requested value.

    If years are stored as row values, use:
    WHERE column IN (...)

    If years are stored as column names, use column arithmetic explicitly.

    For calculations (difference, growth, percentage, ratio):

    Use explicit arithmetic expressions.

    Do not assume hidden columns.

    For conditional logic:

    Use CASE only if required by the question.

    For sorting:

    Use ORDER BY only if explicitly requested.

    For joins:

    Only join tables if necessary.

    Use explicit JOIN syntax.

    Join only on existing columns.

    For subqueries:

    Use only if required.

    Ensure they strictly follow schema.

    If the user asks something unrelated to the schema (definition, explanation, metadata, opinion, etc.), return:
    ERROR_CANNOT_ANSWER

    If ambiguity cannot be resolved strictly using schema, return:
    ERROR_CANNOT_ANSWER

    Before outputting:

    Verify every column exists.

    Verify every table exists.

    Verify aggregation compliance.

    Verify no extra filters are added.

    Verify no requested entity is missing.

    Verify only one SELECT statement is present.
    Dates in this dataset are stored as strings in the format 'DDMMMYYYY' (e.g., '23MAY2023').
- To filter by a specific year, use the LIKE operator with a wildcard: WHERE Join_Date LIKE '%2023'
- To filter by a specific month, use the LIKE operator: WHERE Join_Date LIKE '%MAY%'
- To filter by a specific day, use the LIKE operator: WHERE Join_Date LIKE '23%'
COMPARISON QUERIES: When asked to "Compare" two different years or categories, use conditional aggregation to show them side-by-side.
   Example for "Compare 2022 vs 2023":
   SELECT 
     SUM(CASE WHEN Join_Date LIKE '%2022' THEN 1 ELSE 0 END) AS Total_2022,
     SUM(CASE WHEN Join_Date LIKE '%2023' THEN 1 ELSE 0 END) AS Total_2023
   FROM data_table;
   PERCENTAGE/GROWTH: If asked for growth or difference, perform arithmetic on the SUM(CASE...) results.

{schema}
"""

def nl_to_sql(nl_query, schema):
    system_prompt = build_system_prompt(schema)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": nl_query}
        ],
        temperature=0,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def summarize_dataset(schema, dataframe):
    preview = dataframe.head(10).to_string(index=False)
    prompt = f"Schema:\n{schema}\nData:\n{preview}\nSummarize this dataset."
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def summarize_result(result_df, question):
    preview = result_df.head(20).to_string(index=False)
    prompt = f"Question: {question}\nResult Data:\n{preview}\nSummarize this in business language without mentioning SQL."
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def generate_matplotlib_code(result_df, question):
    columns = list(result_df.columns)

    prompt = f"""
Role: You are a Python Data Visualization expert specializing in Matplotlib.Task: Write a Python function plot_data(df) that generates the most effective visualization to answer the user's question based on the provided dataframe columns using ONLY Matplotlib.USER QUESTION: "{question}"DATAFRAME COLUMNS: {columns}STRICT CODE RULES:Function Signature: Define exactly as def plot_data(df):. It must take only df as an argument.Internal Variable: Store the user's question as a string variable question = "{question}" inside the function.Imports: Include import matplotlib.pyplot as plt, import pandas as pd, and import numpy as np inside the function.Style: The first line of the function body must be plt.style.use('ggplot').Initialization: Create the figure and axis using fig, ax = plt.subplots(figsize=(12, 6)).Return: The function must end with return fig.No Explanations: Provide only the Python code. No markdown code blocks, no text before or after the code.VISUALIZATION LOGIC:Select Chart Type:Comparing numeric values across categories $\rightarrow$ Bar Chart (ax.bar).Relationship between two numbers $\rightarrow$ Scatter Plot (ax.scatter).Distribution/Frequency $\rightarrow$ Histogram (ax.hist) or Boxplot (ax.boxplot).Parts of a whole $\rightarrow$ Pie Chart (ax.pie).Density or Matrix $\rightarrow$ Heatmap (use ax.imshow or ax.pcolor with a colorbar).Data Handling: Perform all necessary logic like .groupby().mean(), .value_counts(), or pd.cut() inside the function before plotting.Aesthetics:Always add ax.set_title(), ax.set_xlabel(), and ax.set_ylabel() based on the question context.For all plots: Use plt.xticks(rotation=45) if labels are categorical and always use plt.tight_layout().For Heatmaps: Use colormaps like 'YlGnBu' or 'RdYlGn', add a colorbar using fig.colorbar(), and display values using ax.text() if possible.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    code = response.choices[0].message.content.strip()
    return code.replace("```python", "").replace("```", "").strip()

# ✅ Added graph detection helper
def is_graph_request(question):
    graph_keywords = [
        "graph", "plot", "chart", "visualize", "visualization",
        "bar", "line", "pie", "histogram", "trend",
        "compare", "distribution"
    ]
    return any(word in question.lower() for word in graph_keywords)
