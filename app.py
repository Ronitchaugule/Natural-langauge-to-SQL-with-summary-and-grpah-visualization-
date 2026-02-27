import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  
from nl_to_sql import (
    nl_to_sql,
    summarize_dataset,
    summarize_result,
    generate_matplotlib_code
)
from database import load_file_to_sqlite, get_schema

st.set_page_config(page_title="AI Data Assistant", layout="wide")
st.title("🧠 AI Data Assistant (NL → SQL + LLM Visualization)")


# Remove graph-related words before SQL generation
def clean_question_for_sql(question):
    graph_words = [
        "graph", "plot", "chart", "visualize", "visualization",
        "bar chart", "line chart", "pie chart",
        "histogram", "bar", "line", "pie " ,"scatter plot","heat map"  
    ]

    cleaned = question.lower()
    for word in graph_words:
        cleaned = cleaned.replace(word, "")

    return cleaned.strip()


uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    conn, TABLE_NAME, df = load_file_to_sqlite(uploaded_file)

    st.subheader("📄 Dataset Preview")
    rows_to_show = st.number_input(
        "Select rows to preview:",
        min_value=5,
        max_value=len(df),
        value=min(50, len(df))
    )
    st.dataframe(df.head(rows_to_show), use_container_width=True)

    schema = get_schema(conn, TABLE_NAME)

    with st.expander("📌 View Schema"):
        st.text(schema)

    mode = st.radio("Choose Action:", ["🔹 Ask Question", "🔹 Summarize Entire Dataset"])

    if mode == "🔹 Ask Question":
        question = st.text_input("Ask a question (you can request a graph too):")

        if question:

            cleaned_question = clean_question_for_sql(question)

            with st.spinner("Generating SQL..."):
                sql = nl_to_sql(cleaned_question, schema)

            if sql != "ERROR_CANNOT_ANSWER":
                try:
                    result = pd.read_sql_query(sql, conn)

                    if not result.empty:

                        st.subheader("🧾 Results")
                        st.dataframe(result, use_container_width=True)

                        with st.spinner("Generating AI Insight..."):
                            summary = summarize_result(result, cleaned_question)
                            st.info(f"**AI Insight:** {summary}")

                        st.divider()
                        st.subheader("📈 AI Generated Visualization")

                        with st.spinner("Designing visualization..."):
                            plot_code = generate_matplotlib_code(result, question)

                            try:
                                # 🔥 FIX: Inject matplotlib into execution scope
                                safe_globals = {
                                    "plt": plt,
                                    "pd": pd,
                                    "__builtins__": __builtins__,
                                }

                                local_vars = {}

                                exec(plot_code, safe_globals, local_vars)

                                if "plot_data" in local_vars:
                                    fig = local_vars["plot_data"](result)
                                    st.pyplot(fig)

                                    with st.expander("View Generated Code"):
                                        st.code(plot_code, language="python")
                                else:
                                    st.error("LLM did not define plot_data(df) properly.")

                            except Exception as e:
                                st.error(f"Visualization Logic Error: {e}")
                                st.warning("Fallback to simple bar chart.")
                                st.bar_chart(result.select_dtypes(include=["number"]))

                    else:
                        st.warning("The query returned no results.")

                except Exception as e:
                    st.error(f"SQL Execution Error: {e}")

            else:
                st.error("❌ Cannot answer with given schema.")

    elif mode == "🔹 Summarize Entire Dataset":
        if st.button("Generate Dataset Summary"):
            with st.spinner("Analyzing..."):
                summary = summarize_dataset(schema, df)
            st.subheader("🧠 Dataset Summary")
            st.success(summary)
