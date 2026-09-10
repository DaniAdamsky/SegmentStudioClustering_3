import os #added os import for safe API key placement
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from huggingface_hub import InferenceClient

#client = InferenceClient(base_url="http://localhost:11434/v1", api_key="ollama")
#LLM_MODEL = "llama3.1:latest"

try:
    OLLAMA_KEY = st.secrets["OLLAMA_API_KEY"]
except Exception:
    OLLAMA_KEY = os.environ.get("OLLAMA_API_KEY", "")

client = InferenceClient(base_url="https://ollama.com/v1", api_key=OLLAMA_KEY)
LLM_MODEL = "gemma4:31b"

def compute_wcss(numeric_df, k_min, k_max):
    # run KMeans for every k and save theWCSS
    X = StandardScaler().fit_transform(numeric_df)
    ks, wcss = [], []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X)
        ks.append(k)
        wcss.append(model.inertia_)
    return pd.DataFrame({"k": ks, "wcss": wcss})


def plot_elbow(results):
    fig, ax = plt.subplots()
    ax.plot(results["k"], results["wcss"], marker="o")
    ax.set_title("Elbow Plot")
    ax.set_xlabel("k")
    ax.set_ylabel("WCSS")
    return fig


def create_clusters(numeric_df, k):
    X = StandardScaler().fit_transform(numeric_df)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_ids = model.fit_predict(X)
    return cluster_ids, model


def cluster_counts_table(df):
    table = df.groupby("cluster_id").size().reset_index(name="count")
    table["name"] = ""
    table["description"] = ""
    return table


def most_common(series):
    modes = series.mode()   # ignores non-number values
    if modes.empty:
        return "N/A"
    return modes.iloc[0]


def build_summary_text(cid, count, mean_row, mode_row=None):
    means = ", ".join(f"{col} average {round(val, 2)}" for col, val in mean_row.items())
    text = f"Cluster {cid}: {count} rows. Numeric averages: {means}."
    if mode_row is not None:
        modes = ", ".join(f"{col} most often {val}" for col, val in mode_row.items())
        text += f" Most common categories: {modes}."
    return text


def build_prompt(summary_text):
    return (
        "Here is a statistical summary of one cluster:\n"
        f"{summary_text}\n"
        "Give it a short group name (2-4 words) and a one-sentence description.\n"
        "Answer ONLY in this format:\n"
        "Name: <name>\n"
        "Description: <description>"
    )


def get_llm_reply(prompt):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content":
                "You name clusters from a dataset. Reply only as "
                "'Name: <name>' then 'Description: <description>'."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
    )
    return response.choices[0].message.content


def parse_reply(reply_text):
    # pull the Name and Description lines out of the model's reply
    name, description = "", ""
    for line in reply_text.splitlines():
        clean = line.strip().lstrip("*#- ").strip()
        low = clean.lower()
        if low.startswith("name:"):
            name = clean.split(":", 1)[1].strip().strip("*").strip()
        elif low.startswith("description:"):
            description = clean.split(":", 1)[1].strip().strip("*").strip()
    return name, description


def label_clusters(df):
    table = cluster_counts_table(df)
    numeric_means = df.groupby("cluster_id").mean(numeric_only=True)

    categorical_cols = df.select_dtypes(include="object").columns
    if len(categorical_cols) > 0:
        categorical_modes = df.groupby("cluster_id")[categorical_cols].agg(most_common)
    else:
        categorical_modes = None
    #looping through the df with pd function .iterrows
    for i, row in table.iterrows():
        cid = row["cluster_id"]
        mean_row = numeric_means.loc[cid]
        mode_row = categorical_modes.loc[cid] if categorical_modes is not None else None
        summary = build_summary_text(cid, row["count"], mean_row, mode_row)
        name, description = parse_reply(get_llm_reply(build_prompt(summary)))
        if not name:#just in case the model didn't follow the format
            name = f"Cluster {cid}"
        table.loc[i, "name"] = name
        table.loc[i, "description"] = description
    return table


#streamlit
st.title("Segment Studio")

#upload and show the CSV
file = st.file_uploader("Choose a CSV file", type="csv")
if file is not None:
    #only read again when a new file is chosen so we dont lose the clusters
    if st.session_state.get("original_name") != file.name:
        st.session_state["df"] = pd.read_csv(file)
        st.session_state["original_name"] = file.name
        st.session_state.pop("labeled_table", None)
    st.subheader("Step n.1: CSV Table")
    st.dataframe(st.session_state["df"])

if "df" in st.session_state:
    df = st.session_state["df"]

    # keep only numeric columns for clustering and fill missing values
    numeric_df = df.drop(columns=["cluster_id"], errors="ignore").select_dtypes(include="number")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.fillna(numeric_df.mean())

    if numeric_df.shape[1] == 0:
        st.error("this CSV has no numeric columns to cluster.")
    else:
        n_rows = len(numeric_df)
        max_k = max(3, min(20, n_rows - 1))

        #elbow/WCSS
        st.subheader("Step n.2: WCSS (Elbow)")
        col1, col2 = st.columns(2)
        k_min = col1.slider("Min k", 2, max_k, 2)
        k_max = col2.slider("Max k", 2, max_k, min(10, max_k))
        if st.button("Run WCSS"):
            if k_min > k_max:
                st.warning("Min k must be less than or equal to Max k.")
            else:
                results = compute_wcss(numeric_df, k_min, k_max)
                st.dataframe(results)
                st.pyplot(plot_elbow(results))

        #choose k and create the clusters
        st.subheader("Step n.3: Choose k and create clusters")
        k = st.slider("Select k", 2, max_k, min(7, max_k))
        if st.button("Create clusters"):
            cluster_ids, _ = create_clusters(numeric_df, k)
            df["cluster_id"] = cluster_ids
            st.session_state["df"] = df
            st.session_state.pop("labeled_table", None)
            st.write("Cluster counts (name/description empty for now)")
            st.dataframe(cluster_counts_table(df))

        #name the clusters with the LLM
        if "cluster_id" in st.session_state["df"].columns:
            st.subheader("Step n4: Generate group name + description")
            if st.button("Generate names/descriptions"):
                labeled = label_clusters(st.session_state["df"])
                st.session_state["labeled_table"] = labeled
            if "labeled_table" in st.session_state:
                st.write("Cluster labels (with name + description)")
                st.dataframe(st.session_state["labeled_table"])

        #export the clustered CSV
        if "labeled_table" in st.session_state:
            st.subheader("Step n.5: Export clustered CSV")
            df = st.session_state["df"]
            labeled = st.session_state["labeled_table"]
            id_to_name = dict(zip(labeled["cluster_id"], labeled["name"]))
            df["cluster_name"] = df["cluster_id"].map(id_to_name)

            original = st.session_state.get("original_name", "data.csv")
            output_name = original.replace(".csv", "") + "_clustered.csv"

            st.download_button(
                "Download clustered CSV",
                data=df.to_csv(index=False),
                file_name=output_name,
                mime="text/csv",
            )