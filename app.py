import streamlit as st
import pandas as pd
from predict import predict_labels

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Comment Analytics & Moderation System",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
## AI Comment Analytics & Moderation System

This application analyzes **user-generated comments** using Machine Learning
to identify **toxic behavior** (insult, hate, threat, harassment) and
**positive intent** (love, support).

The goal is to **support content moderation decisions** through analytics,
not just individual predictions.
""")

st.divider()

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.markdown("### Comment Input")

input_type = st.radio(
"How would you like to analyze comments?",
["Analyze one comment", "Analyze multiple comments"],  
)

comments = []

if input_type == "Single Comment":
    single_comment = st.text_area(
        "Enter a comment for analysis:",
        height=120,
        placeholder="Example: This comment is abusive and offensive"
    )
    if single_comment.strip():
        comments.append(single_comment)

else:
    batch_text = st.text_area(
        "Enter multiple comments (one per line):",
        height=220,
        placeholder=(
            "Thanks for the explanation\n"
            "I will harm you"
        )
    )
    comments = [c.strip() for c in batch_text.split("\n") if c.strip()]

st.divider()

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
if st.button("Run Analysis", use_container_width=True):

    if not comments:
        st.warning("Please provide at least one comment.")
    else:
        analysis_results = []
        label_distribution = {}

        for comment in comments:
            labels = predict_labels(comment)

            analysis_results.append({
                "Comment": comment,
                "Classification Result": (
                    ", ".join(labels.keys()) if labels else "Safe / Neutral"
                )
            })

            for label in labels:
                label_distribution[label] = label_distribution.get(label, 0) + 1

        df_results = pd.DataFrame(analysis_results)

        # --------------------------------------------------
        # SUMMARY METRICS
        # --------------------------------------------------
        st.markdown("### Analysis Summary")

        total = len(comments)
        toxic = sum(
            1 for r in analysis_results
            if r["Classification Result"] != "Safe / Neutral"
        )
        safe = total - toxic

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Comments", total)
        c2.metric("Toxic Comments", toxic)
        c3.metric("Safe / Neutral", safe)

        st.divider()

        # --------------------------------------------------
        # TOXICITY DISTRIBUTION
        # --------------------------------------------------
        if label_distribution:
            st.markdown("### Toxicity Distribution by Category")

            df_labels = pd.DataFrame.from_dict(
                label_distribution,
                orient="index",
                columns=["Occurrences"]
            )

            st.bar_chart(df_labels)

        else:
            st.info("No toxic categories detected in the analyzed comments.")

        st.divider()

        # --------------------------------------------------
        # DETAILED RESULTS
        # --------------------------------------------------
        st.markdown("### Detailed Comment Analysis")

        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True
        )

    
