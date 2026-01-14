import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

from googleapiclient.discovery import build
from predict import predict_labels

# ================= CONFIG =================
st.set_page_config(page_title="YouTube Insights", layout="wide")

YOUTUBE_API_KEY = "AIzaSyBp3pLDcXzQXhAHVv_uOkoWP6Eu2ibkmSI"  # 🔴 PASTE HERE ONLY

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ================= HELPERS =================
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_video_stats(video_id):
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()
    return response["items"][0]


def get_comments(video_id, max_results=100):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request and len(comments) < max_results:
        response = request.execute()
        for item in response["items"]:
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)

        request = youtube.commentThreads().list_next(request, response)

    return comments[:max_results]


# ================= UI =================
st.title("📊 YouTube Video Insights & Comment Moderation")
st.caption("Real-time YouTube analytics with AI-based comment moderation")

st.divider()

video_url = st.text_input(
    "Paste YouTube Video Link",
    placeholder="https://www.youtube.com/watch?v=XXXXXXXXXXX"
)

if st.button("Analyze Video", use_container_width=True):

    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube video link")
        st.stop()

    with st.spinner("Fetching video data..."):
        video_data = get_video_stats(video_id)

    stats = video_data["statistics"]
    snippet = video_data["snippet"]

    # -------- METRICS --------
    c1, c2, c3 = st.columns(3)
    c1.metric("Views", stats.get("viewCount", 0))
    c2.metric("Likes", stats.get("likeCount", 0))
    c3.metric("Comments", stats.get("commentCount", 0))

    st.subheader(snippet["title"])
    st.caption(snippet["channelTitle"])

    st.divider()

    # -------- COMMENTS --------
    with st.spinner("Fetching comments..."):
        comments = get_comments(video_id, max_results=100)

    results = []

    for c in comments:
        labels = predict_labels(c)
        label = labels[0] if labels else "Safe"
        results.append({"Comment": c, "Label": label})

    df = pd.DataFrame(results)

    # -------- COUNTS --------
    toxic = df["Label"].isin(["Toxic", "Negative", "Hate", "Insult", "Harassment"]).sum()
    safe = (df["Label"] == "Safe").sum()
    invalid = (df["Label"] == "Invalid").sum()

    st.divider()

    # -------- GRAPH --------
    st.subheader("📈 Comment Safety Distribution")

    chart_df = pd.DataFrame({
        "Category": ["Toxic", "Safe", "Invalid"],
        "Count": [toxic, safe, invalid]
    })

    fig, ax = plt.subplots()
    ax.bar(chart_df["Category"], chart_df["Count"])
    ax.set_ylabel("Number of Comments")
    ax.set_title("AI Comment Analysis")

    st.pyplot(fig)

    # -------- TABLE --------
    emoji_map = {
        "Toxic": "🔴 Toxic",
        "Negative": "🟠 Negative",
        "Safe": "🟢 Safe",
        "Invalid": "⚪ Invalid",
        "Hate": "🔴 Hate",
        "Insult": "🟠 Insult",
        "Harassment": "🔴 Harassment"
    }

    df["Status"] = df["Label"].map(lambda x: emoji_map.get(x, "🟢 Safe"))

    st.subheader("🧾 Top Analyzed Comments (AI)")
    st.dataframe(df[["Comment", "Status"]], use_container_width=True)
