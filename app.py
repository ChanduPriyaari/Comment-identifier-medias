import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
from predict import predict_labels

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI YouTube Comment Moderation & Insights",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- HIDE SIDEBAR COMPLETELY ----------------
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none !important;}
[data-testid="collapsedControl"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ENV ----------------
# IMPORTANT: supports .env inside assets/ OR root
load_dotenv()
load_dotenv("assets/.env")

API_KEY = os.getenv("YOUTUBE_API_KEY")

# ---------------- MATPLOTLIB DARK MODE ----------------
plt.style.use("dark_background")

# ---------------- UTILS ----------------
def extract_video_id(url):
    if not url:
        return None
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def get_video_stats(youtube, video_id):
    req = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    res = req.execute()
    item = res["items"][0]
    stats = item["statistics"]
    snippet = item["snippet"]
    return {
        "title": snippet["title"],
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0))
    }

def fetch_all_comments(youtube, video_id):
    comments = []
    token = None

    while True:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=token,
            textFormat="plainText"
        )
        res = req.execute()

        for item in res.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)

        token = res.get("nextPageToken")
        if not token:
            break

    return comments

# ---------------- UI ----------------
st.markdown("## 🛡️ AI YouTube Comment Moderation & Insights")
st.caption("Real-time analysis of YouTube video engagement and comment toxicity")
st.divider()

video_url = st.text_input(
    "🔗 Paste YouTube Video Link",
    placeholder="https://www.youtube.com/watch?v=XXXX"
)

if st.button("🚀 Analyze Video", use_container_width=True):

    if not API_KEY:
        st.error("YouTube API key not found. Please add YOUTUBE_API_KEY in .env file.")
        st.stop()

    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube video link.")
        st.stop()

    youtube = build("youtube", "v3", developerKey=API_KEY)

    # -------- VIDEO STATS --------
    stats = get_video_stats(youtube, video_id)

    c1, c2, c3 = st.columns(3)
    c1.metric("👁 Views", f"{stats['views']:,}")
    c2.metric("👍 Likes", f"{stats['likes']:,}")
    c3.metric("💬 Total Comments", f"{stats['comments']:,}")

    st.subheader("📌 Video Title")
    st.write(stats["title"])

    st.divider()

    # -------- COMMENTS --------
    with st.spinner("Fetching and analyzing ALL comments (this may take time)..."):
        comments = fetch_all_comments(youtube, video_id)

    results = []
    for c in comments:
        labels = predict_labels(c)
        label = labels[0] if labels else "Safe"
        results.append({"Comment": c, "Label": label})

    df = pd.DataFrame(results)

    toxic = df["Label"].isin(["Toxic", "Negative"]).sum()
    safe = (df["Label"] == "Safe").sum()
    invalid = (df["Label"] == "Invalid").sum()
    total = len(df)

    # -------- METRICS --------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Analyzed", total)
    m2.metric("Toxic / Negative", toxic)
    m3.metric("Safe", safe)
    m4.metric("Invalid", invalid)

    st.divider()

    # -------- GRAPH --------
    st.subheader("📊 Comment Safety Distribution")

    chart_df = pd.DataFrame({
        "Category": ["Toxic / Negative", "Safe", "Invalid"],
        "Count": [toxic, safe, invalid]
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        chart_df["Category"],
        chart_df["Count"],
        color=["#ef4444", "#22c55e", "#9ca3af"]
    )
    ax.set_ylabel("Number of Comments")
    ax.set_title("Comment Moderation Summary")

    st.pyplot(fig)

    st.divider()

    # -------- TABLE --------
    emoji = {
        "Toxic": "🔴 Toxic",
        "Negative": "🟠 Negative",
        "Safe": "🟢 Safe",
        "Invalid": "⚪ Invalid"
    }
    df["Status"] = df["Label"].map(emoji)

    st.subheader("📋 All Comments Analysis")
    st.dataframe(df[["Comment", "Status"]], use_container_width=True)
