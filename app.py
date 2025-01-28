import streamlit as st
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
import os
from dotenv import load_dotenv
import json
import google.generativeai as genai
import yt_dlp
from moviepy.editor import VideoFileClip, concatenate_videoclips
from datetime import timedelta
import time
import io
load_dotenv()


def seconds_to_hhmmss(seconds):
    """Convert seconds to hh:mm:ss format."""
    return str(timedelta(seconds=int(seconds)))


def fetch_youtube_transcript(video_id):
    try:
        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Create an array of objects with text, start_time, and end_time in hh:mm:ss format
        transcript_array = []
        for i, entry in enumerate(transcript):
            start_time = entry["start"]
            end_time = transcript[i + 1]["start"] if i + 1 < len(transcript) else start_time + entry.get("duration", 0)
            transcript_array.append({
                "text": entry["text"],
                "start_time": seconds_to_hhmmss(start_time),
                "end_time": seconds_to_hhmmss(end_time)
            })
        return transcript_array

    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except Exception as e:
        return f"Error: {str(e)}"


def fetch_video_details(api_key, video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if not data["items"]:
        return {"error": "Video not found"}

    video_snippet = data["items"][0]["snippet"]
    return {
        "id": video_id,
        "title": video_snippet["title"]
    }


def fetch_playlist_videos(api_key, playlist_id):
    url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&playlistId={playlist_id}&maxResults=50&key={api_key}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    videos = []
    for item in data.get("items", []):
        video_id = item["snippet"]["resourceId"]["videoId"]
        video_title = item["snippet"]["title"]
        videos.append({"id": video_id, "title": video_title})
    return videos

def call_gemini_api(api_key, transcript, user_prompt):
    # Configure the API with your API key
    genai.configure(api_key=api_key)

    # Prepare the user prompt and the transcript content for the request
    user_content = f"""
    You are an expert content reviewer. Below is a transcript of a video, and the user has provided a specific prompt. Your task is to filter and extract the sections of the transcript that are most relevant to the given prompt.

    When filtering, analyze groups of transcript entries (not just individual entries) and check for their overall relevance to the prompt. Avoid filtering based solely on a single entry's text and its direct match to the prompt. Instead, consider the broader context of related entries.

    User's prompt: "{user_prompt}"

    Transcript:
    {json.dumps(transcript, ensure_ascii=False, indent=2)}

    Please return a JSON array in below format
    [
        {{
            "text": "The text of the transcript.",
            "start_time": "The start time of the transcript in hh:mm:ss format.",
            "end_time": "The end time of the transcript in hh:mm:ss format."
        }},
        ...
    ]
    Only return array.
    """

    # Call the Google Gemini API for content generation
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(user_content)
    # Extract content between []
    start = response.text.find('[') + 1
    end = response.text.find(']')
    content = response.text[start:end]

    # Extract the response content (filtered transcript) from the model
    try:
        filtered_transcript = json.loads('[' + content + ']') # Assuming response contains valid JSON
    except json.JSONDecodeError:
        filtered_transcript = []

    return filtered_transcript


def process_youtube_link(api_key, anthropic_key, url, user_prompt):
    try:
        if "list=" in url:  # Playlist link
            playlist_id = url.split("list=")[1].split("&")[0]
            videos = fetch_playlist_videos(api_key, playlist_id)
            for video in videos:
                transcript_data = fetch_youtube_transcript(video["id"])
                filtered_transcript = call_gemini_api(anthropic_key, transcript_data, user_prompt)
                video["filtered_transcript"] = filtered_transcript
                video["transcript_data"] = transcript_data
            return videos

        elif "v=" in url:  # Single video link
            video_id = url.split("v=")[1].split("&")[0]
            video_details = fetch_video_details(api_key, video_id)
            if "error" in video_details:
                return video_details
            transcript_data = fetch_youtube_transcript(video_id)
            filtered_transcript = call_gemini_api(anthropic_key, transcript_data, user_prompt)
            video_details["filtered_transcript"] = filtered_transcript
            video_details["transcript_data"] = transcript_data
            return [video_details]

        else:
            return {"error": "Invalid YouTube link"}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def filter_transcripts(transcripts):
    filtered_segments = []
    start_time = None
    end_time = None
    print(transcripts)
    for i in range(len(transcripts)):
        current_segment = transcripts[i]
        if start_time is None:
            start_time = current_segment['start_time']
        
        # Check if next segment is within 2 seconds
        if i + 1 < len(transcripts):
            next_segment = transcripts[i + 1]
            current_end_time = convert_time_to_seconds(current_segment['end_time'])
            next_start_time = convert_time_to_seconds(next_segment['start_time'])

            if next_start_time - current_end_time <= 2:
                continue  # Merge this segment with the next
            else:
                # Save the current segment as a separate clip
                end_time = current_segment['end_time']
                filtered_segments.append({"start_time": start_time, "end_time": end_time})
                start_time = None  # Reset start time for the next segment
        else:
            end_time = current_segment['end_time']
            filtered_segments.append({"start_time": start_time, "end_time": end_time})

    return filtered_segments

def convert_time_to_seconds(time_str):
    if ':' in time_str:
        time_parts = time_str.split(':')
        if len(time_parts) == 2:
            return int(time_parts[0]) * 60 + int(time_parts[1])
        elif len(time_parts) == 3:
            return int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
    return 0

def clip_video(video_id, video_title, timestamps):

    # Use the timestamp directly as the file name
    video_file_name = f"{video_title}.mp4"

    video_url = f'https://www.youtube.com/watch?v={video_id}'

    # Set options for yt-dlp to download the video
    ydl_opts = {
        'format': 'worst',  # Download the best quality video and audio
        'outtmpl': video_file_name,  # Output file name
    }

    # Download the video using yt-dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Process the downloaded video file
    clip = VideoFileClip(video_file_name)

    # Create subclips based on timestamps
    subclips = []
    for timestamp in timestamps:
        # Extract start_time and end_time
        start_time = timestamp["start_time"]
        end_time = timestamp["end_time"]

        # Convert timestamps to seconds
        start_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(":"))))
        end_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(":"))))

        # Create a subclip
        subclip = clip.subclip(start_seconds, end_seconds)
        subclips.append(subclip)

    # Combine all the subclips into a single video
    combined_clip = concatenate_videoclips(subclips)

    # Save the combined video to a file
    output_file = f"{video_title}_clipped.mp4"
    combined_clip.write_videofile(output_file, codec='libx264')

    print(f"Video clipped and saved as {output_file}")

def create_video(data):
    video_name=[]
    # Generate current timestamp in milliseconds (rounded) as a string

    for video in data:
        video_id = video["id"]
        current_timestamp = str(round(time.time() * 1000))
        video_name.append(current_timestamp)
        video_title = current_timestamp
        filtered_transcript = video["filtered_transcript"]
        clip_video(video_id,video_title,filtered_transcript)

    return video_name


def parse_time(time_str):
    """Convert a time string (e.g., '0:00:06') to a timedelta object."""
    h, m, s = map(int, time_str.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)

def combine_transcript(transcript):
    """Group sequential transcript entries based on start and end times."""
    grouped = []
    current_group = None

    for entry in transcript:
        start_time = parse_time(entry["start_time"])
        end_time = parse_time(entry["end_time"])

        if not current_group:
            # Start a new group
            current_group = {
                "text": entry["text"],
                "start_time": entry["start_time"],
                "end_time": entry["end_time"]
            }
        else:
            # Check if this entry is sequential
            last_end_time = parse_time(current_group["end_time"])
            if start_time == last_end_time:
                # Extend the current group
                current_group["text"] += " " + entry["text"]
                current_group["end_time"] = entry["end_time"]
            else:
                # Save the current group and start a new one
                grouped.append(current_group)
                current_group = {
                    "text": entry["text"],
                    "start_time": entry["start_time"],
                    "end_time": entry["end_time"]
                }

    # Append the last group
    if current_group:
        grouped.append(current_group)

    return grouped

def group_transcripts(data):
    """Process an array of transcript objects, grouping each `filtered_transcript`."""
    for item in data:
        item["filtered_transcript"] = combine_transcript(item["filtered_transcript"])
    return data


def save_to_json(data, filename):
    try:
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to JSON file: {str(e)}")


# Streamlit UI
st.title("YouTube Video Transcript Filtering and Clipping")

# Take user input for YouTube URL and prompt
youtube_url = st.text_input("Enter YouTube URL (Video or Playlist)", "")
user_prompt = st.text_input("Enter User Prompt", "")
youtube_api_key = st.text_input("Youtube API Key", "")
gemini_api_key = st.text_input("Gemini API Key", "")

if st.button("Process Video"):
    if youtube_url and user_prompt and youtube_api_key and gemini_api_key:
        with st.spinner('Wait for it...'):
            time.sleep(5)
        result = process_youtube_link(youtube_api_key, gemini_api_key, youtube_url, user_prompt)
        save_to_json(result,"result.json")

        filtered_data=group_transcripts(result)

        for item in filtered_data:
            text=""
            for transcript in item["filtered_transcript"]:
                text =text+ f"\n {transcript['start_time']} - {transcript['end_time']} {transcript['text']}\n"
            # Use custom CSS to create a scrollable container with static height
            st.subheader(item["title"])
            st.write(text)

        save_to_json(filtered_data,"result-group.json")

        video_name = create_video(filtered_data)
        clips=[]
        print(video_name)
        # Read each video file and add it to the clips list
        for video in video_name:
            clip = VideoFileClip(f"{video}_clipped.mp4")
            clips.append(clip)

        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)

        final_video_name=f"{str(round(time.time() * 1000))}_final_video.mp4"
        # Save the combined video
        final_clip.write_videofile(final_video_name, codec="libx264")

        # Display the combined video in Streamlit
        st.title("Final Video")
        st.video(final_video_name)

    else:
        st.error("Please provide all data")
    


         