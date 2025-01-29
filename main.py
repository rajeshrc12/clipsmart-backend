from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from fastapi import FastAPI
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
import json
import google.generativeai as genai
import yt_dlp
from moviepy.editor import VideoFileClip, concatenate_videoclips
from datetime import timedelta
import time
import boto3
# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get the frontend URL from environment variables
frontend_url = os.getenv("FRONTEND_URL")

# Allow CORS from frontend URL (and development URLs)
origins = [
    frontend_url,  # Production URL from environment variable
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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
            end_time = transcript[i + 1]["start"] if i + \
                1 < len(transcript) else start_time + entry.get("duration", 0)
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
    response = genai.GenerativeModel(
        "gemini-1.5-flash").generate_content(user_content)
    print(response)
    # Extract content between []
    start = response.text.find('[') + 1
    end = response.text.find(']')
    content = response.text[start:end]

    # Extract the response content (filtered transcript) from the model
    try:
        # Assuming response contains valid JSON
        filtered_transcript = json.loads('[' + content + ']')
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
                filtered_transcript = call_gemini_api(
                    anthropic_key, transcript_data, user_prompt)
                video["filtered_transcript"] = filtered_transcript
                # video["transcript_data"] = transcript_data
            return videos

        elif "v=" in url:  # Single video link
            video_id = url.split("v=")[1].split("&")[0]
            video_details = fetch_video_details(api_key, video_id)
            if "error" in video_details:
                return video_details
            transcript_data = fetch_youtube_transcript(video_id)
            filtered_transcript = call_gemini_api(
                anthropic_key, transcript_data, user_prompt)
            video_details["filtered_transcript"] = filtered_transcript
            # video_details["transcript_data"] = transcript_data
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
            current_end_time = convert_time_to_seconds(
                current_segment['end_time'])
            next_start_time = convert_time_to_seconds(
                next_segment['start_time'])

            if next_start_time - current_end_time <= 2:
                continue  # Merge this segment with the next
            else:
                # Save the current segment as a separate clip
                end_time = current_segment['end_time']
                filtered_segments.append(
                    {"start_time": start_time, "end_time": end_time})
                start_time = None  # Reset start time for the next segment
        else:
            end_time = current_segment['end_time']
            filtered_segments.append(
                {"start_time": start_time, "end_time": end_time})

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
    if(len(timestamps)==0):
        return
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
        start_seconds = sum(int(x) * 60 ** i for i,
                            x in enumerate(reversed(start_time.split(":"))))
        end_seconds = sum(int(x) * 60 ** i for i,
                          x in enumerate(reversed(end_time.split(":"))))

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
    video_name = []
    # Generate current timestamp in milliseconds (rounded) as a string
    
    for video in data:
        filtered_transcript = video["filtered_transcript"]
        if not filtered_transcript:
            continue
        video_id = video["id"]
        current_timestamp = str(round(time.time() * 1000))
        video_name.append(current_timestamp)
        video_title = current_timestamp
        clip_video(video_id, video_title, filtered_transcript)

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
        item["filtered_transcript"] = combine_transcript(
            item["filtered_transcript"])
    return data

def save_to_json(data, filename):
    try:
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to JSON file: {str(e)}")

region_name = os.getenv("AWS_REGION")
aws_access_key_id = os.getenv("AWS_S3_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_S3_SECRET_KEY")

s3_client = boto3.client('s3', region_name=region_name, 
                         aws_access_key_id=aws_access_key_id, 
                         aws_secret_access_key=aws_secret_access_key)

def upload_to_s3(file_path, bucket_name, s3_path):
    """Uploads the file to AWS S3 and returns a pre-signed URL."""
    try:
        with open(file_path, 'rb') as f:
            s3_client.upload_fileobj(f, bucket_name, s3_path)

        # Generate a pre-signed URL (valid for 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_path},
            ExpiresIn=3600  # URL expires in 1 hour
        )

        print(f"Video uploaded successfully.")
        print(f"Pre-signed URL: {presigned_url}")
        return presigned_url  # Return the secure URL

    except Exception as e:
        print(f"Failed to upload video to S3: {e}")
        return None

class VideoRequest(BaseModel):
    prompt_link: str
    prompt: str
@app.post("/")
async def process_video(video_request: VideoRequest):
    youtube_url = video_request.prompt_link
    user_prompt = video_request.prompt
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    result = process_youtube_link(youtube_api_key, gemini_api_key, youtube_url, user_prompt)
    print(result)
    filtered_data=group_transcripts(result)


    video_name = create_video(filtered_data)

    clips=[]
    print(video_name)
    # Read each video file and add it to the clips list
    for video in video_name:
        clip = VideoFileClip(f"{video}_clipped.mp4")
        clips.append(clip)

    # # Concatenate all clips
    edited_link=""
    if(len(clips)>0):
        print(len(clips),len(clips)>0)
        final_clip = concatenate_videoclips(clips)

        final_video_name=f"{str(round(time.time() * 1000))}_final_video.mp4"
        # Save the combined video
        final_clip.write_videofile(final_video_name, codec="libx264")

        bucket_name = os.getenv("AWS_BUCKET_NAME")
        edited_link=upload_to_s3(final_video_name,bucket_name,final_video_name)

    return {"transcription": filtered_data, "edited_link":edited_link}
