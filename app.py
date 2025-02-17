from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import requests
from youtube_transcript_api import YouTubeTranscriptApi
import json
import google.generativeai as genai
import yt_dlp
from moviepy.editor import VideoFileClip, concatenate_videoclips
from datetime import timedelta
import time
import boto3
from openai import OpenAI
from deep_translator import GoogleTranslator
import replicate
from urllib.parse import urlparse, parse_qs
import re
# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get environment variables
frontend_url = os.getenv("FRONTEND_URL")
bucket_name = os.getenv("AWS_BUCKET_NAME")
region_name = os.getenv("AWS_REGION")
aws_access_key_id = os.getenv("AWS_S3_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_S3_SECRET_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
replicate_api_url = os.getenv("REPLICATE_API_URL")
YOUTUBE_PLAYLIST_ITEMS_API = "https://www.googleapis.com/youtube/v3/playlistItems"
YOUTUBE_VIDEO_API = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_PLAYLIST_API = "https://www.googleapis.com/youtube/v3/playlists"

client = OpenAI(api_key=openai_api_key)
genai.configure(api_key=gemini_api_key)
global_error=[]
# Allow CORS from frontend URL (and development URLs)
origins = [
    frontend_url,
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_client = boto3.client('s3', region_name=region_name,
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)


def get_prompt(transcription, user_prompt, language_code="en"):
    print("Processing the prompt to extract relevant sections from the video transcript...")
    language_prompt = ""
    if ("en" not in language_code.lower()):
        language_prompt = f" User's prompt and Transcript is in non english language. So filter accordingly with proper language context and language code: {language_code} "
    result = f"""
You are an expert content reviewer. Below is a transcript of a video with index number, and the user has provided a specific prompt. 
Your task is to filter and extract the numbers of the transcript that are most relevant to the given prompt. 
When filtering, analyze groups of transcript entries (not just individual entries) and check for their overall relevance to the prompt. 
Avoid filtering based solely on a single entry and its direct match to the prompt. Instead, consider the broader context of related entries.{language_prompt}

User's prompt: {user_prompt}

Transcript:
{transcription}

Please return a JSON array below format
[1,2,3,4....,N]
Only return array of index.
        """
    print("Prompt processing completed.")
    return result


def seconds_to_hhmmss(seconds):
    try:
        result = str(timedelta(seconds=int(seconds)))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to convert seconds to hh:mm:ss {str(e)}")
    return result


def get_video_id(video_url):
    print("Extracting video ID from URL...")
    try:
        result = video_url.split("v=")[1].split("&")[0]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch video id {str(e)}")
    print("Video ID extracted successfully.")
    return result

def parse_duration(iso_duration):
    """Convert ISO 8601 duration format (e.g., PT2M8S) to hh:mm:ss format."""
    pattern = r'PT(\d+H)?(\d+M)?(\d+S)?'
    match = re.match(pattern, iso_duration)
    
    hours = match.group(1)
    minutes = match.group(2)
    seconds = match.group(3)

    hours = int(hours[:-1]) if hours else 0
    minutes = int(minutes[:-1]) if minutes else 0
    seconds = int(seconds[:-1]) if seconds else 0
    
    # Format the result as hh:mm:ss
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def get_video_details(video_id):
    print("Fetching video details from YouTube API...")
    try:
        # Modify the API URL to include both snippet and contentDetails
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails&id={video_id}&key={youtube_api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Check if the video exists
        if not data["items"]:
            global_error.append(f"error while fetching video details, video id:{video_id} error: video not found or private")
            return False
        
        video_snippet = data["items"][0]["snippet"]
        video_content_details = data["items"][0]["contentDetails"]
        
        # Extract title and duration
        title = video_snippet["title"]
        duration = video_content_details["duration"]  # ISO 8601 format
        
        result = {
            "id": video_id,
            "title": title,
            "duration": parse_duration(duration),
        }
        
        print("Video details fetched successfully.")
        return result
    except Exception as e:
        print("Failed to fetch video details:", str(e))
        global_error.append(f"error while fetching video details, video id:{video_id} error: {str(e)}")
        return False


def get_transcription_with_index(video_id, language_code="en"):
    print("Retrieving transcription for the video...")
    try:
        youtube_transcription = YouTubeTranscriptApi.get_transcript(
            video_id, languages=[language_code])
        transcription_with_index = ""
        transcription = []
        for i, entry in enumerate(youtube_transcription):
            text = entry["text"]
            start_time = entry["start"]
            end_time = youtube_transcription[i + 1]["start"] if i + \
                1 < len(youtube_transcription) else start_time + entry.get("duration", 0)
            transcription.append({
                "text": text,
                "start_time": seconds_to_hhmmss(start_time),
                "end_time": seconds_to_hhmmss(end_time)
            })
            transcription_with_index += f"{i+1}.[{seconds_to_hhmmss(start_time)}-{seconds_to_hhmmss(end_time)}] {text} \n"
        print("Transcription retrieved successfully.")
        return {"transcription": transcription, "transcription_with_index": transcription_with_index}
    except:
        return {"transcription": [], "transcription_with_index": ""}


def get_transcription(video_id):
    print("Retrieving transcription for the video...")
    try:
        transcription = YouTubeTranscriptApi.get_transcript(video_id)
        fixed_transcript = []

        for i, entry in enumerate(transcription):
            start = entry['start']
            duration = entry['duration']
            end = start + duration  # Calculate the original end time

            # If there is a next subtitle, adjust the end time to prevent overlap
            if i < len(transcription) - 1:
                next_start = transcription[i + 1]['start']
                if end > next_start:  # Overlap detected
                    end = next_start  # Adjust end time to match next start

            fixed_transcript.append(
                {'text': entry['text'], 'start_time': seconds_to_hhmmss(start), 'end_time': seconds_to_hhmmss(end), "start": start,"end":end, "duration": duration})
        print("Retrieving transcription complete")
        return fixed_transcript
    except Exception as e:
        global_error.append(f"error while fetching transcription, video id:{video_id} error: {str(e)}")
        return []


def download_audio(video_id):
    print("Downloading audio from YouTube...")
    try:
        audio_name = get_random_name_with()
        audio_url = f'https://www.youtube.com/watch?v={video_id}'
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',  # Change to 'wav' or 'm4a' if needed
                'preferredquality': '192',
            }],
            # Save audio with title as filename
            'outtmpl': f"{audio_name}.%(ext)s",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([audio_url])
        print("Audio downloaded successfully.")
        return audio_name
    except Exception as e:
        return False


def create_transcription_with_index_using_openai(audio_name):
    print("Creating a new transcription using openai for the video...")
    try:
        audio_file = open(audio_name+".mp3", "rb")
        result = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        print("create_transcription_with_index_using_openai", result)
        transcription_with_index = ""
        # Assuming transcription is an instance of TranscriptionVerbose
        transcription = [
            {
                'start_time': seconds_to_hhmmss(segment.start),
                'end_time': seconds_to_hhmmss(segment.end),
                'text': segment.text.strip()
            }
            for segment in result.segments
        ]
        for i, entry in enumerate(transcription):
            text = entry["text"]
            start_time = entry["start_time"]
            end_time = entry["end_time"]
            transcription_with_index += f"{i+1}.[{start_time}-{end_time}] {text}\n"

        print("Transcription creation completed using openai.")
        return {"transcription": transcription, "transcription_with_index": transcription_with_index, "language_code": result.language}
    except Exception as e:
        print("Transcription creation failed using openai.", str(e))
        return {"transcription": [], "transcription_with_index": "", "language_code": "en"}


def create_transcription_with_index_using_replicate(audio_name):
    print("Creating a new transcription using replicate for the video...", audio_name)
    try:
        transcription = []
        audio = open(audio_name+".mp3", "rb")

        input = {
            "audio": audio,
            "batch_size": 64
        }

        output = replicate.run(
            replicate_api_url,
            input=input
        )
        transcription_with_index = ""

        for i, chunk in enumerate(output["chunks"]):
            print("outside", chunk['timestamp'], type(chunk['timestamp']))
            try:
                if chunk['timestamp'] is not None and len(chunk['timestamp']) == 2:
                    print("try", chunk['timestamp'][0], chunk['timestamp'][1])
                    start_time = seconds_to_hhmmss(chunk['timestamp'][0])
                    end_time = seconds_to_hhmmss(chunk['timestamp'][1])
                    text = chunk['text']
                    transcription.append({
                        "text": text,
                        "start_time": start_time,
                        "end_time": end_time
                    })
                    transcription_with_index += f"{i+1}.[{start_time}-{end_time}] {text}\n"

                else:
                    print("Invalid timestamp:", chunk['timestamp'])
            except Exception as e:
                print("except", str(e))
        print("Transcription creation completed using replicate.")
        return {"transcription": transcription, "transcription_with_index": transcription_with_index, "language_code": "en"}
    except Exception as e:
        print("Transcription creation failed using replicate.", str(e))
        return {"transcription": [], "transcription_with_index": "", "language_code": "en"}


def filter_using_openai(user_prompt):
    print("Filtering transcription using OpenAI GPT...")
    try:
        response = client.chat.completions.create(
            model="o1",
            messages=[{
                "role": "system", "content": "You are an expert content reviewer."
            }, {
                "role": "user", "content": user_prompt
            }]
        )
        generated_text = response.choices[0].message.content
        start = generated_text.find('[') + 1
        end = generated_text.rfind(']')
        content = generated_text[start:end]
        try:
            generated_text = json.loads('[' + content + ']')
        except json.JSONDecodeError as e:
            print(str(e))
            generated_text = False
        print("OpenAI filtering completed.")
        return generated_text
    except Exception as e:
        print(str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to filter transcription using OpenAI {str(e)}")


def filter_using_gemini(user_prompt):
    print("Filtering transcription using Gemini AI...")
    try:
        response = genai.GenerativeModel(
            "gemini-1.5-flash").generate_content(user_prompt)
        start = response.text.find('[') + 1
        end = response.text.rfind(']')
        content = response.text[start:end]
        try:
            filtered_transcript = json.loads('[' + content + ']')
        except json.JSONDecodeError as e:
            print(str(e))
            filtered_transcript = []
        print("Gemini filtering completed.")
        return filtered_transcript
    except Exception as e:
        print(str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to filter transcription using Gemini {str(e)}")


def parse_time(time_str):
    try:
        h, m, s = map(int, time_str.split(":"))
        result = timedelta(hours=h, minutes=m, seconds=s)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error parsing time: {str(e)}")
    return result


def group_transcription_with_index(transcription, index_array):
    result = []
    start = index_array[0]
    end = index_array[0]

    # First, find the sequences
    for i in range(1, len(index_array)):
        if index_array[i] == index_array[i - 1] + 1:
            end = index_array[i]
        else:
            if start == end:
                result.append(str(start))
            else:
                result.append(f"{start}-{end}")
            start = index_array[i]
            end = index_array[i]

    # Add the last sequence or number
    if start == end:
        result.append(str(start))
    else:
        result.append(f"{start}-{end}")

    # Now, process the sequences with transcription
    processed_transcriptions = []

    # Create a transcription dictionary for easy lookup
    transcription_dict = {i + 1: transcription[i]
                          for i in range(len(transcription))}

    for sequence in result:
        if '-' in sequence:
            # Sequence: get the first and last elements
            sequence_start, sequence_end = map(int, sequence.split('-'))
            # Adjust start_index if it's not zero
            start_index = sequence_start
            end_index = sequence_end

            # Adjust start_index if it's not zero
            if start_index > 0:
                start_index -= 1

            text = " ".join(item["text"]
                            for item in transcription[start_index:end_index-1])

            # Get the start_time from the transcription for the first index
            start_time = transcription_dict[sequence_start]["start_time"]
            # Get the end_time from the transcription for the last index
            end_time = transcription_dict[sequence_end]["end_time"]

            processed_transcriptions.append({
                "start_time": start_time,
                "end_time": end_time,
                "text": text
            })
        else:
            # Single element: process normally
            index = int(sequence)
            processed_transcriptions.append({
                "start_time": transcription_dict[index]["start_time"],
                "end_time": transcription_dict[index]["end_time"],
                "text": transcription_dict[index]["text"]
            })

    return processed_transcriptions


def group_transcription(transcription):
    print("Grouping transcription based on timestamps...")
    try:
        grouped = []
        current_group = None
        for entry in transcription:
            start_time = parse_time(entry["start_time"])
            if not current_group:
                current_group = {
                    "text": entry["text"],
                    "start_time": entry["start_time"],
                    "end_time": entry["end_time"]
                }
            else:
                last_end_time = parse_time(current_group["end_time"])
                if start_time == last_end_time:
                    current_group["text"] += " " + entry["text"]
                    current_group["end_time"] = entry["end_time"]
                else:
                    grouped.append(current_group)
                    current_group = {
                        "text": entry["text"],
                        "start_time": entry["start_time"],
                        "end_time": entry["end_time"]
                    }
        if current_group:
            grouped.append(current_group)
        print("Transcription grouping completed.")
        return grouped
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to group transcription {str(e)}")


def get_random_name_with(ext=""):
    if (ext != ""):
        result = f"{str(round(time.time() * 1000))}.{ext}"
    else:
        result = str(round(time.time() * 1000))
    return result


def download_video(video_id):
    print("Downloading video from YouTube...")
    try:
        video_name = get_random_name_with("mp4")
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        ydl_opts = {
            'format': 'worst',
            'outtmpl': video_name,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print("Video downloaded successfully.")
        return video_name
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to download {str(e)}")


def clip_and_combine_video(video_name, transcription):
    print("Clipping and combining video segments...")
    try:
        clip = VideoFileClip(video_name)
        video_duration = clip.duration  # Get total duration
        subclips = []

        for timestamp in transcription:
            start_time = timestamp["start_time"]
            end_time = timestamp["end_time"]

            start_seconds = sum(int(x) * 60 ** i for i,
                                x in enumerate(reversed(start_time.split(":"))))
            end_seconds = sum(int(x) * 60 ** i for i,
                              x in enumerate(reversed(end_time.split(":"))))

            # If end_seconds exceeds video duration, floor and subtract 1 second
            if end_seconds > video_duration:
                print(
                    f"End time {end_seconds} exceeds video duration {video_duration}. Adjusting to {video_duration}.")
                end_seconds = video_duration - 1

            if start_seconds >= end_seconds:  # Avoid invalid cases
                print(
                    f"Skipping invalid clip: start {start_seconds}, end {end_seconds}")
                continue

            subclip = clip.subclip(start_seconds, end_seconds)
            subclips.append(subclip)

        print("Video clipping and combining completed.")
        return subclips
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clip and combine video {str(e)}")


def create_video_link(video_name):
    print("Creating video link for AWS S3 upload...")
    try:
        with open(video_name, 'rb') as f:
            s3_client.upload_fileobj(f, bucket_name, video_name)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': video_name},
            ExpiresIn=3600
        )
        print("Video link created successfully.")
        return presigned_url
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create aws video url {str(e)}")


def get_another_transcription_language_code(video_id):
    print("Checking for available transcription languages...")
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_langauge = ""
        for transcriptItem in transcript_list:
            if transcriptItem.language_code:
                available_langauge = transcriptItem.language_code
                break
        print("Available transcription language found.")
        return available_langauge
    except:
        print("Failed to check transcription languages.")
        return False


def get_another_transcription(video_id, language_code):
    print(f"Retrieving transcription in language code: {language_code}...")
    try:
        transcription = YouTubeTranscriptApi.get_transcript(
            video_id, languages=[language_code])
        print("Transcription retrieved in specified language.")
        return transcription
    except:
        print("Failed to retrieve transcription.")
        return False


def convert_text(text, language_code):
    print(f"Translating text to English from {language_code}...")
    translator = GoogleTranslator(source=language_code, target="en")
    result = translator.translate(text)
    print("Text translation completed.")
    return result


def convert_transcription(transcription, language_code):
    print(f"Converting transcription to English from {language_code}...")
    try:
        translator = GoogleTranslator(source=language_code, target="en")
        converted_transcription = []
        for entry in transcription:
            translated_text = translator.translate(entry['text'])
            converted_transcription.append({
                'start_time': seconds_to_hhmmss(entry['start']),
                'end_time': seconds_to_hhmmss(entry['start'] + entry['duration']),
                'text': translated_text
            })
        print("Transcription conversion completed.")
        return converted_transcription
    except:
        print("Failed to convert transcription.")
        return False


def check_youtube_ids(url):
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        video_id = query_params.get("v", [None])[0]
        if video_id:
            # Check if the video is valid
            params = {
                "part": "id",
                "id": video_id,
                "key": youtube_api_key
            }
            try:
                response = requests.get(YOUTUBE_VIDEO_API, params=params).json()
                if "items" in response and len(response["items"]) > 0:
                    return [video_id]
                return []
            except:
                return []

        playlist_id = query_params.get("list", [None])[0]
        if playlist_id:
            # Check if the playlist is valid
            params = {
                "part": "id",
                "id": playlist_id,
                "key": youtube_api_key
            }
            response = requests.get(YOUTUBE_PLAYLIST_API, params=params).json()
            if "items" in response and len(response["items"]) > 0:
                video_ids = []
                next_page_token = None

                while True:
                    params = {
                        "part": "snippet",
                        "playlistId": playlist_id,
                        "maxResults": 50,
                        "key": youtube_api_key,
                        "pageToken": next_page_token,
                    }
                    response = requests.get(
                        YOUTUBE_PLAYLIST_ITEMS_API, params=params).json()

                    for item in response.get("items", []):
                        vid_id = item["snippet"]["resourceId"]["videoId"]
                        video_ids.append(vid_id)

                    next_page_token = response.get("nextPageToken")
                    if not next_page_token:
                        break

                return video_ids

        return []

    except Exception as e:
        print(f"Error occurred: {e}")
        return []


class VideoRequest(BaseModel):
    prompt_link: str
    prompt: str


@app.post("/")
async def process_video(video_request: VideoRequest):
    video_url = video_request.prompt_link
    user_prompt = video_request.prompt
    videos = []
    transcription_with_index = ""
    video_ids = check_youtube_ids(video_url)
    for video_id in video_ids:
        video_detail = get_video_details(video_id)
        if not video_detail:
            continue
        transcription = get_transcription(video_id)
        if not transcription or len(transcription)==0:
            continue
        video_detail["transcription"]=transcription
        videos.append(video_detail)


    for video_index,video in enumerate(videos):
        transcription=video["transcription"]
        for transcript_index, transcript in enumerate(transcription):
            text = transcript["text"]
            transcription_with_index += f"{video_index+1}.{transcript_index+1} {text} \n"
            if (transcript_index == len(transcription)-1):
                transcription_with_index += "\n"

    prompt = f"""
You are given a set of video titles, each associated with a unique number (e.g., "1", "2", "3", etc.), and each video contains multiple transcript segments, identified by sub-numbers (e.g., "1.1", "1.2", "1.3", "2.6", "2.8", etc.). 
So there are mainly two task:
1.creating new, meaningful stories by combining transcripts from multiple videos into a cohesive narrative
2.extracting multiple relevant transcripts based on user input. 
Always prioritize story creation or transcript extraction unless the user explicitly asks for something else. Don't skip any video, you can skip irrelevant transcription

Expected output:
[
  {{ "1": ["1.1", "1.2", "1.3"] }},
  {{ "2": ["2.6", "2.8", "2.9"] }}
]

User prompt: {user_prompt}

Transcript Data:
{transcription_with_index}

return output only in array of object.
"""
    print(prompt)
    index_array = filter_using_openai(
        prompt)
    print(index_array)
    final_transcription = []
    try:
        for item in index_array:
            try:
                for video_index, transcription_index_with_video_index_array in item.items():
                    video_index_integer=int(video_index)-1
                    video_detail = {"title": videos[video_index_integer]["title"], 
                                    "id": videos[video_index_integer]["id"], 
                                    "duration": videos[video_index_integer]["duration"], 
                                    "all_transcription": videos[video_index_integer]["transcription"], 
                                    "transcription": []}
                    try:
                        for transcription_index_with_video_index in transcription_index_with_video_index_array:
                            transcription_index_integer=int(transcription_index_with_video_index.split(".")[1])-1
                            video_detail["transcription"].append(videos[video_index_integer]["transcription"][transcription_index_integer])
                            
                        final_transcription.append(video_detail)
                    except Exception as e:
                        print(str(e),videos[video_index_integer]["transcription"][transcription_index_integer],
                                    videos[video_index_integer],
                                    videos[video_index_integer]["transcription"]
                                        )
                        continue
            except Exception as e:
                print(str(e))
                continue
    except Exception as e:
        print(str(e))

    for video in final_transcription:
        transcription = group_transcription(video["transcription"])
        video["transcription"] = transcription
    
    return {"video_link": "", "video_details":final_transcription,"global_error":global_error}
