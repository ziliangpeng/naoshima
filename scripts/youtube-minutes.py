import os
import statsd
import isodate
from googleapiclient.discovery import build

s = statsd.StatsClient("localhost", 8125)

# Replace with your own API key
API_KEY = os.getenv('YOUTUBE_TOKEN')
PLAYLIST_ID = 'PLDvBZlLoGspw2LO7vNUZ8UnpHCUsrzlz3'
PLAYLIST_ID = 'PLNfnQryZV738VB6300-sehYMLlJT4ndBo'

def get_playlist_items(youtube, playlist_id):
    items = []
    request = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=playlist_id,
        maxResults=50
    )
    while request is not None:
        response = request.execute()
        items.extend(response['items'])
        request = youtube.playlistItems().list_next(request, response)
    return items

def get_video_durations(youtube, video_ids):
    durations = []
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part='contentDetails',
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute()
        for video in response['items']:
            duration = isodate.parse_duration(video['contentDetails']['duration'])
            durations.append(duration)
    return durations

if __name__ == '__main__':
    if API_KEY is None:
        print('Error: YOUTUBE_TOKEN environment variable is not set.')
    else:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        
        # Get all items in the playlist
        items = get_playlist_items(youtube, PLAYLIST_ID)
        
        # Extract video IDs
        video_ids = [item['contentDetails']['videoId'] for item in items]
        
        # Get durations of all videos
        durations = get_video_durations(youtube, video_ids)
        
        # Calculate total duration in minutes
        total_duration = int(sum(d.total_seconds() for d in durations) / 60)

        
        print(f'Total duration of all videos in playlist: {total_duration} minutes')
        s.gauge(f"youtube.publicplaylist.total_duration", total_duration)