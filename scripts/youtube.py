import os
import statsd
from googleapiclient.discovery import build

s = statsd.StatsClient("localhost", 8125)

# Replace with your own API key
API_KEY = os.getenv('YOUTUBE_TOKEN')
PLAYLIST_ID = 'PLDvBZlLoGspw2LO7vNUZ8UnpHCUsrzlz3'
PLAYLIST_ID = 'PLNfnQryZV738VB6300-sehYMLlJT4ndBo'

def get_video_count(api_key, playlist_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    request = youtube.playlists().list(
        part='contentDetails',
        id=playlist_id
    )
    response = request.execute()
    
    if 'items' in response and len(response['items']) > 0:
        playlist = response['items'][0]
        video_count = playlist['contentDetails']['itemCount']
        return video_count
    else:
        return None

if __name__ == '__main__':
    count = get_video_count(API_KEY, PLAYLIST_ID)
    if count is not None:
        print(f'Total videos in playlist: {count}')
        s.gauge(f"youtube.publicplaylist.count", count)
    else:
        print('Playlist not found or an error occurred.')
