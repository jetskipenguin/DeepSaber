"""
Taken from :https://gist.github.com/DaBlincx/f037bbfb661797f1f670f987c3a8c1e6
"""

from datetime import datetime, timedelta
import os
import requests
import shutil
import tqdm
from urllib.parse import quote

downloadfolder = "downloads"

historyfile = "history.txt"

folders = [downloadfolder]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

if not os.path.exists(historyfile):
    open(historyfile,"w").close()

with open(historyfile, "r") as hf:
    downloaded = hf.readlines()

for i in range(len(downloaded)):
    downloaded[i] = downloaded[i].strip()


def writeHistory(songid, song_upload_date):
    """writes given song file name to download history"""
    with open(historyfile,"a") as hf:
        hf.write(f"{songid},{song_upload_date}\n")
    downloaded.append(songid.strip())

def getRating(numUpvotes: int, numDownvotes: int) -> float:
    """calculates the rating of a song based on upvotes and downvotes"""
    if numUpvotes + numDownvotes <= 100:
        return 0.0
    return round(numUpvotes / (numUpvotes + numDownvotes), 2)

def downloadSong(song):
    songfilename = f"{song['id'] } ({song['metadata']['levelAuthorName']} - {song['metadata']['songName']})".replace("/", "-").replace(
        "\\", "-").replace(":", "-").replace("*", "-").replace("?", "-").replace("\"", "-").replace("<", "-").replace(">", "-").replace("|", "-")
    if f"{song['id']},{song['uploaded']}".strip() not in downloaded:
        rating: float = getRating(song['stats']['upvotes'], song['stats']['downvotes'])
        if rating < 80.0:
            print(f"Skipping {song['metadata']['songName']} by {song['metadata']['levelAuthorName']} | ID: {song['id']} | Rating: {rating} | Upvotes: {song['stats']['upvotes']} | Downvotes: {song['stats']['downvotes']}")
            writeHistory(song['id'], song['uploaded'])
            return

        print(f"Downloading {song['metadata']['songName']} by {song['metadata']['levelAuthorName']} | ID: {song['id']} | URL: {song['versions'][0]['downloadURL']}")
        finalsongurl = song['versions'][0]['downloadURL']

        songfile = requests.get(finalsongurl, stream=True)
        total_size = int(songfile.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm.tqdm(total=total_size, unit="iB", unit_scale=True)

        
        with open(f"{downloadfolder}/{songfilename}.zip", "wb") as f:
            for data in songfile.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        print(f"Downloaded '{songfilename}'")

    else:
        print(f"Already downloaded {songfilename}")


def main(date: str):
    url = f"https://api.beatsaver.com/maps/latest?before={quote(date, safe='')}&sort=LAST_PUBLISHED&automapper=false"
    response = requests.get(url)
    print(f"fetched {url}")
    maps = response.json()
    lsong = 0
    for song in maps["docs"]:
        try: downloadSong(song)
        except Exception as e: exit(e)
        lsong = song
    main(lsong["uploaded"])


if __name__ == "__main__":
    print("Starting scraper")
    start_date: datetime = datetime.now() - timedelta(days=30)
    start_date_str: str = start_date.strftime('%Y-%m-%dT%H:%M:%S+00:00').strip()
    main(start_date_str)