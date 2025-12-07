import os
import subprocess

files=os.listdir("videos")
# print(files)

for file in files:
    tutorial_number= file.split("_480p")[0].split("#")[1]
    file_name=file.split("online_")[1].split("  Sigma")[0]
    # print(f"{tutorial_number}{file_name}")
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])