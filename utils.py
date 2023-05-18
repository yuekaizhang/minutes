import datetime
from typing import List
import soundfile
import os

def convert_to_wav(in_filename: str) -> str:
    """Convert the input audio file to a wave file"""
    file_root, _ = os.path.splitext(in_filename)
    out_filename = file_root + ".wav"
    # check if out_filename exists
    if os.path.exists(out_filename):
        speech, _ = soundfile.read(out_filename)
        return speech
    if '.mp3' in in_filename:
        _ = os.system(f"ffmpeg -y -i '{in_filename}' -acodec pcm_s16le -ac 1 -ar 16000 '{out_filename}'")
    else:
        _ = os.system(f"ffmpeg -hide_banner -y -i '{in_filename}' -ar 16000 '{out_filename}'")
    speech, _ = soundfile.read(out_filename)
    print(f"load speech shape {speech.shape}")
    return speech



def chunk_strings(input_list: List[str], output_chunk_length: int) -> List[str]:
    output_list, chunk_idx = [], [0]
    current_chunk = ""
    
    for idx, string in enumerate(input_list):
        if len(current_chunk) + len(string) + 1 <= output_chunk_length:
            if current_chunk:
                current_chunk += " " + string
            else:
                current_chunk = string
        else:
            output_list.append(current_chunk)
            current_chunk = string
            chunk_idx.append(idx)
            
    if current_chunk:
        output_list.append(current_chunk)
    
    return output_list, chunk_idx