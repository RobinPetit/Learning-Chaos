import numpy as np
from shutil import copyfile
from os import remove
from os.path import exists
import imageio
from memory import STATE_TYPE

MEMORY_PATH = "mem.dat"
TMP_MEMORY_PATH = MEMORY_PATH + ".copy"

MEM_SIZE = 1000000  #1e6
SUB_ARRAY_SIZE = 10000  # 1e4

OUTPUT_PATH = 'DQN.mp4'  # TO ADAPT

########## DONT USE
def copy_file():
    if exists(TMP_MEMORY_PATH):
        remove(TMP_MEMORY_PATH)
    copyfile(MEMORY_PATH, TMP_MEMORY_PATH)
    MEMORY_PATH = TMP_MEMORY_PATH
##########

def float16_to_int8(array: np.ndarray):
    return np.array(np.round(array*255), dtype=np.uint8)

def save_to_file(imgs: int, path: str):
    output_mp4 = imageio.get_writer(path, mode="I", macro_block_size=42)
    for img in imgs:
        output_mp4.append_data(img)
    output_mp4.close()

memory = np.memmap(MEMORY_PATH, mode="r", shape=(MEM_SIZE, 84, 84), dtype=STATE_TYPE)
if STATE_TYPE == np.float16:
    memory = float16_to_int8(memory[:SUB_ARRAY_SIZE])
save_to_file(memory[20:1020], OUTPUT_PATH)
