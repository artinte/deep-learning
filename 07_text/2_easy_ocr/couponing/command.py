import os
import re
import time
import random
import subprocess

EVENTS = {
    'KEY_POWER': '26'
}


def check_adb():
    search_list = ['adb', 'adb.bat']
    with open(os.devnull, 'w') as devnull:
        for adb in search_list:
            try:
                subprocess.run([adb, 'version'],
                               stdout=devnull,
                               stderr=devnull,
                               check=False)
                return True
            except FileNotFoundError:
                continue
        return False


def print_adb_version():
    try:
        result = subprocess.run([
            'adb', 'version'
        ], capture_output=True,
            text=True, check=True)
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(e)


def get_serials():
    result = {}
    ret = subprocess.run(['adb', 'devices'],
                         stdout=subprocess.PIPE,
                         text=True,
                         check=True)
    if ret.returncode == 0:
        lines = ret.stdout.split('\n')
        for line in lines:
            temp = re.split(r'[\s\t]+', line)
            if len(temp) != 2:
                continue
            result[temp[0]] = temp[1].lower()
    return result

def phone_wake_up(uuid):
    subprocess.run(['adb', '-s', uuid, 'shell', 'input', 'keyevent', EVENTS['KEY_POWER']])
    time.sleep(random.randint(3, 5))


def phone_size(uuid):
    ret = subprocess.run(['adb', '-s', uuid, 'shell', 'wm', 'size'], stdout=subprocess.PIPE)
    if ret.returncode == 0:
        decoded_data = ret.stdout.decode('utf-8')
        match = re.search(r'(\d+)x(\d+)', decoded_data)
        if match:
            width, height = match.groups()
            print(f'Phone {uuid} size: {width}x{height}')
            return (int(width), int(height))
    return (0, 0)

def swipe_up(uuid, size):
    (width, height) = size
    x1 = width // 2 + random.randint(0, width // 10)
    x2 = width // 2 + random.randint(0, width // 10)
    y1 = height * 3 // 4 + random.randint(0, height // 10)
    y2 = height // 4 + random.randint(0, height // 10)
    duration = random.randint(100, 500)
    subprocess.run(['adb', '-s', uuid, 'shell', 'input', 'swipe',
                    str(x1), str(y1), str(x2), str(y2), str(duration)])
    print(f'Phone {uuid} swipe up: ({x1}, {y1}) ({x2}, {y2})')
    time.sleep(random.randint(3, 5))
    
def lanuch_app(uuid, activity):
    subprocess.run(['adb', '-s', uuid, 'shell', 'am', 'start', activity])
    time.sleep(random.randint(3, 5))
