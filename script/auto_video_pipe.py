import argparse

'''
1. Removing blank segments from the video
2. Adding subtitles to the video
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic video processing pipeline')
    parser.add_argument('input_file', help='Input file path')
    args = parser.parse_args()
    
