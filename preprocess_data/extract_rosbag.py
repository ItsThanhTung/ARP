import argparse
import numpy as np
import os
import rosbag
import tqdm

from PIL import Image


list_extracted_topics = [
    'fisheye_front', 
    'fisheye_front.albedo', 
    'fisheye_front.depth/logdepth', 
    'fisheye_front.normal', 
    'fisheye_front.seg', 
    'fisheye_left', 
    'fisheye_left.albedo', 
    'fisheye_left.depth/logdepth', 
    'fisheye_left.normal', 
    'fisheye_left.seg', 
    'fisheye_rear', 
    'fisheye_rear.albedo', 
    'fisheye_rear.depth/logdepth', 
    'fisheye_rear.normal', 
    'fisheye_rear.seg', 
    'fisheye_right', 
    'fisheye_right.albedo', 
    'fisheye_right.depth/logdepth', 
    'fisheye_right.normal', 
    'fisheye_right.seg', 
]


def get_folder(root_path, topic):
    return f"{root_path}/{topic.replace('/', '.')}"


def extract(bag_filepath, extract_folder):

    if not os.path.exists(bag_filepath):
        raise FileNotFoundError(f"not found {bag_filepath}")
    print(f"loading '{bag_filepath}'")
    bag = rosbag.Bag(bag_filepath)
    total_messages = bag.get_message_count()
    print(f"total_messages={total_messages}")
    print("list of topics:")
    for topic_name, topic_type in bag.get_type_and_topic_info()[1].items():
        print(f"- {topic_name} ({topic_type.msg_type})")

    name = os.path.basename(bag_filepath)
    extract_folder = os.path.join(extract_folder, name)
    print(f"extracting messages to folder '{extract_folder}'")
    os.makedirs(extract_folder, exist_ok=True)
    for topic in list_extracted_topics:
        os.makedirs(get_folder(extract_folder, topic), exist_ok=True)

    for topic, message, timestamp in tqdm.tqdm(bag.read_messages(), total=total_messages):
        if topic not in list_extracted_topics:
            continue
        filepath = f"{get_folder(extract_folder, topic)}/{timestamp}"
        if message._type == 'sensor_msgs/Image':
            np_image = np.zeros((message.height, message.width, 3), dtype='uint8')
            dt = np.dtype("uint16")
            dt = dt.newbyteorder('<' if message.is_bigendian == 0 else '>')
            np_red_image = np.frombuffer(message.data, dtype=dt)
            np_red_image = np_red_image.astype(np.dtype("uint8"))
            np_red_image = np.reshape(np_red_image, (message.height, message.width))
            np_image[:, :, 0] = np_red_image
            pil_image = Image.fromarray(np_image)
            pil_image.save(f"{filepath}.png")
        elif message._type == 'sensor_msgs/CompressedImage':
            with open(f"{filepath}.{message.format}", 'wb') as fp:
                fp.write(message.data)
        else:
            raise NotImplementedError(f"not implement {message._type}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ROS Bag')
    parser.add_argument('bag_filepath', type=str, help='file path of the ROS bag')
    parser.add_argument(
        'extract_folder', nargs='?', default='extracted', type=str, 
        help="folder path to save extracted data (default='./extracted/')")
    args = parser.parse_args()
    extract(args.bag_filepath, args.extract_folder)
