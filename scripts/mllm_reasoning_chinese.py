"""
CuriGPT
===========================

High-level task reasoning for CURI manipulation via Multimodal LLM (Qwen-VL-Max).
"""

from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from audio_assistant import AudioAssistant
import pyrealsense2 as rs
import cv2
import time
import numpy as np
import os


local_file_path1 = '../assets/img/curigpt_demo_huawei.png'
realtime_file_path = "../assets/img/realtime_rgb/realtime_rgb.png"

# base_multimodal_prompt = [
#     # Example 1: Speech-to-Speech Reasoning
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "下午好, 你能描述下你现在看到的场景吗?"},
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": "好的，我看到一个白色的桌子上放着一些物品。左上角有一个橙色的洗涤剂，中间有一个绿色的绿茶罐子，一个蓝色的午餐肉罐头和一个红色的盘子，盘子里放着一根香蕉。"}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "你觉得桌上的哪个物品适合作为吃火锅时的配菜呢？"},
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{"text": "我认为午餐肉罐头适合作为吃火锅时的配菜，因为它易于烹饪，可以在火锅中快速煮熟，并可以与其他食材搭配，增加火锅的多样性"}]
#     },
#     # {
#     #     "role": "user",
#     #     "content": [
#     #         {"image": local_file_path1},
#     #         {"text": "那香蕉为什么不适合呢？"},
#     #     ]
#     # },
#     # {
#     #     "role": "assistant",
#     #     "content": [{"text": "因为香蕉是一种水果，其口感和味道与火锅的其他食材相比可能会显得不太协调。此外，香蕉在高温下容易变软和糊化，这可能会影响其口感和营养价值。因此，香蕉通常不是火锅的常见配菜"}]
#     # },
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "我想清洗一下红色盘子，你可以把洗涤剂递给我吗"},
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{"text": "好的，没问题。我将把洗涤剂瓶子递给你。"}]
#     },
# ]

base_multimodal_prompt = [
    {
        "role": "user",
        "content": [
            {"image": local_file_path1},
            {"text": "下午好, 你能描述下你现在看到的场景吗?"},
        ]
    },
    {
        "role": "assistant",
        "content": [{
            "text": "好的，我看到一个白色的桌子上放着一些物品。左上角有一个橙色的洗涤剂，中间有一个绿色的绿茶罐子，一个蓝色的午餐肉罐头和一个红色的盘子，盘子里放着一根香蕉。"}]
    },
    {
        "role": "user",
        "content": [
            {"image": local_file_path1},
            {"text": "你觉得桌上的哪个物品适合作为吃火锅时的配菜呢？"},
        ]
    },
    {
        "role": "assistant",
        "content": [{"text": "我认为午餐肉罐头适合作为吃火锅时的配菜，因为它易于烹饪，可以在火锅中快速煮熟，并可以与其他食材搭配，增加火锅的多样性"}]
    },
    # {
    #     "role": "user",
    #     "content": [
    #         {"image": local_file_path1},
    #         {"text": "那香蕉为什么不适合呢？"},
    #     ]
    # },
    # {
    #     "role": "assistant",
    #     "content": [{"text": "因为香蕉是一种水果，其口感和味道与火锅的其他食材相比可能会显得不太协调。此外，香蕉在高温下容易变软和糊化，这可能会影响其口感和营养价值。因此，香蕉通常不是火锅的常见配菜"}]
    # },
    {
        "role": "user",
        "content": [
            {"image": local_file_path1},
            {"text": "我想清洗一下红色盘子，你可以把洗涤剂递给我吗"},
        ]
    },
    {
        "role": "assistant",
        "content": [{"text": "好的，没问题。我将把洗涤剂瓶子递给你。"}]
    },
]


def save_color_images(folder_path, save_interval=1):
    """
    实时捕捉并保存来自 Realsense D435 相机的彩色图片。

    参数:
        folder_path (str): 保存图像的文件夹路径。
        save_interval (int): 保存图像的时间间隔（秒）。
    """
    # 确保保存路径存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 配置颜色流
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动流
    pipeline.start(config)

    try:
        last_saved_time = time.time()
        for i in range(2):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 将图像转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())

            img_name = "realtime_rgb.png"
            cv2.imwrite(os.path.join(folder_path, img_name), color_image)
            time.sleep(1)

        print(f"RGB image saved")


            # 获取当前时间
            # current_time = time.time()
            # if current_time - last_saved_time >= save_interval:
            #     # 保存图像
            #     img_name = "realtime_rgb.png"
            #     cv2.imwrite(os.path.join(folder_path, img_name), color_image)
            #     last_saved_time = current_time
            #     print(f"Saved {img_name}")

            # 显示图像
            # cv2.imshow('Realsense', color_image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    finally:
        # 停止流
        pipeline.stop()
        cv2.destroyAllWindows()

def add_bbox_patch(ax, draw, bbox, color, w, h, caption):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    # draw.rectangle([x1, y1, x2, y2], outline=color, width=10)

    # Annotate the bounding box with a caption
    ax.text(x1, y1, caption, color=color, weight='bold', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.1'))

def plot_image_with_bbox(image_path, response):
    """Parse the bbox coordinates from the response of mllm and plot them on the image.

    Parameters:
    image_path (str): The path to the image file.
    response (str): The response from the multimodal large language model.
    """

    # Parse the JSON string.
    try:
        output_data = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Load the image.
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    w, h = image.size # get the width and height od the image size


    # Create a plot.
    fig, ax = plt.subplots()

    # Display the image.
    ax.imshow(image)

    # Go through each action and plot the bounding boxes.
    if output_data['robot_actions'] is not None:
        for action in output_data['robot_actions']:
            bbox1 = action['parameters']['arg1']['bbox_coordinates']
            caption1 = action['parameters']['arg1']['description']

            # check if arg2 exists
            if 'arg2' in action['parameters']:
                bbox2 = action['parameters']['arg2']['bbox_coordinates']
                caption2 = action['parameters']['arg2']['description']

                add_bbox_patch(ax, draw, bbox1, 'red', w, h, caption1)
                add_bbox_patch(ax, draw, bbox2, 'blue', w, h, caption2)

            else:
                add_bbox_patch(ax, draw, bbox1, 'red', w, h, caption1)

            # Show the plot with the bounding boxes.
        plt.show()
        # image.show()
    # else:
    #     print("The action is None.")



def single_multimodal_call(base_prompt, query, log=True, return_response=True):

    """single round multimodal conversation call with CURIGPT.
    """

    # Make a copy of the base prompt to avoid modifying the original
    new_prompt = base_prompt.copy()
    new_prompt.append({"role": "user", "content": [{"image": realtime_file_path}, {"text": query}]})
    # response_dict = dashscope.MultiModalConversation.call(model='qwen-vl-max',
    #                                                  messages=new_prompt)

    # response_dict = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
    #                                            messages=new_prompt, top_p=0.1, top_k=10)

    response_dict = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                               messages=new_prompt, top_p=0.9, top_k=100)
    # response_dict = MultiModalConversation.call(model='qwen-vl-plus',
    #                                            messages=new_prompt, top_p=0.1, top_k=10)

    if response_dict.status_code == HTTPStatus.OK:
        response = response_dict[
            "output"]["choices"][0]["message"]["content"]
        print("CURI response:\n", response)

        # Plot the image with bounding boxes if the response contains actions.
        # plot_image_with_bbox(local_file_path1, response)
    else:
        print(response_dict.code)  # The error code.
        print(response_dict.message)  # The error message.

    if return_response:
        return response

def multiple_multimodal_call(base_prompt, query, rounds = 10, log=True, return_response=False):

    """multiple rounds of  multimodal conversation call with CURIGPT.
    """
    base_prompt.append({"role": "user", "content": [{'image': local_file_path1}, {"text": query}]})

    for i in range(rounds):
        new_prompt = base_prompt
        # response = dashscope.MultiModalConversation.call(model='qwen-vl-max',
        #                                                  messages=new_prompt)
        #call with local file
        response_dict = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                                   messages=new_prompt)
        response = response_dict[
            "output"]['choices'][0]['message']['content']

        if response_dict.status_code == HTTPStatus.OK:
            print("CURI response:\n", response)
        else:
            print(response_dict.code)  # The error code.
            print(response_dict.message)  # The error message.

        if return_response:
            return response


def get_curi_response(base_multimodal_prompt, rounds=10, prompt_append=False):

    """Get CURI response for a given number of rounds."""
    inference_results = None
    if not prompt_append:
        print("without prompt append")
        for i in range(rounds):
            instruction = input("Please input your instruction: ")
            inference_results = single_multimodal_call(base_multimodal_prompt, instruction, log=True, return_response=False)

    else:
        if rounds < 1:
            raise ValueError("Number of rounds must be at least 1.")

        for i in range(rounds):
            instruction = input("Please input your instruction: ")

            # Single call for each round
            if i == 1:
                # For the first round or if not returning response, just update the prompt
                inference_results = single_multimodal_call(base_multimodal_prompt, instruction, log=True,
                                                           return_response=False)

            else:
                # For subsequent rounds, get the response as well to append to the prompt
                new_prompt, response_dict = single_multimodal_call(base_multimodal_prompt, instruction, log=True,
                                                                   return_response=True)
                if 'output' in response_dict and 'choices' in response_dict['output'] and len(
                        response_dict['output']['choices']) > 0:
                    # Append the model's response to the prompt for the next round
                    response_content = response_dict['output']['choices'][0]['message']['content']
                    response_role = response_dict['output']['choices'][0]['message']['role']
                    new_prompt.append({'role': response_role, 'content': [{'text': response_content}]})
                base_multimodal_prompt = new_prompt

def get_curi_response_with_audio(api_key, base_url, user_input_filename, curigpt_output_filename,
                                 base_multimodal_prompt, rounds=10,
                                 prompt_append=False):

    """Get CURI response with audio input and output."""

    assistant = AudioAssistant(api_key, base_url, user_input_filename, curigpt_output_filename)

    rgb_img_path = "../assets/img/realtime_rgb"

    inference_results = None
    if not prompt_append:
        # print("without prompt append")
        for i in range(rounds):
            save_color_images(rgb_img_path)
            assistant.record_audio()
            transcription = assistant.transcribe_audio()
            instruction = transcription
            response = single_multimodal_call(base_multimodal_prompt, instruction, log=True, return_response=True)

            print("CURI audio response:\n", response)

                # Go through each action and plot the bounding boxes.
            if response is not None:
                assistant.text_to_speech(response)

    else:
        if rounds < 1:
            raise ValueError("Number of rounds must be at least 1.")

        for i in range(rounds):
            instruction = input("Please input your instruction: ")

            # Single call for each round
            if i == 1:
                # For the first round or if not returning response, just update the prompt
                inference_results = single_multimodal_call(base_multimodal_prompt, instruction, log=True,
                                                           return_response=False)

            else:
                # For subsequent rounds, get the response as well to append to the prompt
                new_prompt, response_dict = single_multimodal_call(base_multimodal_prompt, instruction, log=True,
                                                                   return_response=True)
                if 'output' in response_dict and 'choices' in response_dict['output'] and len(
                        response_dict['output']['choices']) > 0:
                    # Append the model's response to the prompt for the next round
                    response_content = response_dict['output']['choices'][0]['message']['content']
                    response_role = response_dict['output']['choices'][0]['message']['role']
                    new_prompt.append({'role': response_role, 'content': [{'text': response_content}]})
                base_multimodal_prompt = new_prompt



if __name__ == '__main__':
    # multiple_call_with_local_file()
    # rounds = int(input("How many rounds of conversation do you want? "))
    # get_curi_response(base_multimodal_prompt, prompt_append=False)

    api_key = "sk-59XTKMjGzgbgSJjpC9D770A52eBd4d68902223561eE3F242"
    base_url = "https://www.jcapikey.com/v1"
    user_input_filename = '../assets/chat_audio/user_input.wav'
    curigpt_output_filename = '../assets/chat_audio/curigpt_output.mp3'
    # curigpt_output_filename = '../assets/chat_audio/huawei_demo3.mp3'
    get_curi_response_with_audio(api_key, base_url, user_input_filename, curigpt_output_filename,base_multimodal_prompt, rounds=10, prompt_append=False)