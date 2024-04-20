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

# local_file_path1 = '/home/zhuoli/PycharmProjects/CuriGPT/assets/img/curigpt_test_tabletop.png'
local_file_path1 = '../assets/img/curigpt_demo_huawei.png'
#base prompt for model qwen_vl_chat_v1
# base_multimodal_prompt = [
#     {
#         'role': 'system',
#         'content': [{
#             'text': 'you are an excellent interpreter of human instructions for robot manipulation tasks. Given an '
#                     'instruction and a image about the environment, you first need to respond to human instructions '
#                     'and then select plausible actions to finish the task. '
#                      'ROBOT ACTION LIST is defined as follows: '
#                      'grasp_and_place(arg1, arg2): The robot grasps the object at position arg1 and places it at position arg2. '
#                      'grasp_handover_place(arg1, arg2): The robot grasps the object at position arg1 with the left hand, '
#                      'hands it over to the right hand, and places it at position arg2. '
#                      'grasp_and_give(arg1): The robot grasps the object at position arg1 and gives it to the user. '
#                      'The parameters arg1 and arg2 are the bounding box coordinates of the objects, '
#                      'which are detected by yourself from the given image. Here is an example of the conversation:'
#         }]
#     },
# {
#             "role": "user",
#             "content": [
#                 {'image': local_file_path1},
#                 {"text": "hey CURI, what do you see right now?"},
#             ]
#     },
# {"role": "assistant", "content":
# [{
#             'text': json.dumps({
#                 "robot_response": "now I see a red plate in the center of the table, and an empty spam can, a banana, and a soda can near the plate. There is also a green container on the upper left corner of the table.",
#                 "robot_actions": None
#             }, indent=4)
#         }]},
#     {
#         'role': 'user',
#         'content': [
#             {'image': local_file_path1},
#             {'text': "Can you put the spam can in the container?"}
#         ]
#     },
#     {
#         'role': 'assistant',
#         'content': [{
#             'text': json.dumps({
#                 "robot_response": "Sure thing.",
#                 "robot_actions": [
#                     {
#                         "action": "grasp_and_place",
#                         "parameters": {
#                             "arg1": [137.0, 169.0, 157.0, 209.0],
#                             "arg2": [184.0, 152.0, 204.0, 182.0]
#                         }
#                     }
#                 ]
#             }, indent=4)
#         }]
#     }
# ]

#base prompt for model qwen-vl-plus

# Useful version of the base prompt for the multimodal conversation-2024-04-08
# base_multimodal_prompt = [
#     {
#         "role": "system",
#         "content": [{
#             "text": 'you are an excellent interpreter of human instructions for robot manipulation tasks. Given an '
#                     'instruction and a image about the environment, you first need to respond to human instructions '
#                     'and then select plausible actions to finish the task. '
#                      'ROBOT ACTION LIST is defined as follows: '
#                      'grasp_and_place(arg1, arg2): The robot grasps the object at position arg1 and places it at position arg2. '
#                      'grasp_handover_place(arg1, arg2): The robot grasps the object at position arg1 with the left hand, '
#                      'hands it over to the right hand, and places it at position arg2. '
#                      'grasp_and_give(arg1): The robot grasps the object at position arg1 and gives it to the user. '
#                      'The parameters arg1 and arg2 are the bounding box coordinates of the objects, '
#                      'which are detected by yourself from the given image. Here is an example of the conversation:'
#         }]
#     },
# {
#             "role": "user",
#             "content": [
#                 {"image": local_file_path1},
#                 {"text": "hey CURI, what do you see right now?"},
#             ]
#     },
# {"role": "assistant", "content":
# [{
#             "text": json.dumps({
#                 "robot_response": "now I see a red plate in the center of the table, and an empty spam can, a banana, and a soda can near the plate. There is also a green container on the upper left corner of the table.",
#                 "robot_actions": None
#             }, indent=4)
#         }]},
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "Can you put the spam can in the container?"}
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": json.dumps({
#                 "robot_response": "Sure thing.",
#                 "robot_actions": [
#                     {
#                         "action": "grasp_and_place",
#                         "parameters": {
#                             "arg1": [137.0, 169.0, 157.0, 209.0],
#                             "arg2": [184.0, 152.0, 204.0, 182.0]
#                         }
#                     }
#                 ]
#             }, indent=4)
#         }]
#     }
# ]

# base_multimodal_prompt = [
#     {
#         "role": "system",
#         "content": [{
#             "text": 'You are an advanced interpreter of human instructions for robot manipulation tasks. When provided with '
#                     'an instruction and an image of the environment, your role is to interpret the scene and then execute one '
#                     'action from the ROBOT ACTION LIST, using the bounding box coordinates you determine from the image. '
#                     'The responses should be structured in JSON format with two keys: "robot_response" for your description '
#                     'and "robot_actions" for the selected action. Ensure that the examples you provide are consistent and '
#                     'illustrate correct usage of the bounding box coordinates for arg1 and arg2.'
#                      'ROBOT ACTION LIST is defined as follows:'
#                     """
#                     grasp_and_place(arg1, arg2): The robot grasps the object within the bounding box coordinates arg1 and places it at arg2. '
#                     grasp_handover_place(arg1, arg2): The robot grasps the object within arg1, transfers it to the other hand, and places it at arg2. '
#                     grasp_and_give(arg1): The robot grasps the object within arg1 and hands it to a human user.
#                     """
#                     'Bounding box coordinates (arg1, arg2) should be determined by you based on the image provided. '
#                     'Here are some examples of expected inputs and outputs:'
#         }]
#     },
#     # Example 1: Describing the environment
# {
#             "role": "user",
#             "content": [
#                 {"image": local_file_path1},
#                 {"text": "hey CURI, what do you see right now?"},
#             ]
#     },
# {"role": "assistant", "content":
# [{
#             "text": json.dumps({
#                 "robot_response": "now I see a red plate in the center of the table, and an empty spam can, a banana, and a soda can near the plate. There is also a green recycling bin on the upper left corner of the table.",
#                 "robot_actions": None
#             }, indent=4)
#         }]},
#     #  Example 2: Executing a task based on user instruction
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "Can you give me something to drink?"}
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": json.dumps({
#                 "robot_response": "Sure, you can have the soda to drink.",
#                 "robot_actions": [
#                     {
#                         "action": "grasp_and_give",
#                         "parameters": {
#                             "arg1": [634, 672, 815, 780] # Hypothetical coordinates for the soda can
#                         }
#                     }
#                 ]
#             }, indent=4)
#         }]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "Can you put the banana in the plate?"}
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": json.dumps({
#                 "robot_response": "Yes, I can put the banana in the plate.",
#                 "robot_actions": [
#                     {
#                         "action": "grasp_and_place",
#                         "parameters": {
#                             "arg1": [143, 719, 317, 857], # Hypothetical coordinates for the banana
#                             "arg2": [234, 336, 624, 638] # Hypothetical coordinates for the plate
#                         }
#                     }
#                 ]
#             }, indent=4)
#         }]
#     },
#     # Example 3: Answering a user's question about object placement
# {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "where do you think the empty spam can should go?"}
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": json.dumps({
#                 "robot_response": "The empty spam can should be placed in the recycling bin for disposal.",
#                 "robot_actions": None
#             }, indent=4)
#         }]
#     },
#     # Example 4: Performing a placement task based on the previous question
# {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "can you do it?"}
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": json.dumps({
#                 "robot_response": "Of course, I can do it for you.",
#                 "robot_actions": [
#                     {
#                         "action": "grasp_handover_place",
#                         "parameters": {
#                             "arg1": [143, 719, 317, 857], # Hypothetical coordinates for the spam can
#                             "arg2": [579, 64, 964, 298] # Hypothetical coordinates for the recycling bin
#                         }
#                     }
#                 ]
#             }, indent=4)
#         }]
#     }
# ]


# base_multimodal_prompt = [
#     {
#         "role": "system",
#         "content": [{
#             "text": 'You are an advanced interpreter of human instructions for robot manipulation tasks. When provided with '
#                     'an instruction and an image of the environment, your role is to interpret the scene and then execute one '
#                     'action from the ROBOT ACTION LIST, using the bounding box coordinates you determine from the image. '
#                     'The responses should be structured in JSON format with two keys: "robot_response" for your description '
#                     'and "robot_actions" for the selected action. Ensure that the examples you provide are consistent and '
#                     'illustrate correct usage of the bounding box coordinates for arg1 and arg2.'
#                      'ROBOT ACTION LIST is defined as follows:'
#                     """
#                     grasp_and_place(arg1, arg2): The robot grasps the object within the bounding box coordinates arg1 and places it at arg2. '
#                     grasp_handover_place(arg1, arg2): The robot grasps the object within arg1, transfers it to the other hand, and places it at arg2. '
#                     grasp_and_give(arg1): The robot grasps the object within arg1 and hands it to a human user.
#                     """
#                     'Bounding box coordinates (arg1, arg2) should be determined by you based on the image provided. '
#                     'Here are some examples of expected inputs and outputs:'
#         }]
#     },
#     # Example 1: Describing the environment
# {
#             "role": "user",
#             "content": [
#                 {"image": local_file_path1},
#                 {"text": "hey CURI, what do you see right now?"},
#             ]
#     },
# {"role": "assistant", "content":
# [{
#             "text": json.dumps({
#                 "robot_response": "now I see a red plate in the center of the table, and an empty spam can, a banana, and a soda can near the plate. There is also a green recycling bin on the upper left corner of the table.",
#                 "robot_actions": None
#             }, indent=4)
#         }]},
#     #  Example 2: Executing a task based on user instruction
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "Can you give me something to drink?"}
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": json.dumps({
#                 "robot_response": "Sure, you can have the soda to drink.",
#                 "robot_actions": [
#                     {
#                         "action": "grasp_and_give",
#                         "parameters": {
#                             "arg1": [634, 672, 815, 780] # Hypothetical coordinates for the soda can
#                         }
#                     }
#                 ]
#             }, indent=4)
#         }]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "Can you put the banana in the plate?"}
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": json.dumps({
#                 "robot_response": "Yes, I can put the banana in the plate.",
#                 "robot_actions": [
#                     {
#                         "action": "grasp_and_place",
#                         "parameters": {
#                             "arg1": [143, 719, 317, 857], # Hypothetical coordinates for the banana
#                             "arg2": [234, 336, 624, 638] # Hypothetical coordinates for the plate
#                         }
#                     }
#                 ]
#             }, indent=4)
#         }]
#     }
# ]

# base_multimodal_prompt = [
#     {
#         "role": "system",
#         "content": [{
#             "text": 'you are an advanced interpreter of human instructions for robot manipulation tasks. When '
#                     'provided with an instruction and an image of the environment, your role is to interpret the '
#                     'scene and then select one '
#                     'action from the ROBOT ACTION LIST. '
#                     'The responses should be structured in JSON format with two keys: "robot_response" for your description '
#                     'and "robot_actions" for the selected action. Ensure that the examples you provide are consistent and '
#                     'illustrate correct usage of the bounding box coordinates for arg1 and arg2. '
#                      'ROBOT ACTION LIST is defined as follows: '
#                      'grasp_and_place(arg1, arg2): The robot grasps the object at position arg1 and places it at position arg2. '
#                      'grasp_handover_place(arg1, arg2): The robot grasps the object at position arg1 with the left hand, '
#                      'hands it over to the right hand, and places it at position arg2. '
#                      'grasp_and_give(arg1): The robot grasps the object at position arg1 and gives it to the user. '
#                      'Bounding box coordinates (arg1, arg2) should be determined by you based on the image provided. Here are some examples of expected inputs and outputs:'
#         }]
#     },
# {
#             "role": "user",
#             "content": [
#                 {"image": local_file_path1},
#                 {"text": "hey CURI, what do you see right now?"},
#             ]
#     },
# {"role": "assistant", "content":
# [{
#             "text": json.dumps({
#                 "robot_response": "now I see a red plate in the center of the table, and an empty spam can, a banana, and a soda can near the plate. There is also a green container on the upper left corner of the table.",
#                 "robot_actions": None
#             }, indent=4)
#         }]},
# {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#              {"text": "Can you give me something to drink?"}
#          ]
#      },
#      {
#          "role": "assistant",
#          "content": [{
#              "text": json.dumps({
#                  "robot_response": "Sure, you can have the soda to drink.",
#                  "robot_actions": [
#                     {
#                          "action": "grasp_and_give",
#                          "parameters":
#                              {"arg1": {"description": "soda can",
#                                 "coordinates": [634, 672, 815, 780] } # Hypothetical coordinates for the soda can
#                          }
#                     }
#                  ]
#              }, indent=4)
#          }]
#      },
#
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "Can you put the spam can in the container?"}
#         ]
#     },
#
#     {
#         "role": "assistant",
#         "content": [
#             {
#                 "text": json.dumps({
#                     "robot_response": "Sure thing.",
#                     "robot_actions": [
#                         {
#                             "action": "grasp_and_place",
#                             "parameters": {
#                                 "arg1": {
#                                     "description": "spam can",
#                                     "coordinates": [137.0, 169.0, 157.0, 209.0]
#                                 },
#                                 "arg2": {
#                                     "description": "container",
#                                     "coordinates": [184.0, 152.0, 204.0, 182.0]
#                                 }
#                             }
#                         }
#                     ]
#                 }, indent=4)
#             }
#         ]
#     }
# # {
# #         "role": "user",
# #         "content": [
# #             {"image": local_file_path1},
# #             {"text": "where do you think the empty spam can should go?"}
# #         ]
# #     },
# #     {
# #         "role": "assistant",
# #         "content": [{
# #             "text": json.dumps({
# #                 "robot_response": "The empty spam can should be placed in the recycling bin for disposal.",
# #                 "robot_actions": None
# #             }, indent=4)
# #         }]
# #     },
# # {
# #         "role": "user",
# #         "content": [
# #             {"image": local_file_path1},
# #             {"text": "can you do it?"}
# #         ]
# #     },
# #     {
# #         "role": "assistant",
# #         "content": [{
# #             "text": json.dumps({
# #                 "robot_response": "Of course, I can do it for you.",
# #                 "robot_actions": [
# #                     {
# #                         "action": "grasp_handover_place",
# #                         "parameters": {
# #                             "arg1": [143, 719, 317, 857], # Hypothetical coordinates for the spam can
# #                             "arg2": [579, 64, 964, 298] # Hypothetical coordinates for the recycling bin
# #                         }
# #                     }
# #                 ]
# #             }, indent=4)
# #         }]
# #     }
# ]
base_multimodal_prompt = [
#     {
#         "role": "system",
#         "content": [{
#             "text": '''You are a multimodal large language model serving as the brain for a humanoid robot. Your capabilities include understanding and processing both visual data and natural language. Here is what you need to do:
#
# 1. **Speech-to-Speech Reasoning**: When provided with a human query and a scene image, analyze the image, understand the query's context, and generate an appropriate verbal response that demonstrates your understanding of the image content. The robot action is None in this case.
#
# 2. **Speech-to-Action Reasoning**: When the human command involves a task that you should perform, assess the necessary action, identify the relevant object in the image, and determine the bounding box coordinates for that object. Then, formulate a response plan to execute the task.
#
# Upon processing the information, output your responses in a structured JSON format with the following keys:
#
# - "robot_response" for the verbal response to the human query.
# - "robot_actions" for the description of the physical action you will perform, including the bounding box coordinates of the object you will manipulate.''',
#
#             "extra": '''ROBOT ACTION LIST is defined as follows:
# grasp_and_place(arg1, arg2): The robot grasps the object at position arg1 and places it at position arg2.
# grasp_handover_place(arg1, arg2): The robot grasps the object at position arg1 with the left hand, hands it over to the right hand, and places it at position arg2.
# grasp_and_give(arg1): The robot grasps the object at position arg1 and gives it to the user. Bounding box coordinates (arg1, arg2) should be determined by you based on the image provided. Here are some examples of expected inputs and outputs:'''
#         }]
#     },
    # Example 1: Speech-to-Speech Reasoning
    {
        "role": "user",
        "content": [
            {"image": local_file_path1},
            {"text": "hey CURI, what do you see right now?"},
        ]
    },
    {
        "role": "assistant",
        "content": [{
            "text": json.dumps({
                "robot_response": "Now I see a red plate in the center of the table, and an empty spam can, a banana, and a soda can near the plate. There is also a green container on the upper left corner of the table.",
                "robot_actions": None
            }, indent=4)
        }]
    },
    # Example 2: Speech-to-Action Reasoning
    {
        "role": "user",
        "content": [
            {"image": local_file_path1},
            {"text": "Can you give me something to drink?"}
        ]
    },
    {
        "role": "assistant",
        "content": [{
            "text": json.dumps({
                "robot_response": "Sure, you can have the soda to drink.",
                "robot_actions": [
                    {
                        "action": "grasp_and_give",
                        "parameters": {
                            "arg1": {
                                "description": "soda can",
                                "bbox_coordinates": [634, 672, 815, 780]  # Hypothetical coordinates for the soda can
                                # [x1, y1, x2, y2]
                            }
                        }
                    }
                ]
            }, indent=4)
        }]
    },
    {
        "role": "user",
        "content": [
            {"image": local_file_path1},
            {"text": "Can you put the spam can in the container?"}
        ]
    },
    {
        "role": "assistant",
        "content": [{
            "text": json.dumps({
                "robot_response": "Sure thing.",
                "robot_actions": [
                    {
                        "action": "grasp_and_place",
                            "parameters": {
                                "arg1": {
                                    "description": "spam can",
                                    "bbox_coordinates": [139, 719, 317, 862] # [x1, y1, x2, y2]
                                },
                                "arg2": {
                                    "description": "container",
                                    "bbox_coordinates": [579, 67, 961, 300] # [x1, y1, x2, y2]
                                }
                    }
                }
                ]
            }, indent=4)
        }]
    }
]


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
#             "text": json.dumps({
#                 "robot_response":
#                     "好的，我看到一个白色的桌子上放着一些物品。左上角有一个橙色的洗涤剂，中间有一个绿色的绿茶罐子，一个蓝色的午餐肉罐头和一个红色的盘子，盘子里放着一根香蕉。",
#                 "robot_actions": None
#             }, indent=4)
#         }]
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
#         "content": [{
#             "text": json.dumps({
#                 "robot_response": "我认为午餐肉罐头适合作为吃火锅时的配菜，因为它易于烹饪，可以在火锅中快速煮熟，并可以与其他食材搭配，增加火锅的多样性",
#                 "robot_actions": None
#             }, indent=4)
#         }]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "那香蕉为什么不适合呢？"},
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": json.dumps({
#                 "robot_response":
#                     "因为香蕉是一种水果，其口感和味道与火锅的其他食材相比可能会显得不太协调。此外，香蕉在高温下容易变软和糊化，这可能会影响其口感和营养价值。因此，香蕉通常不是火锅的常见配菜",
#                 "robot_actions": None
#             }, indent=4)
#         }]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"image": local_file_path1},
#             {"text": "我想清洗一下红色盘子，你可以把洗涤剂递给我吗"},
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [{
#             "text": json.dumps({
#                 "robot_response": "好的，没问题",
#                 "robot_actions": None
#             }, indent=4)
#         }]
#     },
# ]

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
    new_prompt.append({"role": "user", "content": [{"image": local_file_path1}, {"text": query}]})
    # response_dict = dashscope.MultiModalConversation.call(model='qwen-vl-max',
    #                                                  messages=new_prompt)
    #call with local file
    response_dict = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                               messages=new_prompt, top_p=0.5, top_k=50)
    # response_dict = MultiModalConversation.call(model='qwen-vl-plus',
    #                                            messages=new_prompt)

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

    inference_results = None
    if not prompt_append:
        # print("without prompt append")
        for i in range(rounds):
            assistant.record_audio()
            transcription = assistant.transcribe_audio()
            instruction = transcription
            response_dict = single_multimodal_call(base_multimodal_prompt, instruction, log=True, return_response=True)

            print("CURI audio response:\n", response_dict)
            # Parse the JSON string.
            try:
                output_data = json.loads(response_dict)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return

                # Go through each action and plot the bounding boxes.
            if output_data['robot_response'] is not None:
                response = output_data['robot_response']
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
    get_curi_response(base_multimodal_prompt, prompt_append=False)

    # api_key = "sk-59XTKMjGzgbgSJjpC9D770A52eBd4d68902223561eE3F242"
    # base_url = "https://www.jcapikey.com/v1"
    # user_input_filename = '/home/zhuoli/PycharmProjects/CuriGPT/assets/chat_audio/user_input.wav'
    # curigpt_output_filename = '/home/zhuoli/PycharmProjects/CuriGPT/assets/chat_audio/curigpt_output.mp3'
    #
    # get_curi_response_with_audio(api_key, base_url, user_input_filename, curigpt_output_filename,base_multimodal_prompt, rounds=10, prompt_append=False)