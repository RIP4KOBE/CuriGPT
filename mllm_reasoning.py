"""
CuriGPT
===========================

High-level task reasoning for CURI manipulation via Multimodal LLM (Qwen-VL-Max).
"""

from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation

local_file_path1 = 'file:///home/zhuoli/Pictures/curigpt_test_tabletop.jpeg'

curigpt_prompt = [
    {
        'role': 'system',
        'content': [{
            'text': 'You are a humanoid robot named CURI, which has two anthropomorphic five-fingered hand. Your '
                    'role is to '
                    'understand '
                    'the given questions and images and then output the correct answers and robot action sequences '
        }]
    },
    {
            "role": "user",
            "content": [
                {'image': local_file_path1},
                {"text": "hey CURI, what do you see right now?"},
            ]
    },
{"role": "assistant", "content":
        '''
        now I see a red plate in the center of the table, and an empty spam can, a banana, and a soda can near the 
        plate. There is also a green container on the upper left corner of the table.
        '''},
    {
        "role": "user",
        "content": [
            {'image': local_file_path1},
            {"text": "What do you think the empty spam can should go?"},
        ]
    },
{"role": "assistant", "content":
        '''
        The empty spam can should go in the green container.
        '''},
    {
        "role": "user",
        "content": [
            {'image': local_file_path1},
            {"text": "Great, can you put them in?"},
        ]
    },
{"role": "assistant", "content":
        '''
        Answer: 
        Of course, I will put the empty spam can in the green container.
        Action:
        pos_origin, pos_target = locate_object(empty_spam_can, green_container)
        grasp_and_place(pos_origin, pos_target)
        '''},

]


def simple_multimodal_conversation_call():

    """Simple single round multimodal conversation call.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
                {"text": "这是什么?"}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',
                                                     messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.

    #streaming output
    # for response in responses:
    #     print(response)

def conversation_call():
    """Sample of multiple rounds of conversation.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
                {"text": "这是什么?"},
            ]
        }
    ]
    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                           messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.
    messages.append({'role': response.output.choices[0].message.role,
                     'content': [{'text': response.output.choices[0].message.content}]})
    messages.append({"role": "user",
                     "content": [
                         {"text": "她们在干什么?", }
                     ]})

    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                           messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)

def call_with_local_file():
    """Sample of use local file.
       linux&mac file schema: file:///home/images/test.png
       windows file schema: file://D:/images/abc.png
    """
    local_file_path1 = 'file:///home/zhuoli/Pictures/tabletop_fruit.png'
    messages = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }, {
        'role':
        'user',
        'content': [
            {
                'image': local_file_path1
            },
            # {
            #     'image': local_file_path2
            # },
            {
                'text': 'what do you see right now?'
            },
        ]
    }]
    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1, messages=messages)
    print(response)

def multiple_call_with_local_file():
    """multiple rounds of conversation with local file.
    """

    local_file_path1 = 'file:///home/zhuoli/Pictures/tabletop_fruit.png'
    messages = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    },
        {
            "role": "user",
            "content": [
                {'image': local_file_path1},
                {"text": "what do you see right now?"},
            ]
        }
    ]
    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                           messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.
    messages.append({'role': response.output.choices[0].message.role,
                     'content': [{'text': response.output.choices[0].message.content}]})
    messages.append({"role": "user",
                     "content": [
                         {'image': local_file_path1},
                         {"text": "Can you give me the bounding box coordinate of the largest orange?", }
                     ]})

    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                           messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)


if __name__ == '__main__':
    # simple_multimodal_conversation_call()
    # conversation_call()
    # call_with_local_file()
    multiple_call_with_local_file()