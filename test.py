from imagine import Imagine
from imagine.styles import GenerationsStyle
from imagine.models import Status

# Initialize the Imagine client with your API token
client = Imagine(token="vk-x2PRU1EZkaoiwEPFBZIQWPbv0aWVdMqcH3LykWeL2pDrwkc")
text = 'black flower'
# Generate an image using the generations feature
# try:
client.generations(
    prompt=text,
    style=GenerationsStyle.IMAGINE_V5
)
    # print(response)
# except:
#     None