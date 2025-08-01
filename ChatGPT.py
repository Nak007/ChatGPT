'''
Available methods are the followings:
[1] ChatModels
[2] ImageGenerators

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-04-2025
'''
import requests
from warnings import warn
import numpy as np
import openai
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from collections import namedtuple
import urllib3, warnings, base64, io
urllib3.disable_warnings()

__all__ = ["ChatModels", 
           "ImageGenerators"]

class ValidateParams:
    
    '''Validate parameters'''
    
    def Interval(self, Param, Value, dtype=int, 
                 left=None, right=None, closed="both"):

        '''
        Validate numerical input.

        Parameters
        ----------
        Param : str
            Parameter's name

        Value : float or int
            Parameter's value

        dtype : {int, float}, default=int
            The type of input.

        left : float or int or None, default=None
            The left bound of the interval. None means left bound is -∞.

        right : float, int or None, default=None
            The right bound of the interval. None means right bound is +∞.

        closed : {"left", "right", "both", "neither"}
            Whether the interval is open or closed. Possible choices are:
            - "left": the interval is closed on the left and open on the 
              right. It is equivalent to the interval [ left, right ).
            - "right": the interval is closed on the right and open on the 
              left. It is equivalent to the interval ( left, right ].
            - "both": the interval is closed.
              It is equivalent to the interval [ left, right ].
            - "neither": the interval is open.
              It is equivalent to the interval ( left, right ).

        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        Options = {"left"    : (np.greater_equal, np.less), # a<=x<b
                   "right"   : (np.greater, np.less_equal), # a<x<=b
                   "both"    : (np.greater_equal, np.less_equal), # a<=x<=b
                   "neither" : (np.greater, np.less)} # a<x<b

        f0, f1 = Options[closed]
        c0 = "[" if f0.__name__.find("eq")>-1 else "(" 
        c1 = "]" if f1.__name__.find("eq")>-1 else ")"
        v0 = "-∞" if left is None else str(dtype(left))
        v1 = "+∞" if right is None else str(dtype(right))
        if left  is None: left  = -np.inf
        if right is None: right = +np.inf
        interval = ", ".join([c0+v0, v1+c1])
        tuples = (Param, dtype.__name__, interval, Value)
        err_msg = "%s must be %s or in %s, got %s " % tuples    

        if isinstance(Value, dtype):
            if not (f0(Value, left) & f1(Value, right)):
                raise ValueError(err_msg)
        else: raise ValueError(err_msg)
        return Value

    def StrOptions(self, Param, Value, options, dtype=str):

        '''
        Validate string or boolean inputs.

        Parameters
        ----------
        Param : str
            Parameter's name
            
        Value : float or int
            Parameter's value

        options : set of str
            The set of valid strings.

        dtype : {str, bool}, default=str
            The type of input.
        
        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        if Value not in options:
            err_msg = f'{Param} ({dtype.__name__}) must be either '
            for n,s in enumerate(options):
                if n<len(options)-1: err_msg += f'"{s}", '
                else: err_msg += f' or "{s}" , got %s'
            raise ValueError(err_msg % Value)
        return Value
    
    def check_range(self, param0, param1):
        
        '''
        Validate number range.
        
        Parameters
        ----------
        param0 : tuple(str, float)
            A lower bound parameter e.g. ("name", -100.)
            
        param1 : tuple(str, float)
            An upper bound parameter e.g. ("name", 100.)
        '''
        if param0[1] >= param1[1]:
            raise ValueError(f"`{param0[0]}` ({param0[1]}) must be less"
                             f" than `{param1[0]}` ({param1[1]}).")

class ChatModels(ValidateParams):
    
    '''
    Connect to OpenAI chat models via API 
    
    Parameters
    ----------
    api_key : str
        The API key used for authenticating requests to the ChatGPT API.
    
    model : str, default="gpt-4o-mini"
        Specifies the model of ChatGPT to be used for responses, which 
        can be one of the following: "gpt-4o-mini", "gpt-3.5-turbo", 
        "gpt-4o", "gpt-4-turbo".
    
    temperature : float, default=1.0
        Controls the randomness of the output. The value can range from 0 
        (deterministic responses) to 2 (more random and varied responses).
    
    timeout : int, default=120
        Specifies the maximum number of seconds to wait for the client to 
        establish a connection and receive a response.
    
    Attributes
    ----------
    chathistory : list
        A list of message objects representing the conversation history. 
        Each message is represented as a dictionary with the following 
        structure: {"role": value, "content": value}:
        - role (str): The role of the message sender, which can be 
          "system", "user", or "assistant".
        - content (str): The text content of the message.
    
    '''
    
    def __init__(self, api_key, model="gpt-4o-mini", 
                 temperature=1.0, timeout=120):

        # request's parameters
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {"Authorization": f'Bearer {self.api_key}',
                        "Content-Type" : 'application/json'}
        args = (int, 1, None, "left")
        self.timeout = self.Interval("timeout", timeout, *args)
        
        # json's parameters (form-encoded data)
        args = (["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"], str)
        self.model = self.StrOptions('model', model, *args)
        args = (float, 0., 2., "both")
        self.temperature = self.Interval("temperature", temperature, *args)

        # Other attributes
        self.chathistory = list()
        
    def chat(self, prompt):
        
        '''
        Sending massege to ChatGPT
        
        Parameters
        ----------
        prompt : str
            The input message to send to the ChatGPT for processing.
        
        Returns
        -------
        content : str
            The text content generated in response to the input prompt. 
            If an error occurs during processing, it returns None.
        
        '''
        # Append user's message to conversation history
        self.chathistory += [{"role": "user", "content": prompt}]
        payload = {"model": self.model,
                   "messages": self.chathistory,
                   "temperature": self.temperature, 
                   "n": 1}
        
        try:
            # Send a request
            response = requests.post(url=self.url,
                                     headers=self.headers,
                                     json=payload,
                                     timeout=self.timeout, 
                                     verify=False)

            # Check status code
            if response.status_code == 200:
                # Append assistant's reply to conversation history
                content = response.json()['choices'][0]['message']['content']
                self.chathistory += [{"role": "assistant", "content": content}]
                return content  
            else: warn(f"API Error: {response.status_code}, {response.text}") 

        # It serves as a general exception for any issues that occur while making 
        # HTTP requests. This can include a range of problems, such as: 
        # Connection errors, Timeout errors, Invalid URLs or SSL errors.
        except requests.exceptions.RequestException as e:
            warn(f"Request Error: {e}")

        # Handle other errors
        except Exception as e:
            warn(f"General Error: {e}")
            
        return None
            
    def interactivechat(self):
        
        '''
        Interactive chat with ChatGPT. Type "exit" or "quit" to stop the 
        conversation.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        user_input : function
            Returns an `input` function that allows the user to interactively 
            converse with ChatGPT. The conversation continues until the user 
            types "exit" or "quit".
        
        '''
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]: 
                print("Assistant: This session has been terminated.")
                break
            else: print("Assistant:", self.chat(user_input))

class ImageGenerators(ValidateParams):
    
    '''
    Connect to OpenAI to create image(s) given a prompt
    
    Parameters
    ----------
    api_key : str
        The API key used for authenticating requests to API.
    
    model : str, default="dall-e-2"
        Specifies the model of the image generator to be used for 
        responses. Options include: "dall-e-2" or "dall-e-3".

    size : str, default=None
        The size of the generated images. Must be one of "256x256", 
        "512x512", or "1024x1024" for `dall-e-2`. Must be one of 
        "1024x1024", "1792x1024", or "1024x1792" for `dall-e-3`.
        
    n : int, default=1
        The number of images to generate; must be between 1 and 10.
        Note: For `dall-e-3`, only a value of n=1 is supported.

    timeout : int, default=120
        Specifies the maximum number of seconds to wait for the client 
        to establish a connection and receive a response.

    Attributes
    ----------
    chathistory : list
        A list of message objects representing the conversation history.
        Each message is a dictionary with the following structure:
        {"prompt": value, "image": value}:
        - prompt (str): A textual description of the desired image(s).
        - image (list): A list of outputs in Base64 format.
    
    '''
    
    def __init__(self, api_key, model="dall-e-2", 
                 size=None, n=1, timeout=120):

        # Initialize parameters
        opt = namedtuple('Options', ['size', 'args', 'len'])
        self.params = {"dall-e-2" : opt(["256x256", "512x512", "1024x1024"], 
                                        (int, 1, 10, "both"), 1000),
                       "dall-e-3" : opt(["1024x1024", "1792x1024", "1024x1792"], 
                                        (int, 1, 2, "left"), 4000)}
    
        # request's parameters
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/images/generations"
        self.headers = {"Authorization": f'Bearer {self.api_key}',
                        "Content-Type" : 'application/json'}
        args = (int, 1, None, "left")
        self.timeout = self.Interval("timeout", timeout, *args)
        
        # Select model
        args = (self.params.keys(), str)
        self.model = self.StrOptions('model', model, *args)

        # Select image size
        if size is not None: 
            args = (self.params[self.model].size, str)
            self.size = self.StrOptions('size', size, *args)
        else: self.size = self.params[self.model].size[0]

        # Select number of images
        self.n = self.Interval("n", n, * self.params[self.model].args)

        # Other attributes
        self.chathistory = list()

    def generate(self, prompt, display=False):
  
        '''
        Generate image(s) based on the provided text description.
        
        Parameters
        ----------
        prompt : str
            A textual description of the desired image(s). The maximum 
            length is limited to 1000 characters for `dall-e-2` and 4000 
            characters for `dall-e-3`.
 
        display : bool, default=False
            If True, it returns a list of Matplotlib axes objects 
            containing the displayed image(s); otherwise, it returns a 
            list of Base64-encoded images.

        Returns
        -------
        base64_images : list
            A list of Base64-encoded images generated in response to the 
            input prompt. This is relevant when `display` is False. If 
            an error occurs during processing, an empty list will be 
            returned.

        ax : list of matplotlib.axes.Axes
            A list of Matplotlib axis objects containing the displayed 
            image(s). This is relevant when `display` is True.
            
        '''
        # Initialize parameters
        payload = {"prompt": (prompt:=prompt[:self.params[self.model].len]),
                   "size": self.size, "n": self.n,
                   "response_format" : "b64_json"}
        
        try:
            # Send a request
            response = requests.post(url=self.url,
                                     headers=self.headers,
                                     json=payload,
                                     timeout=self.timeout, 
                                     verify=False)

            if response.status_code == 200:
                base64_images = [data['b64_json'] for data in response.json()['data']] 
                self.chathistory += [{"prompt": prompt, "image": base64_images}]

                
                if display:
                    for image in base64_images:
                        ax = self.display(image)
                        plt.tight_layout()
                        plt.show()
                else:return base64_images
                    
            else: warn(f"API Error: {response.status_code}, {response.text}") 
                
        # It serves as a general exception for any issues that occur while making 
        # HTTP requests. This can include a range of problems, such as: 
        # Connection errors, Timeout errors, Invalid URLs or SSL errors.
        except requests.exceptions.RequestException as e:
            warn(f"Request Error: {e}")

        # Handle other errors
        except Exception as e:
            warn(f"General Error: {e}")
            
        return None

    def __ConvertBase64__(self, base64_image):

        '''
        Convert a Base64-encoded image to a NumPy array.

        Parameters
        ----------
        base64_image : str
            A Base64-encoded image input.

        Returns
        -------
        image_array : numpy.ndarray
            A NumPy array representation of the decoded image. If an error 
            occurs during processing, it will return None. 
        
        '''
        try:
            # Decode the Base64 string
            encoded_bytes = base64.b64decode(base64_image)
    
            # Convert binary data to a NumPy array
            image_array = np.array(Image.open(io.BytesIO(encoded_bytes))) 
            return image_array

        except: return None

    def display(self, base64_image, ax=None):

        '''
        Display a Base64-enconded image.

        Parameters
        ----------
        base64_image : str
            A Base64-encoded image input.

        ax : matplotlib.axes.Axes, default=None
            An optional predefined Matplotlib axis. If None, a new axis 
            will be created based on the default settings.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib axis object containing the displayed image. 
            If an error occurs during processing, it will return None. 

        '''
        # Decode the Base64 string
        image = self.__ConvertBase64__(base64_image)

        # Plot image
        if image is None: return None
        if ax is None: ax = plt.subplots(figsize=(5,5))[1]
        ax.imshow(image)
        ax.axes.yaxis.set_visible(False)
        ax.axes.xaxis.set_visible(False)
        
        return ax