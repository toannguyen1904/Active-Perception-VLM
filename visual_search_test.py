#!/usr/bin/env python3
"""
Active Perception System for Robot Arm Control
Uses GPT-5 to suggest robot arm movements to capture a target object.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import base64
from typing import Optional, Tuple
import cv2
import random
# from PIL import Image
import json

class Scene:
    def __init__(self, image_path: str, window_size: Tuple[int, int]):
        self.image_path = image_path
        self.window_size = window_size
        self.current_view = None  # Placeholder for the current camera view
        self.image = cv2.imread(image_path)
        self.height, self.width, _ = self.image.shape
        self.curr_x = random.randint(0, self.width - window_size[0])
        self.curr_y = random.randint(0, self.height - window_size[1])
        self.camera_stride = 75
        self.vis_save_dir = "visualization"
        os.makedirs(self.vis_save_dir, exist_ok=True)

    def get_unavailable_directions(self):
        unavailable_directions = []
        if self.curr_y <= 0:
            unavailable_directions.append("up")
        if self.curr_y + self.window_size[1] >= self.height:
            unavailable_directions.append("down")
        if self.curr_x <= 0:
            unavailable_directions.append("left")
        if self.curr_x + self.window_size[0] >= self.width:
            unavailable_directions.append("right")
        return unavailable_directions

    def move_robot_arm(self, direction: str, distance: float):
        """
        Move the robot arm in the specified direction and distance.
        """
        distance = int(distance * 20) # Assume 1cm = 10 pixels

        if direction == "up":
            self.curr_y = max(0, self.curr_y - distance)
        elif direction == "down":
            self.curr_y = min(self.height - self.window_size[1], self.curr_y + distance)
        elif direction == "left":
            self.curr_x = max(0, self.curr_x - distance)
        elif direction == "right":
            self.curr_x = min(self.width - self.window_size[0], self.curr_x + distance)

    def get_current_view(self, save_path):
        """
        Get the current camera view based on the robot arm's position.
        """
        self.current_view = self.image[self.curr_y:self.curr_y + self.window_size[1],
                                       self.curr_x:self.curr_x + self.window_size[0]]
        cv2.imwrite(save_path, self.current_view)
        return save_path
    
    def visualized_current_view(self, text, id, target_object:str= None):
        """
        Visualize the current camera view with annotations.
        """
        vis_image = self.image.copy()
        cv2.rectangle(vis_image, (self.curr_x, self.curr_y),
                      (self.curr_x + self.window_size[0], self.curr_y + self.window_size[1]),
                      (0, 255, 0), 2)
        if target_object:
            cv2.putText(vis_image, f"Target: {target_object}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        # response text
        cv2.putText(vis_image, f"Response: {text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.resize(vis_image, (200, 200), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(os.path.join(self.vis_save_dir, f"vis_view_{id}.png"), vis_image)
        # cv2.imshow("Robot Arm View", vis_image)
        # cv2.waitKey(500)  # Display for 500 ms

# This function is just a placeholder for the actual robot arm movement, will be implemented in the future
def move_robot_arm(direction: str, distance: float):
    """
    Move the robot arm in the specified direction and distance.
    """
    pass

class ActivePerceptionController:
    def __init__(self, api_key: str, target_object: str, max_iterations: int = 30, max_corrections: int = 5):
        """
        Initialize the active perception controller.
        
        Args:
            api_key: OpenAI API key
            max_iterations: Safety limit to prevent infinite loops
            max_corrections: Safety limit to prevent infinite corrections
        """
        self.client = OpenAI(api_key=api_key)
        self.target_object = target_object
        self.max_iterations = max_iterations
        self.max_corrections = max_corrections
        self.messages = []
        self.temp_save_dir = "temp_views"
        os.makedirs(self.temp_save_dir, exist_ok=True)

    def __del__(self):
        # Clean up temporary images
        for file in os.listdir(self.temp_save_dir):
            os.remove(os.path.join(self.temp_save_dir, file))
        os.rmdir(self.temp_save_dir)

        
    def validate_movement_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that the movement command follows the required format.
        
        Args:
            command: The movement command to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if command is "DONE!"
        if command.strip().upper() == "DONE!":
            return True, {}, None
            
        # Pattern to match: direction + space + number + "cm"
        pattern = r'^(up|down|left|right)\s+(\d+(?:\.\d+)?)\s*cm$'
        match = re.match(pattern, command.strip().lower())
        
        if not match:
            return False, {}, f"Invalid format. Expected: 'direction Xcm' (e.g., 'up 4cm') or 'DONE!'"
        
        direction, distance = match.groups()
        distance_float = float(distance)

        return_dict = {
            "direction": direction,
            "distance": distance_float
        }
        
        # Check distance constraints (reasonable range)
        if distance_float <= 0:
            return False, {}, "Distance must be positive"
        if distance_float > 50:  # 50cm max movement
            return False, {}, "Distance too large (max 50cm)"
            
        return True, return_dict, None

    def get_initial_prompt(self) -> str:
        """
        Get the initial prompt for the robot arm control task with image.
            
        Returns:
            Initial prompt string
        """
        return f"""You are controlling a robot arm with a wrist-mounted camera to capture a target object: {self.target_object}. 

        Your task is to suggest movements for the robot arm until the camera can fully capture the target object: {self.target_object}.

        Available movements:
        - "up Xcm" - move arm up by X centimeters
        - "down Xcm" - move arm down by X centimeters  
        - "left Xcm" - move arm left by X centimeters
        - "right Xcm" - move arm right by X centimeters

        Rules:
        1. Only suggest ONE movement at a time
        2. Use the exact format: "direction Xcm" (e.g., "up 4cm", "left 2cm")
        3. When the target object is fully captured and visible, respond with "DONE!"
        4. Be strategic - consider the current view and what might be needed to capture the target
        """

    def get_observation_prompt(self, additional_sub_prompt="") -> str:
        """
        Get the prompt for subsequent observations with image.

        Returns:
            Prompt string for the observation
        """

        return f"""This is the current camera view.

        Look at the current camera view and determine where the robot arm should move next. {additional_sub_prompt}
        Remember to respond with either:
        - A movement command: "direction Xcm" (e.g., "up 4cm", "left 2cm") 
        - "DONE!" if the target object: {self.target_object} is fully captured"""

    def get_gpt_response(self, prompt: str, image_path: str = None) -> str:
        """
        Get response from GPT-5.
        
        Args:
            prompt: The prompt to send to GPT
            image_path: path to the image observation, now using pre-collected image
            
        Returns:
            GPT response string
        """

        # Encode image for vision API
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
        
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=self.messages,
            reasoning_effort="low",   # can be "minimal", "low", "medium", "high". Use "low" now as our example is simple.
            stream=False
        )
        response_content = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": response_content})
        return response_content

    def run_active_perception(self, scene:Scene):
        """
        Run the active perception loop.
        """
        print("🤖 Starting Active Perception System")
        print("=" * 60)

        # append the system prompt to self.messages
        self.messages.append({"role": "system", "content": self.get_initial_prompt()})
        
        # Main loop
        for iteration in range(0, self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Assume a robot movement has been made and get new image
            # image_path = input("📸 Enter the path to the camera image: ")
            temp_image_path = os.path.join(self.temp_save_dir ,f"view_{iteration}.png")
            scene.get_current_view(save_path=temp_image_path)
            
            # Get next movement suggestion with new image
            unavailable_directions = scene.get_unavailable_directions()
            if unavailable_directions:
                additional_sub_prompt = f"The following directions are unavailable due to camera limits: {', '.join(unavailable_directions)}. Please avoid suggesting these directions."
                prompt = self.get_observation_prompt(additional_sub_prompt)
            else:
                prompt = self.get_observation_prompt()

            response = self.get_gpt_response(prompt, temp_image_path)
            print(f"🤖 GPT Response: {response}")
            
            # Validate response
            is_valid, return_dict, error = self.validate_movement_command(response)
            correction_attempts = 0
            while not is_valid and correction_attempts < self.max_corrections:
                print(f"❌ Invalid response: {error}")
                print("🔄 Asking for correction...")
                correction_prompt = f"Invalid response: {error}\nPlease provide a valid movement command or 'DONE!'"
                response = self.get_gpt_response(correction_prompt, temp_image_path)
                print(f"🤖 Corrected Response: {response}")
                is_valid, return_dict, error = self.validate_movement_command(response)
                correction_attempts += 1
            
            # If still invalid, return
            if not is_valid:
                print(f"❌ Still invalid after {self.max_corrections} attempts: {error}")
                return
            
            # Check if done
            if response.upper() == "DONE!":
                print(f"🎉 DONE! Target object: {self.target_object} is fully captured!")
                scene.visualized_current_view(text="DONE!", id=iteration, target_object=self.target_object)
                return
            # if not done, then move the robot arm according to the return_dict
            else:
                scene.move_robot_arm(return_dict["direction"], return_dict["distance"])
                scene.visualized_current_view(text=response, id=iteration, target_object=self.target_object)

        print(f"⚠️  Maximum iterations ({self.max_iterations}) reached. Stopping.")

def main():
    """
    Main function to run the active perception system.
    """
    print("🚀 Active Perception System for Robot Arm Control")
    print("Using GPT-5 for movement suggestions")

    load_dotenv()

    # Set up the parameters
    api_key = os.getenv("API_KEY") # "Create a .env file and add your OpenAI API key as API_KEY"
    target_object = "eyeglasses"
    max_iterations = 30
    max_corrections = 5

    # image and window parameters
    base_image = "images/desk1.png"
    window_size = (300, 300)

    scene = Scene(base_image, window_size)

    # Initialize controller
    controller = ActivePerceptionController(api_key, target_object, max_iterations, max_corrections)
    
    # Run the active perception loop
    controller.run_active_perception(scene)

# Run the main function
if __name__ == "__main__":
    main()
