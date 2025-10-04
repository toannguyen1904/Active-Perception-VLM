#!/usr/bin/env python3
"""
Active Perception System for Robot Arm Control
Uses GPT-5 to suggest robot arm movements to capture a target object.
"""

from openai import OpenAI
import re
import base64
from typing import Optional, Tuple

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

    def get_observation_prompt(self) -> str:
        """
        Get the prompt for subsequent observations with image.

        Returns:
            Prompt string for the observation
        """

        return f"""This is the current camera view.

        Look at the current camera view and determine where the robot arm should move next. Remember to respond with either:
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

    def run_active_perception(self):
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
            image_path = input("📸 Enter the path to the camera image: ")
            
            # Get next movement suggestion with new image
            prompt = self.get_observation_prompt()
            response = self.get_gpt_response(prompt, image_path)
            print(f"🤖 GPT Response: {response}")
            
            # Validate response
            is_valid, return_dict, error = self.validate_movement_command(response)
            correction_attempts = 0
            while not is_valid and correction_attempts < self.max_corrections:
                print(f"❌ Invalid response: {error}")
                print("🔄 Asking for correction...")
                correction_prompt = f"Invalid response: {error}\nPlease provide a valid movement command or 'DONE!'"
                response = self.get_gpt_response(correction_prompt, image_path)
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
                return
            # if not done, then move the robot arm according to the return_dict
            else:
                move_robot_arm(return_dict["direction"], return_dict["distance"])
        
        print(f"⚠️  Maximum iterations ({self.max_iterations}) reached. Stopping.")

def main():
    """
    Main function to run the active perception system.
    """
    print("🚀 Active Perception System for Robot Arm Control")
    print("Using GPT-5 for movement suggestions")

    # Set up the parameters
    api_key = "Your API Key goes here, check Slack for the key"
    target_object = "a blue cup"
    max_iterations = 30
    max_corrections = 5

    # Initialize controller
    controller = ActivePerceptionController(api_key, target_object, max_iterations, max_corrections)
    
    # Run the active perception loop
    controller.run_active_perception()

# Run the main function
if __name__ == "__main__":
    main()
