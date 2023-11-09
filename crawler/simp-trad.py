from collections import defaultdict
import os
from hanziconv import HanziConv

# Function to determine if a character is Simplified or Traditional
def check_character_script(character):
    # Convert the character to both scripts
    to_simplified = HanziConv.toSimplified(character)
    to_traditional = HanziConv.toTraditional(character)
    
    # Check if the character is the same after conversion
    if character == to_simplified and character != to_traditional:
        return "Simplified"
    elif character == to_traditional and character != to_simplified:
        print(character)
        return "Traditional"
    elif character == to_simplified and character == to_traditional:
        # print(character)
        # return "Both (Character is the same in Simplified and Traditional)"
        return "Simplified"
    else:
        # return "Neither (Character may not be Chinese or is rare/uncommon)"
        return "Simplified"

# Function to convert a character to the specified script
def convert_character(character, target_script):
    if target_script == "Simplified":
        return HanziConv.toSimplified(character)
    elif target_script == "Traditional":
        return HanziConv.toTraditional(character)
    else:
        raise ValueError("The target script must be 'Simplified' or 'Traditional'.")

# # Example usage
# character = '開'  # A Traditional Chinese character
# script_type = check_character_script(character)
# converted_to_simplified = convert_character(character, "Simplified")

# print(f"The character '{character}' is in {script_type} script.")
# print(f"The Simplified form of '{character}' is '{converted_to_simplified}'.")



def main():
    DIR = './wiki-pages/zh/'
    for fn in os.listdir(DIR):
        filename = DIR + fn
        print(filename)
        cnt = defaultdict(int)
        with open(filename) as f:
            text = f.read() 
            for c in text:
                what = check_character_script(c)
                cnt[what] += 1
            for k, v in cnt.items():
                print(k, v)

if __name__ == '__main__':
    main()