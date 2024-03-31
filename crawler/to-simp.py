from hanziconv import HanziConv
from collections import defaultdict
import os

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

def main():
    DIR = './wiki-pages/zh/'
    SIMP_DIR = './wiki-pages/zh-simp/'
    for fn in os.listdir(DIR):
        filename = DIR + fn
        print(filename)
        # cnt = defaultdict(int)
        with open(filename) as f:
            text = f.read() 
            text_simp = ''
            for c in text:
                s = convert_character(c, 'Simplified')
                text_simp += s
        fn_simp = ''.join([convert_character(c, 'Simplified') for c in fn])
        with open(SIMP_DIR + fn_simp, 'w') as f:
            f.write(text_simp)
            #     what = check_character_script(c)
            #     cnt[what] += 1
            # for k, v in cnt.items():
            #     print(k, v)

if __name__ == '__main__':
    main()