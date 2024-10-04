import os
import pickle
from loguru import logger
from urllib.parse import unquote

def load_state(output_directory):
    state_file = os.path.join(output_directory, "crawl_state.pkl")
    if os.path.exists(state_file):
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        logger.info(f"Loaded state from {state_file}")
        return state
    logger.error(f"State file not found: {state_file}")
    return None

def print_stats(state):
    if state is None:
        logger.error("No state data available")
        return

    url_counter = state['url_counter']

    print("Top 10 most frequent URLs:")
    for url, count in url_counter.most_common(10):
        decoded_url = unquote(url)
        print(f"{decoded_url}: {count}")

if __name__ == "__main__":
    output_directory = "wikipedia_content"  # Make sure this matches the directory in wikic.py
    state = load_state(output_directory)
    print_stats(state)
