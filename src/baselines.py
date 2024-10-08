table = {0: 'ack', 1: 'affirm', 2: 'bye', 3: 'confirm', 4: 'deny', 5: 'hello', 6: 'inform', 7: 'negate', 8: 'null', 9: 'repeat', 10: 'reqalts', 11: 'reqmore', 12: 'request', 13: 'restart', 14: 'thankyou'}

# Returns the class name for a given class number
def lookup_table(id):
    """Lookup table for mapping class numbers to class names"""
    return table[id]

def baseline_model_majority(_):
    """Baseline model always returns the majority class"""
    # return data.value_counts().idxmax()
    # return "inform"
    return lookup_table(6)

import re

def baseline_model_keywords(input):
    """
    Baseline model based on hand-crafted rules: keyword matching
    """

    # Keywords for all 15 labels, handpicked from the dataset
    keywords = {
        "ack": [r"\b(thatll do dear|im good|just pick|well take|okay um|great day)\b"],
        "affirm": [r"\b(ye|yes|yeah)\b", r"^(right)$"],
        "bye": [
            r"\b(okay thank you and good bye|alright thank you good bye|see you)\b",
            r"^(goodbye|good bye|bye)$",
        ],
        "confirm": [r"\b(does it|do they)\b"],
        "deny": [r"\b(dont want|no not|wrong|something else|not important)\b"],
        "hello": [r"\b(hello|hi|hey)\b"],
        "inform": [
            r"\b(spanish|italian|east|west|it doesnt matter|korean|moroccan|center|north|doesnt matter|expensive|cheap|south|any|dont care|vietnamese|irish|spaninsh|oriental|portugese|indian|lebanese|thai|chinese|african|asian|part|town|priced|moderate|gastropub)\b"
        ],
        "negate": [r"\b(no |not|never)\b"],
        "null": [r"\b(sil|noise|cough|unintelligible)\b|\bunintelligible\n"],
        "repeat": [
            r"\b(try this again|again please)\b",
            r"^(please repeat|back|again|repeat|repeat that|go back|can you repeat that)$",
        ],
        "reqalts": [r"\b(how about|what about|next|anything else|is there)\b"],
        "reqmore": [r"^(more)$"],
        "request": [
            r"\b(type of food|phone number|address|need|post code|price range)\b"
        ],
        "restart": [r"\b(start|oh jesus|reset)\b"],
        "thankyou": [r"\b(thank you|thanks|appreciate|thank you good bye)\b"],
    }

    # Function to match keywords
    def match_keywords(sentence):
        for label, patterns in keywords.items():
            for pattern in patterns:
                if re.search(pattern, sentence.lower()):
                    return label

        # If no keyword is matched, return 'inform'
        return "inform"

    return match_keywords(input)
