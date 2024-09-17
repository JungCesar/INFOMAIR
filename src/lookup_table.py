table = {0: 'ack', 1: 'affirm', 2: 'bye', 3: 'confirm', 4: 'deny', 5: 'hello', 6: 'inform', 7: 'negate', 8: 'null', 9: 'repeat', 10: 'reqalts', 11: 'reqmore', 12: 'request', 13: 'restart', 14: 'thankyou'}

# Returns the class name for a given class number
def lookup_table(id):
    """Lookup table for mapping class numbers to class names"""
    return table[id]