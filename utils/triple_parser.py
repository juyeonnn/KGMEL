import json

from tqdm import tqdm
import re
from typing import List, Optional, Dict


def parse(data, key="response"):
    for i,item in enumerate(data):
        
        if key not in item:
            continue

        sections = split_section(item[key])
        type = preprocess_type(sections["type"],item["mention"])
        desc = preprocess_desc(sections["desc"],item["mention"])
        triple = preprocess_triple(sections["triple"],item["mention"])
        triple = add_type_triple(type, triple, item["mention"])

        data[i]["type"] = type
        data[i]["desc"] = desc
        data[i]["triple"] = triple
    return data



def get_matching_entity(entity: str, mentions: List[str]) -> Optional[str]:
    """Find matching entity in mentions list using normalized comparison."""
    if not entity or not mentions:
        return None
        
    norm_entity = normalize(entity)
    entity_words = set(norm_entity.split())
    
    for mention in mentions:
        norm_mention = normalize(mention)
        
        # Check exact match, substring, or significant word overlap
        if (norm_entity == norm_mention or 
            norm_entity.find(norm_mention) != -1 or
            norm_mention.find(norm_entity) != -1 or
            len(entity_words & set(norm_mention.split())) > 0):
            return mention
            
    return None


def split_section(data):
    data = data.split("\n")
    idx = {"type" : -1, "desc": -1, "triple": -1}
    for i, line in enumerate(data):
        if "###" in line :
            if '1' in line and 'Step'  in line:
                idx["type"] = i
            elif '2' in line and 'Step'  in line:
                idx["desc"] = i
            elif '3' in line and 'Step'  in line:
                idx["triple"] = i
    return {'type': data[idx["type"]+1:idx["desc"]], 'desc': data[idx["desc"]+1:idx["triple"]], 'triple': data[idx["triple"]+1:]}


def clean_triple(data):
    """
    Clean and normalize knowledge graph triples from various formats.
    
    Args:
        data: A string representing a potential knowledge graph triple
        
    Returns:
        list: A list containing [subject, predicate, object] if valid, None otherwise
    """
    # Step 1: Handle table format (remove surrounding pipes)
    if data.startswith("|") and data.endswith("|"):
        data = data[1:-1].strip()
    
    # Step 2: Normalize separators for consistency
    # Convert mixed separators to a single consistent separator
    if '; ' in data and ': ' in data:
        data = data.replace('; ', ': ')
    if ': ' in data and ', ' in data:
        data = data.replace(', ', ': ')
    if '| ' in data and ': ' in data:
        data = data.replace(': ', '| ')
    
    # Step 3: Determine the separator being used
    if '|' in data:
        sep = '|'
    elif ':' in data:
        sep = ':'
    elif ',' in data:
        sep = ','
    else:
        # No recognized separator found
        return None
    
    # Step 4: Split the data using the identified separator and clean each part
    data = data.split(sep)
    data = [clean_text(x) for x in data]
    
    # Step 5: Validate that no part is None after cleaning
    if None in data:
        return None
    
    # Step 6: Filter out triples with placeholder object values
    invalid_object_words = ['unknown', 'none', 'no triple', 'unspecified']
    if any(word in data[-1].lower() for word in invalid_object_words):
        return None
    
    # Step 7: Handle extremely long object values (potential errors)
    if len(data[-1]) > 200:
        data[-1] = clean_repetition(data[-1])
    
    # Step 8: Validate and format the final triple
    if len(data) == 3:
        # Perfect triple format
        return data
    elif len(data) > 3:
        # Too many elements - combine excess elements into the object position
        return [data[0], data[1], " ".join(data[2:])]
    else:
        # Too few elements - invalid triple
        return None
def clean_repetition(data):
    """
    Clean text with excessive repetitions by keeping only the first and last elements.
    
    Args:
        data: String that might contain repetitive elements
        
    Returns:
        str: Cleaned string with repetitions removed
    """
    # Handle comma-separated repetitions
    if data.count(', ') > 10:
        elements = data.split(', ')
        return elements[0] + ', ' + elements[-1]
    
    # Handle semicolon-separated repetitions
    elif data.count('; ') > 10:
        elements = data.split(';')
        return elements[0] + ';' + elements[-1]
    
    # Return original if no excessive repetitions found
    return data


def case_wo_sep(data, mention):
    """
    Handle cases where standard separators are missing by trying to identify entity mentions.
    
    Args:
        data: String to analyze
        mention: List of entity mentions to search for
        
    Returns:
        tuple: (entity, description) if entity found in data, otherwise (None, None)
    """
    # Case 1: Full mention is in the data
    for m in mention:
        if m in data:
            # Extract entity and description by removing the mention from the data
            return m, data.replace(m, '').strip()
    
    # Case 2: Part of a multi-word mention is in the data
    for m in mention:
        word_level_m = m.split()
        # Check if any word from the mention appears in the data
        if any(word in data for word in word_level_m):
            return m, data.replace(m, '').strip()
    
    # No match found
    return None, None


def split_line(data, mention):
    """
    Split a line of text into two parts based on identified separators.
    
    Args:
        data: String to split
        mention: Entity mention to handle cases without standard separators
        
    Returns:
        tuple: (first_part, second_part) if successfully split, otherwise (None, None)
    """
    # Step 1: Identify the separator being used
    if '|' in data:
        sep = '|'
    elif ':' in data:
        sep = ':'
    elif ',' in data:
        sep = ','
    else:
        # No standard separator found, try special case handling
        data0, data1 = case_wo_sep(data, mention)
        if not data0:
            return None, None
        return data0, clean_text(data1)

    # Step 2: Split the data using the identified separator
    data = data.split(sep)

    # Step 3: Handle special cases where split resulted in more than 2 parts
    if len(data) != 2:
        # If there are more than 2 parts, keep the first part as is
        # and join the rest using the same separator
        return clean_text(data[0]), clean_text(sep.join(data[1:]))
    
    # Step 4: Return the cleaned parts
    return clean_text(data[0]), clean_text(data[1])



def get_type(data, type_lst=["person", "nationality", "religious group", "political group", 
                            "organization", "country", "city", "state", "building", "airport", 
                            "highway", "bridge", "company", "agency", "institution", "product", 
                            "event", "work of art", "law", "language"]):
    """
    Identify entity type from text by looking for predefined type keywords.
    
    Args:
        data: String containing potential entity type information
        type_lst: List of recognized entity types to search for
        
    Returns:
        str: Detected entity type or None if no match found
    """
    # Check if any of the predefined types appear in the data
    for entity_type in type_lst:
        if entity_type in data.lower():
            return entity_type
    
    return None


def preprocess_type(data, mention):
    """
    Process and map entity types to mentions.
    
    Args:
        data: List of strings containing entity type information
        mention: List of entity mentions to map types to
        
    Returns:
        dict: Mapping of entity mentions to their identified types
    """
    # Remove duplicates and empty strings
    data = list(set([x for x in data if x != '']))
    
    # Initialize output dictionary with all mentions set to None
    output = {k: None for k in mention}
    
    # Process each line to extract entity and type pairs
    for line in data:
        entity, entity_type = split_line(line, mention)
        
        if entity in mention:
            # Direct match found
            output[entity] = entity_type
        elif entity is not None:
            # Try to find a partial match
            matched_mention = get_matching_entity(entity, mention)
            if matched_mention and not output.get(matched_mention):
                # If we found a matching mention and it doesn't have a type yet
                output[matched_mention] = entity_type
            elif len(mention) == 1:
                # If there's only one mention, assign type to it regardless
                output[mention[0]] = entity_type
    
    # Fill in missing types by searching the entire text
    for m in mention:
        if output[m] is None:
            output[m] = get_type(' '.join(data))
            
    return output


def preprocess_desc(data, mention):
    """
    Process and map entity descriptions to mentions.
    
    Args:
        data: List of strings containing entity description information
        mention: List of entity mentions to map descriptions to
        
    Returns:
        dict: Mapping of entity mentions to their descriptions
    """
    # Remove duplicates and empty strings
    data = list(set([x for x in data if x != '']))
    
    # Initialize output dictionary with all mentions set to None
    output = {k: None for k in mention}
    
    # Process each line to extract entity and description pairs
    for line in data:
        entity, desc = split_line(line, mention)
        
        if entity in mention:
            # Direct match found
            output[entity] = desc
        elif entity is not None:
            # Try to find a partial match
            matched_m = get_matching_entity(entity, mention)
            if matched_m and not output.get(matched_m, None):
                # If we found a matching mention and it doesn't have a description yet
                output[matched_m] = desc
            elif len(mention) == 1:
                # If there's only one mention, assign description to it regardless
                output[mention[0]] = desc
                
    return output


def preprocess_triple(data, mention):
    """
    Process and map knowledge graph triples to entity mentions.
    
    Args:
        data: List of strings containing triple information
        mention: List of entity mentions to map triples to
        
    Returns:
        dict: Mapping of entity mentions to their associated triples
    """
    # Remove duplicates and empty strings
    data = list(set([x for x in data if x != '']))
    
    # Initialize output dictionary with all mentions set to empty lists
    output = {k: [] for k in mention}
    
    # Process each line to extract and assign triples
    for line in data:
        mapped = False
        
        # First try direct mention matching
        for m in mention:
            if m in line:
                triple = clean_triple(line)
                if triple is None:
                    continue
                output[m].append(triple)
                mapped = True
                
        # If no direct match, try entity matching or fallback to single mention
        if not mapped:
            triple = clean_triple(line)
            if triple is None:
                continue
                
            matched_m = get_matching_entity(triple[0], mention)
            if matched_m:
                # Found a matching entity in the subject position
                output[matched_m].append(triple)
            elif len(mention) == 1:
                # If there's only one mention, assign triple to it
                output[mention[0]].append(triple)

    return output


def add_type_triple(types, triples, mention):
    """
    Add 'instance of' triples based on identified entity types.
    
    Args:
        types: Dictionary mapping entity mentions to their types
        triples: Dictionary mapping entity mentions to their triples
        mention: List of entity mentions
        
    Returns:
        dict: Updated triples dictionary with added type information
    """
    for m in mention:
        entity_type = types.get(m, None)
        entity_triples = triples.get(m, None)
        
        # Skip if either type or triples are missing
        if entity_type is None or entity_triples is None:
            continue
            
        # Check if we already have a type triple
        existing_objects = [x[2] for x in entity_triples]
        if entity_type not in existing_objects:
            # Add a new 'instance of' triple
            entity_triples.append([m, 'instance of', entity_type])
            
    return triples


def clean_text(text: str) -> str:
    """
    Clean text by removing specific characters and extra whitespace.
    
    Args:
        text: The input text to clean
        
    Returns:
        str: Cleaned text or None if text is empty or placeholder
    """
    # Return None for empty or placeholder text
    if text is None or text.strip() == '...':
        return None
    
    # Remove specific characters
    chars_to_remove = ['[', ']', '(', ')', '"', '...', ':', '#', "\\_", '""']
    for char in chars_to_remove:
        text = text.replace(char, '')

    # Remove assumption text (partial information)
    if "(assum" in text:
        text = text.split("(assum")[0]

    # Remove list markers at start of text
    if text.startswith('- ') or text.startswith('* ') or text.startswith('**'):
        text = text[2:].strip()

    # Clean up any remaining quotes and whitespace
    return text.strip("'").strip('"').strip()

def normalize(text: str) -> str:
    """Normalize text by removing special chars, possessives, and @ mentions."""
    if not text:
        return ""
    return re.sub(r'[^\w\s]', ' ', text.lower().replace("'s", "")).replace('the ', '').strip()