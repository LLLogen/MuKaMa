def tokenize_and_insert_random_char(input_text, tokenizer):
    tokenized_output = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
    tokenized_ids = tokenized_output["input_ids"][0]
    tokens = [tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in tokenized_ids]
    
    try:
        val_index = tokens.index('val')
    except ValueError:
        val_index = -1  

    if val_index != -1:
        random_char = random.choice(string.ascii_letters)  
        tokens.insert(val_index + 1, random_char)

    modified_text = tokenizer.convert_tokens_to_string(tokens)

    return modified_text