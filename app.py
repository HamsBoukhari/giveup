# Importing necessary libraries
import streamlit as st  # Streamlit for creating a web-based interface
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Libraries for tokenizing and GPT-2 model
import torch  # PyTorch for machine learning model operations
import re  # Regular expressions for pattern matching and text manipulation
import xmltodict  # For parsing and un-parsing XML data
import xml.etree.ElementTree as ET  # XML parsing library to work with XML trees

# This function initializes and caches the tokenizer and model, allowing mutations to the cached data.
@st.cache(allow_output_mutation=True)
def get_model():
    # Load the tokenizer for the distilgpt2 model
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    # Set pad_token to be the end-of-sequence (EOS) token
    tokenizer.pad_token = tokenizer.eos_token
    # Load a pre-trained GPT2 model from a specific path (hams2/giveup)
    model = GPT2LMHeadModel.from_pretrained("hams2/giveup")
    return tokenizer, model

# Retrieve the cached tokenizer and model
tokenizer, model = get_model()

# Title for the Streamlit app
st.title('Give Up')

# User input areas for trade message and SGW operation 
trade_msg = st.text_area('Trade Message')  # Text area for trade message input
sgw_op = st.text_area('SGW Operation')  # Text area for SGW operation input
# Button to trigger the prediction
button = st.button("Predict")

# Check if both text areas have input and if the button was pressed
if (trade_msg and sgw_op) and button:
    try:
        # Replace newlines in the inputs with empty strings
        trade_msg1 = trade_msg.replace('\n', '')
        sgw_op1 = sgw_op.replace('\n', '')

        # Parse the SGW operation XML and remove specific elements
        root = ET.fromstring(sgw_op)  # Create an XML tree from the SGW operation
        hdr = root.find('Hdr')  # Find the 'Hdr' (Header) element
        if hdr is not None:
            root.remove(hdr)  # Remove the header if it exists
        
        instrmt = root.find('Instrmt')  # Find the 'Instrmt' (Instrument) element
        if instrmt is not None:
            root.remove(instrmt)  # Remove the instrument if it exists
        
        # Find and remove the first three 'Pty' (Party) elements, if they exist
        pty_elements = root.findall('Pty')
        for i in range(3):
            if i < len(pty_elements):
                root.remove(pty_elements[i])
        
        # Convert the modified XML tree back into a string
        ex_sgw_op = ET.tostring(root, encoding='utf-8', method='xml').decode()
        
        # Combine the cleaned trade message with the modified SGW operation XML
        input_seq = trade_msg1 + ' ' + ex_sgw_op
        # Remove certain special characters (<, \ ,", /) from the sequence
        pattern = r'[<\"/]'
        cleaned_seq1 = re.sub(pattern, '', input_seq)
        
        # Replace ">" with a space
        seq = cleaned_seq1.replace('>', ' ')
        
        # Put the model into evaluation mode
        model.eval()
        
        # Encode the sequence into input IDs for the pre-trained GPT-2 model
        input_ids = tokenizer.encode(seq, return_tensors='pt')
        
        # Generate output from the pre-trained GPT-2 model, setting a maximum length and specifying padding
        output = model.generate(
            input_ids, 
            max_length=1000, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id  # Pad with the EOS token
        )
        
        # Decode the generated output into text, skipping special tokens
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the original input sequence from the beginning of the generated output
        output_sequence = decoded_output[len(seq):].strip()
        
        # Parse the SGW message and the generated output into dictionaries using xmltodict
        sgw_op_dict = xmltodict.parse(sgw_op)  # Original SGW message
        output_sequence_dict = xmltodict.parse(output_sequence)  # Generated output
        
        # Ensure certain attributes are consistent between the original and generated XML
        grp_id1 = sgw_op_dict['AllocInstrctn']['@GrpID']  # Original Group ID
        grp_id2 = output_sequence_dict['AllocRpt']['@GrpID']  # Generated Group ID
        if grp_id1 != grp_id2:  # If Group IDs do not match, correct it
            output_sequence_dict['AllocRpt']['@GrpID'] = grp_id1
        
        # Ensure consistent 'ID' in the XML, correcting for escape characters
        id1 = xmltodict.parse(sgw_op)['AllocInstrctn']['@ID']  # Original 'ID'
        id2 = id1.replace(">", "&gt;")  # Correct escape characters
        output_sequence_dict['AllocRpt']['@ID'] = id2  # Set the corrected 'ID' in the generated XML
        
        # Unparse the dictionary back to XML and correct any misrepresentations
        modified_ccp_message = xmltodict.unparse(output_sequence_dict, pretty=False)
        
        # If the XML message starts with an XML declaration, remove it
        if modified_ccp_message.startswith("<?xml"):
            modified_ccp_message = modified_ccp_message.split("?>", 1)[1].strip()
        
        # Remove newline and tab characters from the XML message
        modified_ccp_message = modified_ccp_message.replace("\n", "").replace("\t", "")
        
        # Display the final modified message in the Streamlit app
        st.write("CCP Message: ", modified_ccp_message)  # Output the final message to the user

    except Exception as e:
        # If an error occurs, display an error message in the Streamlit app
        st.write("There is an error in generating CCP Message. Please verify the Trade Message and the SGW Operation.")
        
