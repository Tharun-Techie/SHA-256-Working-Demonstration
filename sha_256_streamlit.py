import streamlit as st
import hashlib

def hash_message(message, private_key):
    # Combine message and private key
    combined = f"{message}{private_key}"
    return hashlib.sha256(combined.encode()).hexdigest()

def verify_message(message, private_key, hashed_message):
    # Combine message and private key for verification
    return hash_message(message, private_key) == hashed_message

def main():
    st.title("SHA-256 Encryption Demo")

    # Input for message to encrypt
    message = st.text_input("Enter message to encrypt:")
    
    # Input for private key
    private_key = st.text_input("Enter your private key:", type="password")
    
    # Hash the message when button is clicked
    if st.button("Encrypt Message"):
        if message and private_key:
            hashed = hash_message(message, private_key)
            st.success(f"Encrypted Message (SHA-256): {hashed}")
            st.session_state.hashed_message = hashed  # Store hashed message in session state
            st.session_state.original_message = message  # Store original message in session state
            st.session_state.private_key = private_key  # Store private key in session state

    # Input for decryption (verification)
    if 'hashed_message' in st.session_state:
        verification_hashed_value = st.text_input("Enter hashed value for verification:")
        
        # Input for private key during decryption
        verification_private_key = st.text_input("Enter your private key for verification:", type="password")
        
        if st.button("Decrypt Message"):
            if verification_hashed_value and verification_private_key:
                # Try to find the original message by checking all possible messages
                found = False
                for possible_message in [st.session_state.original_message]:  # Here we assume we only have one original message for simplicity
                    if verify_message(possible_message, verification_private_key, verification_hashed_value):
                        st.success(f"Decrypted Message matches: {possible_message}")
                        found = True
                        break
                
                if not found:
                    st.error("Decrypted Message does not match.")

if __name__ == "__main__":
    main()