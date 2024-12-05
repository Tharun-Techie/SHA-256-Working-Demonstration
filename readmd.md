# SHA-256 Encryption Demo

This is a simple web application built using Streamlit that demonstrates how to encrypt a message using the SHA-256 hashing algorithm. The application allows users to input a message and a private key to generate a hashed value. It also provides functionality to verify the hashed message by checking it against the original message and private key.

## Features

- **Message Encryption**: Input a message and a private key to generate its SHA-256 hash.
- **Message Verification**: Verify the hashed message by entering the original message and private key.
- **Session State Management**: The application uses session state to store user inputs and hashed values for verification.

## Requirements

To run this application, you need to have Python installed along with the following packages:

- Streamlit
- hashlib (part of Python's standard library)

You can install Streamlit using pip:

```bash
pip install streamlit