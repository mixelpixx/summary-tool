1. Clone this repository or download the source code.
2. Install the required dependencies:

## Usage

1. Run the application:
=======
1. Clone this repository or download the source code:
   ```
   git clone https://github.com/yourusername/advanced-file-summarizer.git
   cd advanced-file-summarizer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the root directory of the project
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. Run the application:
   ```
   python sumsum3.py
   ```
   
2. The Gradio interface will launch in your default web browser.

3. Use the interface to:
   - Upload a markdown or text file
   - Set the chunk size (default: 1000 words)
   - Set the overlap between chunks (default: 100 words)
   - Specify an output directory (optional)
   - Provide custom instructions for the AI (optional)

4. Click "Submit" to generate the summary.

5. The application will process the file and provide links to the generated summary files.

## Configuration

- **Chunk Size**: Adjust this to control the length of text segments processed by the AI. Larger chunks may provide more context but take longer to process.
- **Overlap**: This determines how much text is shared between adjacent chunks, helping to maintain context across chunk boundaries.
- **Custom Instructions**: Use this to guide the AI's summarization style or focus. For example, "Focus on technical details" or "Summarize in a casual tone".

## Troubleshooting

- If you encounter an API error, ensure your OpenAI API key is correctly set in the `.env` file.
- For "File not found" errors, check that you're running the script from the correct directory.
- If summaries are too long or short, try adjusting the chunk size and overlap settings.

## Contributing

Contributions to the Advanced File Summarizer are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact Information

For any questions or support, please contact [your email/contact information here].
