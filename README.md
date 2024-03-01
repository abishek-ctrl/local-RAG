# LLM x Website URLs

Most of the language models like GPT 4 and even Gemini Pro have the issue of not having the latest information considering they were trained with a set of data for pretraining.

In cases ,We may need to access real-time information from websites to get up-to-date information. LLMs do not have this functionality by default. But through Retrieval Augmented Generation(RAG), we can allow them to access the data that we provide. 

Here, we use Langchain's Webloader and get the data from the site and store the data into a vectorstore, from where we can get the necessary answers for our queries. 

## Usage Instructions:

1. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Application:**
    ```bash
    python main.py
    ```

3. **Access the Gradio Interface:**
   - Open your web browser and navigate to the provided URL.

4. **Input Parameters:**
   - Query: Enter your question or input.
   - URL: Provide the website URL for context.
   - OpenAI Key: Enter your OpenAI key for authentication.

5. **Interact with the Chatbot:**
   - Receive responses based on your input and website context.

## Code Overview:

- **main.py:** Python script containing the main application code.
- **Functions:**
    - `getvecstore(url)`: Retrieves a vector store from the given website URL.
    - `getcontext(vector_store)`: Obtains a context-aware retriever chain.
    - `getragchain(retriever_chain)`: Retrieves a conversational RAG (Retrieval-Augmented Generation) chain.
    - `getresp(user_input, website_url)`: Gets a response from the chatbot based on user input and website URL.
    - `interact(Query, URL, OpenAI_Key)`: Function to interact with the chatbot through Gradio interface.

## Future Improvements:

For future enhancements, I am considering incorporating a complete local and private workflow using the following technologies:

- **Ollama for Embeddings:** Explore the use of Ollama for embeddings to enhance local processing and privacy.
- **LLM (Language Model):** The use of local LLM inference through LM Studio and Ollama can be added to make it completely private to our system alone.

## Important Note:

- Keep your OpenAI key secure and avoid sharing it publicly.
- Implement additional security measures for handling sensitive information.

Feel free to explore and contribute to the codebase for further improvements and customization.
