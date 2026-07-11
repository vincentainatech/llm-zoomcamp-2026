# System-level instructions given to the LLM — defines its role and behavior.
# This tells the model to answer strictly using the provided context,
# and to admit when it doesn't know rather than hallucinating.

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

# Template used to format the final prompt sent to the LLM.
# {question} and {context} are placeholders filled in later via .format().

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


# A base class implementing a simple Retrieval-Augmented Generation (RAG) pipeline:
# 1. Search a document index for relevant content.
# 2. Build a prompt combining the question + retrieved context.
# 3. Send that prompt to an LLM and return its answer.

class RAGBase:

    # Defines the constructor — a special method that runs automatically every time you create a new RAGBase object.
    def __init__(
        self,  # refers to the new object being created
        index,  # the search index object (e.g. a text/vector search engine)
        llm_client,  # client used to call the LLM API (e.g. OpenAI client)
        instructions=INSTRUCTIONS,  # system instructions for the LLM
        prompt_template=PROMPT_TEMPLATE,  # template for formatting question+context
        course="llm-zoomcamp",  # default course filter for search
        model="gpt-5.4-mini"  # default LLM model to use
    ):
        # Takes the values that are passed in, and stores them as attributes on this specific object (self).
        # This means the values "stick" to the object permanently so other methods (search, llm, rag) can access them via self.
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    # The search method delegates to the index:
    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    # The build_context and build_prompt methods format the search results:
    def build_context(self, search_results):
        # Convert the list of search result documents into a single
        # plain-text block, formatted as Q&A pairs grouped by section.
        # This text becomes the 'CONTEXT' the LLM will read.

        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    # Build the final prompt text by inserting the user's question
    # and the formatted context into the prompt template
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    # The llm method sends the prompt to the LLM:
    # - 'developer' role message carries the system-level instructions.
    # - 'user' role message carries the actual question+context prompt.
    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    # The main entry point: runs the full RAG pipeline end-to-end.
    # 1. Search for relevant documents.
    # 2. Build a prompt from the question + retrieved context.
    # 3. Get the LLM's answer based on that prompt.
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer