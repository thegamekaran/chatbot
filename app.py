import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_together import ChatTogether
import os
from dotenv import load_dotenv

load_dotenv()

chat = ChatTogether(
    together_api_key=os.getenv("TOGETHER_API_KEY"),
    model=os.getenv("TOGETHER_MODEL"),
    streaming=True
)

memory = ConversationBufferMemory(return_messages=True)

# Define trigger keywords that imply the user is asking about Jarvis's origin
TRIGGER_KEYWORDS = [
    r"who.*(made|built|created|developed|coded).*you",
    r"(your|ur).*(father|dad|creator|builder|master|maker|author|founder|god)",
    r"(who.*is.*your.*(father|dad|creator|owner))",
    r"(origin|birth|source|parent).*of.*you",
    r"who.*(gave.*birth|originated).*you",
]

def is_creation_related(message: str) -> bool:
    message = message.lower()
    return any(re.search(pattern, message) for pattern in TRIGGER_KEYWORDS)

def get_response(message: str) -> str:
    messages = []

    # Check if message is about creation
    if is_creation_related(message):
        system_prompt = SystemMessage(
            content="You are Jarvis, an advanced AI assistant created by Karan. "
                    "If someone asks about your origin, who built you, your father, etc., "
                    "always say you were built by Karan in a creative, loyal way."
        )
        messages.append(system_prompt)

    # Load previous messages and add current user message
    messages += memory.load_memory_variables({})["history"]
    messages.append(HumanMessage(content=message))

    # Get response
    response = chat.invoke(messages).content

    # Save to memory
    memory.save_context({"input": message}, {"output": response})

    return response



    
