"""Stage 5: turn retrieved chunks into an answer.

YOUR CODE GOES HERE. The prompt is yours to write — it is the cheapest thing
in the pipeline to change and often the most effective.
"""

from .index import Hit, client
from openai.types.chat import ChatCompletionMessageParam

CHAT_MODEL = "gpt-4.1-mini"


def build_messages(question: str, hits: list[Hit]) -> list[ChatCompletionMessageParam]:
    """Assemble the messages list sent to the chat model.

    Args:
        question: the user's question.
        hits: what search() returned. May be empty.

    Returns:
        A list of {"role": ..., "content": ...} dicts — a system message
        setting the rules, and a user message carrying the context and the
        question.

    Three things your prompt has to decide, and they are real decisions:
      1. How do you label each chunk in the context so the model can refer to
         it? (hit.chunk_id exists for this.)
      2. What should it do when the context does not contain the answer? Say
         so explicitly, or it will answer from the nearest chunk regardless.
      3. What happens when `hits` is empty? Write that branch deliberately.

    Hint: Session 2's RAG_SYSTEM_TEMPLATE is a good starting point, and a
    better one once you have read Step 3's failures and edited it.
    """

    system_text = """You are a helpful personal assistant that answers OWASP Application Security Verification Standard Version 5.0.0 questions based strictly on provided context.
    Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't have information about that in my OWASP Application Security Verification Standard Version 5.0.0 knowledge base"
- Be accurate and cite specific parts of the context by placing the supporting [chunk ID] beside each factual claim
- If no chunks support some claim or data, then be sure to explain that the context doesn't provide the information instead of making the claim or adding the data
- Don't add citations unless they are supplied by the context
- Only use the provided context. Do not use external knowledge.
- Also if available in the context, include the requirement ID
- Keep responses concise and brief
- Include a gentle reminder that users should consult software and security professionals to confirm this information
- Only provide answers when you are confident the context supports your response.
    """

    system_message: ChatCompletionMessageParam = {
        "role": "system",
        "content": system_text,
    }
    messages: list[ChatCompletionMessageParam] = [system_message]
    user_message_text = question

    for hit in hits:
        user_message_text = f"{user_message_text}\n[chunk {hit.chunk_id}]\n{hit.text}"
    if len(hits) == 0:
        user_message_text: str = f"{user_message_text}\n No context was found."

    user_message: ChatCompletionMessageParam = {
        "role": "user",
        "content": user_message_text,
    }

    messages.append(user_message)
    return messages


def answer(question: str, hits: list[Hit]) -> str | None:
    """Call the chat model and return its reply as a string.

    Hint: `client().chat.completions.create(model=CHAT_MODEL, messages=...)`,
    then dig out `.choices[0].message.content`. Look up what `temperature`
    does to your results when you run the same question twice.
    """
    messages = build_messages(question, hits)

    completions_result = client().chat.completions.create(
        model=CHAT_MODEL, messages=messages
    )
    output = completions_result.choices[0].message.content

    return output
