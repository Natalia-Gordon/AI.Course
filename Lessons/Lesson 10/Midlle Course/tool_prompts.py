from langchain.prompts import PromptTemplate

# === Tools PROMPTS ===
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following document section. Focus on key events, decisions, and include any available time-based details.

Text:
{text}

Summary with timeline:
"""
)

combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Combine the following summaries into a cohesive document-level summary.
Ensure all chronological events are preserved and logically ordered.

Summaries:
{text}

Combined summary with timeline:
"""
)

# === REFINE PROMPTS ===

refine_initial_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Write an initial summary of the following document section.
Highlight key points and extract any time-based or chronological events if available.

Text:
{text}

Initial summary with timeline:
"""
)

refine_update_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template="""
The current summary is:
----------------
{existing_answer}
----------------

Given the following new section of the document:
----------------
{text}
----------------

Update the summary by:
- Adding important new information
- Including additional events in chronological order
- Keeping the format clear and ordered

Refined summary with updated timeline:
"""
)